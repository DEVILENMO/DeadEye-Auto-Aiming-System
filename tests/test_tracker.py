# coding: utf-8
"""TargetTracker 单元测试：匹配门控、生命周期、KF 平滑与 ID 稳定性（纯 CPU）。"""
import numpy as np
import pytest

from deadeye.core.target_tracker import TargetTracker, iou_matrix
from deadeye.utils.math_tools import BoxKalmanFilter


def det(label, x1, y1, x2, y2):
    return [label, (x1, y1), (x2, y2)]


# ---------------------------------------------------------------- IoU

def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float64)
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(25 / 175)
    assert iou[0, 2] == pytest.approx(0.0)


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((2, 4))).shape == (0, 2)


# ---------------------------------------------------------------- 匹配门控

def test_class_gate_prevents_cross_label_match():
    """回归：旧实现跨类别代价为 0，会被优先匹配；现在必须禁止。"""
    tracker = TargetTracker(match_iou_threshold=0.1, max_missed_frames=0)
    tracker.update([det('enemy', 0, 0, 100, 100)])
    # 完全同位置的 ally 检测不得继承 enemy 的 ID；enemy 漏检即过期
    targets = tracker.update([det('ally', 0, 0, 100, 100)])
    assert [t.label for t in targets] == ['ally']
    assert targets[0].index != 0  # 新 ID，不是旧的 0


def test_iou_gate_prevents_distant_match():
    tracker = TargetTracker(match_iou_threshold=0.3)
    tracker.update([det('enemy', 0, 0, 100, 100)])
    # 远处出现的新目标不应吃掉旧目标的 ID
    targets = tracker.update([det('enemy', 400, 400, 500, 500)])
    assert len(targets) == 2


def test_same_target_keeps_id_on_overlap():
    tracker = TargetTracker()
    tracker.update([det('enemy', 0, 0, 100, 100)])
    targets = tracker.update([det('enemy', 5, 5, 105, 105)])
    assert len(targets) == 1
    assert targets[0].index == 0


# ---------------------------------------------------------------- 生命周期

def test_missed_frames_keep_target_with_same_id():
    """核心需求：漏检数帧内目标保留（KF 预测），恢复后 ID 不变。"""
    tracker = TargetTracker(max_missed_frames=3)
    tracker.update([det('enemy', 0, 0, 100, 100)])
    # 连续 3 帧漏检：目标仍在且位置由 KF 预测推进
    for _ in range(3):
        targets = tracker.update([])
        assert len(targets) == 1
        assert targets[0].index == 0
    moved_x = targets[0].right_bottom[0]
    assert moved_x >= 100  # 无速度先验时位置不回退
    # 第 4 帧漏检：超过 max_missed_frames=3，删除
    targets = tracker.update([])
    assert len(targets) == 0


def test_recovery_after_miss_keeps_id():
    tracker = TargetTracker(max_missed_frames=5)
    tracker.update([det('enemy', 0, 0, 100, 100)])
    tracker.update([])  # 漏检一帧
    targets = tracker.update([det('enemy', 8, 8, 108, 108)])  # 恢复
    assert len(targets) == 1
    assert targets[0].index == 0  # 仍是同一个目标，没有生成新 ID


def test_min_hits_to_confirm_delays_target():
    """连续命中 M 帧才确认：抑制单帧误检。"""
    tracker = TargetTracker(min_hits_to_confirm=2)
    targets = tracker.update([det('enemy', 0, 0, 100, 100)])
    assert len(targets) == 0  # 第 1 帧不确认
    targets = tracker.update([det('enemy', 2, 2, 102, 102)])
    assert len(targets) == 1  # 第 2 帧确认
    assert targets[0].hits >= 2


def test_single_frame_false_positive_never_confirms():
    tracker = TargetTracker(min_hits_to_confirm=2)
    tracker.update([det('enemy', 0, 0, 100, 100)])  # 单帧误检
    targets = tracker.update([])  # 下帧消失
    assert len(targets) == 0


def test_min_hits_1_confirms_immediately():
    tracker = TargetTracker(min_hits_to_confirm=1)
    targets = tracker.update([det('enemy', 0, 0, 100, 100)])
    assert len(targets) == 1
    assert targets[0].index == 0


# ---------------------------------------------------------------- ID 稳定性

def test_crossing_targets_keep_ids():
    """两个目标交错运动：ID 不交换（KF 速度先验的功劳）。"""
    tracker = TargetTracker(match_iou_threshold=0.1)
    # A 向右移动，B 向左移动，10 帧后交错
    for step in range(12):
        ax = step * 20
        bx = 500 - step * 20
        targets = tracker.update([
            det('enemy', ax, 0, ax + 60, 60),
            det('enemy', bx, 0, bx + 60, 60),
        ])
    indices = sorted(t.index for t in targets)
    assert len(targets) == 2
    assert indices == [0, 1]


def test_moving_target_tracking_smooths_noise():
    """KF 对观测噪声的平滑：位置估计应明显比原始观测平滑。"""
    rng = np.random.default_rng(7)
    tracker = TargetTracker()
    truth_x = 0.0
    estimates = []
    for _ in range(40):
        truth_x += 5.0
        noisy_x1 = truth_x + rng.normal(0, 8.0)
        targets = tracker.update([det('enemy', noisy_x1, 0, noisy_x1 + 50, 50)])
        estimates.append(targets[0].left_top[0])
    # 后半段估计的相邻帧差应小于噪声标准差
    diffs = np.abs(np.diff(estimates[20:]))
    assert diffs.mean() < 8.0
    # 跟上真值（不滞后太多）
    assert abs(estimates[-1] - truth_x) < 15.0


# ---------------------------------------------------------------- KF

def test_box_kalman_filter_converges_to_constant_velocity():
    kf = BoxKalmanFilter(0, 0, 50, 50)
    for i in range(1, 30):
        cx = i * 10.0
        kf.predict()
        kf.update(cx, 0, 50, 50)
    cx, cy, w, h = kf.predict()
    assert cx == pytest.approx(300.0, abs=5.0)  # 速度已学到 ~10/帧
    assert w == pytest.approx(50.0, abs=1.0)
