# coding: utf-8
"""AimLab 颜色检测模块测试：合成图 + 真实截图回归 + 速度。"""
import cv2
import numpy as np
import pytest

from deadeye.detection_module.aimlab_color_detector import AimLabColorDetector


def make_scene(*circles, size=(320, 320)):
    """灰白棋盘格背景 + 彩色圆（模拟 AimLab 场景），circles: (cx, cy, r, rgb)"""
    img = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    img[::16, ::8] = 110  # 棋盘格纹理
    for cx, cy, r, rgb in circles:
        cv2.circle(img, (cx, cy), r, rgb, -1)
    return img


CYAN = (60, 230, 240)
YELLOW = (240, 230, 60)
GREEN = (60, 220, 60)  # AimLab 场景里的绿色横条，应被排除


def test_single_cyan_ball_center_and_box():
    det = AimLabColorDetector()
    img = make_scene((160, 160, 50, CYAN))
    targets = det.target_detect(img)
    assert len(targets) == 1
    label, (x1, y1), (x2, y2) = targets[0]
    assert label == 'ball'
    assert abs((x1 + x2) / 2 - 160) <= 2
    assert abs((y1 + y2) / 2 - 160) <= 2
    assert 90 <= (x2 - x1) <= 110  # 半径 50 的圆外接框约 100


def test_multiple_balls_both_colors():
    det = AimLabColorDetector()
    img = make_scene((80, 80, 30, CYAN), (240, 160, 40, YELLOW), (160, 260, 25, CYAN))
    targets = det.target_detect(img)
    assert len(targets) == 3


def test_green_bar_excluded():
    det = AimLabColorDetector()
    img = make_scene((160, 160, 50, CYAN))
    cv2.rectangle(img, (60, 30), (260, 50), GREEN, -1)  # 绿色横条
    targets = det.target_detect(img)
    assert len(targets) == 1  # 只有球，绿条被色相区间排除


def test_glow_strips_rejected_by_circularity():
    """辉光/反光类条状连通域应被圆形度过滤剔除。"""
    det = AimLabColorDetector()
    img = make_scene((160, 160, 50, CYAN))
    cv2.rectangle(img, (80, 260), (240, 285), CYAN, -1)   # 地面反光条（同色）
    cv2.rectangle(img, (70, 40), (250, 60), CYAN, -1)     # 上方光晕条（同色）
    targets = det.target_detect(img)
    assert len(targets) == 1


def test_gray_background_never_detected():
    det = AimLabColorDetector()
    img = make_scene()
    assert det.target_detect(img) == []


def test_color_toggles():
    img = make_scene((160, 160, 40, CYAN), (160, 160, 0, YELLOW))
    img = make_scene((80, 160, 40, CYAN), (240, 160, 40, YELLOW))
    cyan_only = AimLabColorDetector(detect_yellow=False)
    assert len(cyan_only.target_detect(img)) == 1
    yellow_only = AimLabColorDetector(detect_cyan=False)
    assert len(yellow_only.target_detect(img)) == 1


def test_both_disabled_raises():
    with pytest.raises(ValueError):
        AimLabColorDetector(detect_cyan=False, detect_yellow=False)


def test_min_area_filters_small_blobs():
    det = AimLabColorDetector(min_area_px=5000)
    img = make_scene((160, 160, 30, CYAN))  # 半径 30，面积约 2800
    assert det.target_detect(img) == []


# ------------------------------------------------ 真实截图回归

def test_real_screenshot_single_ball():
    det = AimLabColorDetector()
    for name in ('aimlab_cyan_ball.png', 'aimlab_yellow_ball.png'):
        img = cv2.cvtColor(cv2.imread(f'res/images/{name}'), cv2.COLOR_BGR2RGB)
        assert img is not None, f'缺少测试资源 res/images/{name}'
        targets = det.target_detect(img)
        assert len(targets) == 1, f'{name}: 应精确检出 1 个球，实际 {len(targets)}'
        _, (x1, y1), (x2, y2) = targets[0]
        w, h = x2 - x1, y2 - y1
        assert max(w, h) / min(w, h) <= 1.2  # 球的框应接近正方形


# ------------------------------------------------ 速度

def test_speed_under_5ms_at_640():
    det = AimLabColorDetector()
    img = make_scene((100, 100, 40, CYAN), (220, 220, 40, YELLOW), size=(640, 640))
    for _ in range(30):
        det.target_detect(img)
    assert det.last_detect_ms < 5.0, f'640x640 单帧耗时 {det.last_detect_ms:.2f}ms 超标'
