# coding: utf-8
# cython: language_level=3
"""
多目标追踪器：KF 预测 + 匈牙利匹配 + 可配置生命周期。

相对旧版 DeadEyeCore 内置实现的改进：
- 代价从「曼哈顿角点距离」改为 IoU 代价（尺度无关，且修复了旧版跨类别代价为 0
  导致求解器*优先*跨类别匹配的反向 bug）；
- 类别门控 + IoU 门控：不满足门控的目标对不参与匹配；
- 匹配用 KF 预测框而非上一帧观测框，运动目标帧间关联更稳；
- 生命周期可配置：漏检 N 帧内目标保留（KF 预测推进，ID 不变），连续命中 M 帧
  才确认为正式目标（抑制单帧误检），解决「一帧漏检就换新 ID」的问题。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from deadeye.base.modules import Target

# 门控外的代价（线性分配中视为「不可行」；求解后仍需显式过滤）
_GATED_COST = 1e6


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """两组 xyxy 框的 IoU 矩阵，shape (N, M)。"""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class TargetTracker:
    """
    Parameters
    ----------
    match_iou_threshold : 匹配门控的最小 IoU（低于此值的目标对不参与关联）
    max_missed_frames : 目标连续漏检多少帧后删除（期间由 KF 预测保持，ID 不变）
    min_hits_to_confirm : 新目标连续命中多少帧后确认为正式目标（1 = 立即确认）
    """

    def __init__(
        self,
        *,
        match_iou_threshold: float = 0.3,
        max_missed_frames: int = 5,
        min_hits_to_confirm: int = 1,
    ):
        self.match_iou_threshold = float(match_iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits_to_confirm = max(1, int(min_hits_to_confirm))

        self.targets: list[Target] = []      # 已确认目标（输出给执行模块）
        self._tentative: list[Target] = []   # 待确认目标（未达到 min_hits）
        self._next_index = 0
        self._tentative_index = -1

    # -------------------------------------------------- 主入口

    def update(self, detections: list) -> list[Target]:
        """
        输入一帧检测结果 ``[label, (x1, y1), (x2, y2)]`` 列表，
        返回确认目标列表（同一 list 对象，供执行模块读取）。
        """
        det_labels = [d[0] for d in detections]
        det_boxes = np.array(
            [[d[1][0], d[1][1], d[2][0], d[2][1]] for d in detections],
            dtype=np.float64,
        ).reshape(-1, 4)

        # 1) 所有目标先做 KF 先验预测（匹配用预测框；漏检目标由此推进）
        for target in self.targets + self._tentative:
            target.predict()

        # 2) 已确认目标关联
        matched_det_ids = self._associate(self.targets, det_boxes, det_labels)

        # 3) 未命中的确认目标：保留至 max_missed_frames
        self.targets = [t for t in self.targets if t.missed_frames <= self.max_missed_frames]

        # 4) 剩余检测与待确认目标关联（tentative 一帧未命中即淘汰）
        unmatched = [j for j in range(len(detections)) if j not in matched_det_ids]
        matched_det_ids |= self._associate(
            self._tentative,
            det_boxes[unmatched] if unmatched else det_boxes[:0],
            [det_labels[j] for j in unmatched],
            detection_ids=unmatched,
        )
        self._tentative = [t for t in self._tentative if t.missed_frames == 0]

        # 5) 达到 min_hits 的升级为确认目标
        promoted = [t for t in self._tentative if t.hits >= self.min_hits_to_confirm]
        if promoted:
            for target in promoted:
                target.index = self._next_index
                self._next_index += 1
            self._tentative = [t for t in self._tentative if t.hits < self.min_hits_to_confirm]
            self.targets.extend(promoted)

        # 6) 全新检测进入待确认队列
        for j in range(len(detections)):
            if j in matched_det_ids:
                continue
            label, left_top, right_bottom = detections[j]
            target = Target(label, self._tentative_index, left_top, right_bottom)
            self._tentative_index -= 1
            # min_hits=1 时立即升级，保持与旧行为一致（单帧确认、ID 即分配）
            if self.min_hits_to_confirm <= 1:
                target.index = self._next_index
                self._next_index += 1
                self.targets.append(target)
            else:
                self._tentative.append(target)

        return self.targets

    # -------------------------------------------------- 关联

    def _associate(
        self,
        target_list: list[Target],
        det_boxes: np.ndarray,
        det_labels: list,
        detection_ids: list[int] | None = None,
    ) -> set[int]:
        """
        匈牙利匹配：IoU 代价 + 类别/IoU 双门控。
        命中的目标就地 update_position；未命中的保持 predict 状态（missed_frames 已 +1）。
        返回命中的检测索引集合（若传了 detection_ids 则映射回原始索引）。
        """
        if detection_ids is None:
            detection_ids = list(range(len(det_labels)))

        if not target_list or det_boxes.shape[0] == 0:
            return set()

        pred_boxes = np.array(
            [[*t.left_top, *t.right_bottom] for t in target_list], dtype=np.float64
        )
        iou = iou_matrix(pred_boxes, det_boxes)

        cost = 1.0 - iou
        # 类别门控 + IoU 门控 -> 不可行
        label_mismatch = np.array(
            [[t.label != l for l in det_labels] for t in target_list], dtype=bool
        )
        cost[label_mismatch | (iou < self.match_iou_threshold)] = _GATED_COST

        row_ind, col_ind = linear_sum_assignment(cost)
        matched: set[int] = set()
        for ti, di in zip(row_ind, col_ind):
            if cost[ti, di] >= _GATED_COST:
                continue  # 求解器为凑满行列可能选中门控对，显式丢弃
            det = det_boxes[di]
            target_list[ti].update_position(
                (int(det[0]), int(det[1])), (int(det[2]), int(det[3]))
            )
            matched.add(detection_ids[di])
        return matched
