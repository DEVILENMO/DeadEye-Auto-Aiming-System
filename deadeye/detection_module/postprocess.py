# coding: utf-8
# cython: language_level=3
"""
统一的目标检测后处理：支持 YOLO 系列三种输出布局的自动识别与解码。

输出布局与代表版本：
- ``v5_v7``  : [batch, anchors, 5 + nc]  —— cx,cy,w,h,obj,cls...（YOLOv5 / YOLOv7 导出图内含 sigmoid）
- ``v8``     : [batch, 4 + nc, anchors]  —— cx,cy,w,h,cls...（YOLOv8 / YOLOv11 / YOLOv12，无 objectness）
- ``end2end``: [batch, max_det, 6]       —— x1,y1,x2,y2,score,cls（YOLOv10 / YOLO26 / 带 NMS 的导出，免后处理 NMS）

``auto`` 模式优先使用模型导出时写入的 Ultralytics 元数据（end2end 标记），
缺失时按输出张量形状启发式判断。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deadeye.detection_module.preprocess import LetterboxInfo

# 启发式判断时认为端到端模型的最大候选框数（YOLOv10/v26 常见为 300）
_E2E_MAX_CANDIDATES = 1024


@dataclass(frozen=True)
class Detection:
    """原始检测结果（原图坐标系）。"""

    box: np.ndarray  # [4] float32 xyxy
    score: float
    class_id: int


def detect_output_layout(output: np.ndarray, end2end_meta: bool | None = None) -> str:
    """
    根据输出张量形状与元数据判断输出布局。

    Parameters
    ----------
    output : 去掉 batch 维后的二维输出张量
    end2end_meta : 模型元数据中的 end2end 标记（None 表示未知）
    """
    if output.ndim != 2:
        raise ValueError(f'期望二维输出张量（去掉 batch 维），收到 shape={output.shape}')
    rows, cols = output.shape

    if end2end_meta is True:
        return 'end2end'
    if end2end_meta is False:
        # Ultralytics 导出（nms=False）恒为 channels-first 解码头，与类别数无关
        # （COCO 80 类的 [4+80, anchors] 不能按行数启发式误判为 v5/v7）
        return 'v8'

    # 无元数据时按形状启发式：anchor 维是较大的那一维
    if cols == 6 and rows <= _E2E_MAX_CANDIDATES:
        return 'end2end'
    if rows < cols and rows <= 4 + 200:
        return 'v8'
    if cols < rows and cols <= 5 + 200:
        return 'v5_v7'
    raise ValueError(f'无法识别的输出张量布局: shape={output.shape}')


def decode_detections(
    output: np.ndarray,
    info: LetterboxInfo,
    *,
    conf_thres: float = 0.45,
    iou_thres: float = 0.5,
    max_dets: int = 300,
    layout: str = 'auto',
    end2end_meta: bool | None = None,
) -> np.ndarray:
    """
    把网络原始输出解码为原图坐标系的检测框数组。

    Returns
    -------
    np.ndarray, shape [n, 6]，每行 x1, y1, x2, y2, score, class_id（float32）
    """
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if layout == 'auto':
        layout = detect_output_layout(output, end2end_meta)

    if layout == 'end2end':
        dets = _decode_end2end(output)
        need_nms = False
    elif layout == 'v8':
        dets = _decode_v8(output)
        need_nms = True
    elif layout == 'v5_v7':
        dets = _decode_v5_v7(output)
        need_nms = True
    else:
        raise ValueError(f'未知输出布局: {layout}')

    if dets.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # 置信度过滤
    keep = dets[:, 4] >= conf_thres
    dets = dets[keep]
    if dets.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if need_nms:
        dets = multiclass_nms(dets[:, :4], dets[:, 4], dets[:, 5].astype(np.int32), iou_thres, max_dets)
        if dets.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

    # 限制数量（端到端输出图内已按分数排序，直接截断）
    if dets.shape[0] > max_dets:
        order = np.argsort(-dets[:, 4])[:max_dets]
        dets = dets[order]

    # 模型输入坐标 -> 原图坐标
    boxes = dets[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
    dets[:, :4] = boxes
    return dets.astype(np.float32)


def _decode_v8(pred: np.ndarray) -> np.ndarray:
    """[4+nc, anchors] -> [n, 6]（模型输入坐标，未过滤）。导出图内已包含 sigmoid。"""
    pred = pred.T  # [anchors, 4+nc]
    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] * 0.5
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] * 0.5

    out = np.zeros((pred.shape[0], 6), dtype=np.float32)
    out[:, :4] = boxes
    out[:, 4] = scores
    out[:, 5] = class_ids
    return out


def _decode_v5_v7(pred: np.ndarray) -> np.ndarray:
    """[anchors, 5+nc] -> [n, 6]（模型输入坐标，未过滤）。conf = obj * cls。"""
    boxes_cxcywh = pred[:, :4]
    obj = pred[:, 4:5]
    class_scores = obj * pred[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] * 0.5
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] * 0.5

    out = np.zeros((pred.shape[0], 6), dtype=np.float32)
    out[:, :4] = boxes
    out[:, 4] = scores
    out[:, 5] = class_ids
    return out


def _decode_end2end(pred: np.ndarray) -> np.ndarray:
    """[max_det, 6] (x1,y1,x2,y2,score,cls) -> [n, 6]（模型输入坐标，未过滤）。"""
    out = np.asarray(pred, dtype=np.float32)
    if out.shape[1] == 6:  # 常见：x1,y1,x2,y2,score,cls
        return out.copy()
    if out.shape[1] == 7:  # 少数导出附带 batch index
        return out[:, 1:].copy()
    raise ValueError(f'端到端输出列数异常: {out.shape}')


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """单类 NMS（向量化 IoU 计算），返回保留的索引（按分数降序）。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thres]
    return keep


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thres: float,
    max_dets: int,
) -> np.ndarray:
    """按类别做 NMS，返回 [n, 6]。"""
    result: list[np.ndarray] = []
    for class_id in np.unique(class_ids):
        mask = class_ids == class_id
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        keep = nms(cls_boxes, cls_scores, iou_thres)
        if not keep:
            continue
        kept = np.empty((len(keep), 6), dtype=np.float32)
        kept[:, :4] = cls_boxes[keep]
        kept[:, 4] = cls_scores[keep]
        kept[:, 5] = class_id
        result.append(kept)
    if not result:
        return np.zeros((0, 6), dtype=np.float32)
    dets = np.concatenate(result, axis=0)
    if dets.shape[0] > max_dets:
        order = np.argsort(-dets[:, 4])[:max_dets]
        dets = dets[order]
    return dets


def format_targets(dets: np.ndarray, class_names: dict[int, str] | list[str]) -> list:
    """
    转换为流水线约定的目标格式：
    ``[class_name, (x1, y1), (x2, y2)]``
    """
    if isinstance(class_names, dict):
        name_of = lambda i: class_names.get(int(i), str(int(i)))
    else:
        name_of = lambda i: class_names[int(i)] if int(i) < len(class_names) else str(int(i))

    targets = []
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        targets.append([name_of(det[5]), (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))])
    return targets
