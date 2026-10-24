# coding: utf-8
"""后处理解码与 NMS 的单元测试（纯 numpy，不依赖 GPU / onnxruntime）。"""
import numpy as np
import pytest

from deadeye.detection_module.postprocess import (
    decode_detections,
    detect_output_layout,
    format_targets,
    multiclass_nms,
    nms,
)
from deadeye.detection_module.preprocess import LetterboxInfo


def _info(scale=1.0, pad_x=0.0, pad_y=0.0):
    return LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y)


# ---------------------------------------------------------------- layout 识别

def test_layout_v8_by_shape():
    assert detect_output_layout(np.zeros((8, 2100))) == 'v8'


def test_layout_v5_v7_by_shape():
    assert detect_output_layout(np.zeros((25200, 85))) == 'v5_v7'


def test_layout_end2end_by_shape():
    assert detect_output_layout(np.zeros((300, 6))) == 'end2end'


def test_layout_metadata_overrides_shape():
    # nc=2 的 v8 解码头 shape 恰为 (6, N)，靠元数据消除歧义
    assert detect_output_layout(np.zeros((6, 1344)), end2end_meta=False) == 'v8'
    assert detect_output_layout(np.zeros((6, 1344)), end2end_meta=True) == 'end2end'


# ---------------------------------------------------------------- v8/v11 解码

def test_decode_v8_single_box():
    # 4+2 类，anchor 在 (100, 120)，wh = (40, 60)，类别 1 分数 0.9
    pred = np.zeros((6, 1), dtype=np.float32)
    pred[:4, 0] = [100, 120, 40, 60]
    pred[5, 0] = 0.9
    dets = decode_detections(pred, _info(), conf_thres=0.5, layout='v8', iou_thres=0.5)
    assert dets.shape == (1, 6)
    assert dets[0, 4] == pytest.approx(0.9, abs=1e-5)
    assert dets[0, 5] == 1
    assert dets[0, 0] == pytest.approx(80)  # cx - w/2
    assert dets[0, 2] == pytest.approx(120)
    assert dets[0, 1] == pytest.approx(90)  # cy - h/2
    assert dets[0, 3] == pytest.approx(150)


def test_decode_v8_letterbox_inverse():
    """letterbox 坐标应正确映射回原图。"""
    pred = np.zeros((6, 1), dtype=np.float32)
    pred[:4, 0] = [164, 184, 40, 60]  # 模型输入坐标
    pred[5, 0] = 0.9
    info = LetterboxInfo(scale=0.5, pad_x=4.0, pad_y=6.0)
    dets = decode_detections(pred, info, conf_thres=0.5, layout='v8')
    assert dets[0, 0] == pytest.approx((164 - 4 - 20) / 0.5)
    assert dets[0, 1] == pytest.approx((184 - 6 - 30) / 0.5)


def test_decode_v8_conf_filter_and_batch_dim():
    pred = np.zeros((1, 6, 2), dtype=np.float32)  # 保留 batch 维
    pred[0, :4, 0] = [100, 120, 40, 60]
    pred[0, 5, 0] = 0.9
    pred[0, :4, 1] = [200, 220, 40, 60]
    pred[0, 5, 1] = 0.1  # 低于阈值
    dets = decode_detections(pred, _info(), conf_thres=0.5, layout='v8')
    assert dets.shape == (1, 6)
    assert dets[0, 0] == pytest.approx(80)


# ---------------------------------------------------------------- v5/v7 解码

def test_decode_v5_v7_objectness():
    pred = np.zeros((1, 7), dtype=np.float32)  # cx,cy,w,h,obj,c0,c1
    pred[0, :4] = [50, 60, 20, 40]
    pred[0, 4] = 0.8
    pred[0, 5] = 0.2
    pred[0, 6] = 0.5  # conf = 0.8 * 0.5 = 0.4
    dets = decode_detections(pred, _info(), conf_thres=0.3, layout='v5_v7')
    assert dets.shape == (1, 6)
    assert dets[0, 4] == pytest.approx(0.4, abs=1e-5)
    assert dets[0, 5] == 1


def test_decode_v5_v7_low_conf_filtered():
    pred = np.zeros((1, 7), dtype=np.float32)
    pred[0, :4] = [50, 60, 20, 40]
    pred[0, 4] = 0.5
    pred[0, 5] = 0.2
    pred[0, 6] = 0.3  # conf = 0.15
    dets = decode_detections(pred, _info(), conf_thres=0.3, layout='v5_v7')
    assert dets.shape == (0, 6)


# ---------------------------------------------------------------- 端到端解码

def test_decode_end2end():
    pred = np.array([[10, 20, 110, 220, 0.92, 2.0], [50, 60, 80, 90, 0.10, 0.0]], dtype=np.float32)
    dets = decode_detections(pred, _info(), conf_thres=0.5, layout='end2end')
    assert dets.shape == (1, 6)
    assert dets[0, 4] == pytest.approx(0.92)
    assert dets[0, 5] == 2


# ---------------------------------------------------------------- NMS

def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_thres=0.5)
    assert sorted(keep) == [0, 2]


def test_nms_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = nms(boxes, scores, iou_thres=0.5)
    assert sorted(keep) == [0, 1]


def test_multiclass_nms_cross_class_kept():
    """不同类别的重叠框都应保留。"""
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)
    dets = multiclass_nms(boxes, scores, class_ids, 0.5, 300)
    assert dets.shape[0] == 2


def test_multiclass_nms_max_dets():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    class_ids = np.array([0, 0, 0], dtype=np.int32)
    dets = multiclass_nms(boxes, scores, class_ids, 0.5, 2)
    assert dets.shape[0] == 2
    assert dets[0, 4] > dets[1, 4]


# ---------------------------------------------------------------- 目标格式

def test_format_targets():
    dets = np.array([[10.4, 20.6, 110.0, 220.9, 0.9, 1.0]], dtype=np.float32)
    targets = format_targets(dets, {0: 'ally', 1: 'enemy'})
    assert targets == [['enemy', (10, 21), (110, 221)]]


def test_format_targets_list_names():
    dets = np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)
    targets = format_targets(dets, ['tag'])
    assert targets == [['tag', (0, 0), (10, 10)]]
