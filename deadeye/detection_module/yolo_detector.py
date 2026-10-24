# coding: utf-8
# cython: language_level=3
"""
统一 YOLO 检测器：一个类覆盖 YOLOv5/v7/v8/v11/v12/26 的部署全流程。

按模型文件后缀自动选择推理后端：
- ``.pt``              -> UltralyticsBackend（torch，训练权重直接推理）
- ``.onnx``            -> OnnxRuntimeBackend（CUDA EP / TRT EP / CPU 自适应）
- ``.engine/.plan/.trt`` -> TensorRTBackend（原生 TRT 10 + cuda-python）

后处理按输出布局自动解码（详见 postprocess.py），输入图像约定为 RGB uint8。
"""
from __future__ import annotations

import os
import time

from deadeye.base.modules import DetectModule
from deadeye.base.configurable import ConfigurableMixin
from deadeye.detection_module.backends.base import BaseInferBackend
from deadeye.detection_module.backends.onnx_backend import OnnxRuntimeBackend
from deadeye.detection_module.backends.tensorrt_backend import ENGINE_EXTS, TensorRTBackend
from deadeye.detection_module.backends.ultralytics_backend import UltralyticsBackend
from deadeye.detection_module.postprocess import decode_detections, format_targets
from deadeye.detection_module.preprocess import Letterboxer

ULTRALYTICS_EXTS = ('.pt',)


def _default_model_path() -> str:
    candidates = [
        os.path.abspath(os.path.join('.', 'weights', 'test_v11s_320.onnx')),
        os.path.abspath(os.path.join('.', 'weights', 'apex_yolov8.pt')),
        os.path.abspath(os.path.join('.', 'weights', 'best.pt')),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ''


def _parse_class_names(raw: str) -> dict[int, str]:
    names = [name.strip() for name in str(raw).split(',') if name.strip()]
    return {i: name for i, name in enumerate(names)}


class YoloDetector(DetectModule, ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_path(
            'model_path',
            _default_model_path(),
            label='模型路径',
            filetypes=[
                ('YOLO Models', '*.pt *.onnx *.engine *.plan *.trt'),
                ('All files', '*.*'),
            ],
        )
        cls.register_float('conf_threshold', 0.45, label='置信度阈值', min_value=0.01, max_value=0.99, step=0.05)
        cls.register_float('iou_threshold', 0.5, label='NMS IoU 阈值', min_value=0.1, max_value=0.95, step=0.05)
        cls.register_int('max_dets', 300, label='最大检测数', min_value=1, max_value=1000, step=1)
        cls.register_choice('device', 'auto', label='推理设备(ONNX)', choices=['auto', 'cuda', 'cpu'])
        cls.register_bool('use_tensorrt_ep', False, label='ONNX 走 TRT EP(实验)')
        cls.register_str('class_names', '', label='类别名覆盖(逗号分隔,可空)')

    def __init__(
        self,
        *,
        model_path: str = '',
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        max_dets: int = 300,
        device: str = 'auto',
        use_tensorrt_ep: bool = False,
        class_names: str = '',
        **kwargs,
    ):
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_dets = int(max_dets)
        self._device = str(device)
        self._use_trt_ep = bool(use_tensorrt_ep)

        self.model: BaseInferBackend | None = None
        self._letterboxer: Letterboxer | None = None
        self._ultralytics_raw = None  # UltralyticsBackend 专用的快速通道
        self.last_infer_ms = 0.0
        self.last_pre_ms = 0.0
        self.last_post_ms = 0.0

        override_names = _parse_class_names(class_names)
        self.load_model(model_path, override_names)

    # -------------------------------------------------- 模型加载

    def load_model(self, model_path: str, override_names: dict[int, str] | None = None):
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path!r}')

        lower = model_path.lower()
        if lower.endswith(ULTRALYTICS_EXTS):
            backend = UltralyticsBackend(model_path)
        elif lower.endswith('.onnx'):
            backend = OnnxRuntimeBackend(model_path, device=self._device, use_tensorrt_ep=self._use_trt_ep)
        elif lower.endswith(ENGINE_EXTS):
            backend = TensorRTBackend(model_path)
        else:
            raise ValueError(f'不支持的模型文件: {model_path}（支持 .pt / .onnx / .engine / .plan / .trt）')

        if override_names:
            backend.meta.names = override_names

        self.model = backend
        self._ultralytics_raw = backend if isinstance(backend, UltralyticsBackend) else None
        if not isinstance(backend, UltralyticsBackend):
            h, w = backend.meta.input_size
            self._letterboxer = Letterboxer(h, w)

        print(f'YoloDetector: {backend!r} | backend={backend.backend_name} | '
              f'conf={self.conf_threshold} iou={self.iou_threshold}')
        if not backend.meta.names:
            print('警告：模型未携带类别名元数据，将使用类别下标作为名称；'
                  '可在「类别名覆盖」中填入逗号分隔的名称。')

    def on_exit(self):
        if self.model is not None:
            self.model.on_exit()
            self.model = None

    # -------------------------------------------------- 推理

    def target_detect(self, img) -> list:
        if self.model is None:
            raise RuntimeError('模型未加载，请先调用 load_model()。')

        # Ultralytics 快速通道：内部完成预处理与 NMS
        if self._ultralytics_raw is not None:
            t0 = time.perf_counter()
            dets = self._ultralytics_raw.detect(img, conf=self.conf_threshold, iou=self.iou_threshold)
            self.last_infer_ms = (time.perf_counter() - t0) * 1000
            self.last_pre_ms = self.last_post_ms = 0.0
            return format_targets(dets, self.model.meta.names)

        t0 = time.perf_counter()
        blob, info = self._letterboxer(img)
        t1 = time.perf_counter()

        outputs = self.model.infer(blob)
        t2 = time.perf_counter()

        dets = decode_detections(
            outputs[0],
            info,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            max_dets=self.max_dets,
            layout='auto',
            end2end_meta=self.model.meta.end2end,
        )
        targets = format_targets(dets, self.model.meta.names)
        t3 = time.perf_counter()

        self.last_pre_ms = (t1 - t0) * 1000
        self.last_infer_ms = (t2 - t1) * 1000
        self.last_post_ms = (t3 - t2) * 1000
        return targets
