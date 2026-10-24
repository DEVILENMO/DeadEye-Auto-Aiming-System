# coding: utf-8
# cython: language_level=3
"""
Ultralytics 原生后端：.pt / .engine 权重直接走 ``ultralytics.YOLO``。

适用于 YOLOv8 / v11 / v12 / 26 的 ``.pt`` 训练权重（训练-验证闭环最方便），
预处理、NMS 均由 ultralytics 完成；本后端只做结果格式统一。
依赖较重（torch），部署场景建议优先使用 ONNX / TensorRT 后端。
"""
from __future__ import annotations

import os

import numpy as np

from deadeye.detection_module.backends.base import BaseInferBackend, ModelMeta
from deadeye.detection_module.cuda_env import bootstrap_nvidia_dlls


class UltralyticsBackend(BaseInferBackend):
    def __init__(self, model_path: str, *, device: int | str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('未安装 ultralytics，请执行: pip install ultralytics') from exc

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'模型不存在: {model_path}')

        bootstrap_nvidia_dlls()
        self.model = YOLO(model_path, task='detect')
        self._device = device

        names = {int(k): str(v) for k, v in (self.model.names or {}).items()}
        imgsz = getattr(self.model.overrides, 'get', lambda *_: None)('imgsz') or 640
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[-1]
        self.meta = ModelMeta(
            input_size=(int(imgsz), int(imgsz)),
            names=names,
            end2end=None,  # ultralytics 内部完成后处理，布局标记无意义
            description=f'ultralytics {getattr(self.model, "version", "")}'.strip(),
        )
        self._warmup(int(imgsz))

    def _warmup(self, imgsz: int) -> None:
        """触发 torch/cudnn 初始化，把一次性开销挡在启动阶段而非首帧。"""
        try:
            dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, device=self._device)
        except Exception as exc:  # 预热失败不阻塞加载（如无 GPU 时）
            print(f'警告：ultralytics 预热失败（不影响功能）: {exc}')

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        """Ultralytics 后端不走 blob 约定：YoloDetector 直接调用 detect()。"""
        raise NotImplementedError('UltralyticsBackend 请通过 detect() 使用')

    def detect(self, img_rgb: np.ndarray, conf: float, iou: float):
        """跑一次完整检测，返回 [n, 6]（xyxy score cls，原图坐标）。"""
        results = self.model(
            img_rgb,
            verbose=False,
            conf=conf,
            iou=iou,
            device=self._device,
        )
        result = results[0]
        dets = np.zeros((0, 6), dtype=np.float32)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(np.float32)
            dets = np.concatenate([boxes, scores[:, None], classes[:, None]], axis=1)
        return dets

    @property
    def backend_name(self) -> str:
        return 'UltralyticsBackend(pt)'

    def on_exit(self) -> None:
        self.model = None
