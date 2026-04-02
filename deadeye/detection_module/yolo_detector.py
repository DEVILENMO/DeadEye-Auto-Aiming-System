# coding: utf-8
# cython: language_level=3
import os

from ultralytics import YOLO

from deadeye.base.modules import DetectModule
from deadeye.base.configurable import ConfigurableMixin


def _default_model_path() -> str:
    candidates = [
        os.path.abspath(os.path.join('.', 'weights', 'apex.pt')),
        os.path.abspath(os.path.join('.', 'weights', 'best.pt')),
        os.path.abspath(os.path.join('.', 'weights', 'best.engine')),
        os.path.abspath(os.path.join('.', 'weights', 'apex_v8s.engine')),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ''


class YoloDetector(DetectModule, ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_path(
            'model_path',
            _default_model_path(),
            label='模型路径',
            filetypes=[
                ('YOLO Weights', '*.pt *.engine *.trt'),
                ('All files', '*.*'),
            ],
        )

    def __init__(self, *, model_path: str = '', **kwargs):
        self.model = None
        self.model_type = None
        self.load_model(model_path)

    def on_exit(self):
        if self.model_type == 'trt':
            self.model.on_exit()

    def load_model(self, model: str):
        lower = model.lower()
        if lower.endswith('.pt'):
            self.model = YOLO(model)
            self.model_type = 'pt'
        elif lower.endswith('.engine'):
            print('Loading Ultralytics YOLO engine...')
            self.model = YOLO(model, task='detect')
            self.model_type = 'engine'
        elif lower.endswith('.trt'):
            print('Loading TensorRT .trt model (requires pycuda + tensorrt)...')
            from deadeye.detection_module.tensorrt_engine import Yolov8TensorRTEngine
            self.model = Yolov8TensorRTEngine(model)
            self.model_type = 'trt'
        else:
            raise ValueError(f'不支持的模型文件: {model}（支持 .pt / .engine / .trt）')

    def target_detect(self, img) -> list:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        results = []
        if self.model_type in ('pt', 'engine'):
            result = self.model(img, verbose=False, half=True, iou=0.8, conf=0.75)
            detections = result[0]

            boxes = detections.boxes.xyxy.cpu().numpy()
            labels = detections.boxes.cls.cpu().numpy().astype(int)

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                class_name = detections.names[label]
                one = [class_name, (int(x1), int(y1)), (int(x2), int(y2))]
                results.append(one)
        else:
            results = self.model.inference(img, conf=0.75, end2end=True)

        return results
