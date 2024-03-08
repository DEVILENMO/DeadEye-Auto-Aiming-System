# coding: utf-8
# cython: language_level=3
import argparse
import random

import numpy as np
import onnxruntime
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging
from utils.torch_utils import select_device, TracedModel


class Yolov7Helper:
    def __init__(self, weight: str, conf_thres=0.5, iou_thres=0.6):
        print('Trying to init Yolo v7...')
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=weight, help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=conf_thres, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=iou_thres, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_false', help='display results')
        parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        self.opt = parser.parse_args()
        # check_requirements(exclude=('pycocotools', 'thop'))

        with torch.no_grad():
            if self.opt.update:  # update all models (to fix SourceChangeWarning)
                for self.opt.weights in ['yolov7.pt']:
                    strip_optimizer(self.opt.weights)

        # 获取基本选项参数
        print('Loading parameters...')
        weight, view_img, save_txt, imgsz, trace = self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, self.opt.no_trace
        print('weights:', weight)

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        print('Device type:', self.device.type)

        # Load model
        print('Model will predict based on .pt file.')
        self.model = attempt_load(weight, map_location=self.device)  # load FP32 model
        self.model_type = 'pt'
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, self.opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print('Weight file classes:', self.names)

        # Run inference
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1
        print('Yolo v7 is ready!')

    def target_detection(self, img0):
        """
        Do target detection here.
        :param img0: input image.
        :return: List<(label, left_top_corner_pos, right_bottom_corner_pos)>
        """
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # img预处理
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (
                self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.opt.augment)[0]

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)

        # Process detections
        result = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    result.append((int(cls), c1, c2))

        return result