# coding: utf-8
# cython: language_level=3
import os
from collections import OrderedDict, namedtuple
from enum import Enum

import cv2
import numpy as np
import tensorrt as trt
# For error 'FileNotFoundError: Could not find: nvinfer.dll. Is it on your PATH?'
# Download TensorRT pack and add TensorRT-x.x.x.x\lib to environment variable PATH.
# For error 'FileNotFoundError: Could not find: cudnn64_8.dll. Is it on your PATH?'
# Download cudnn64_8.dll then also add it to path like C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxx.x\bin.
# For errors like 'Could not find module 'C:\...\nvinfer_plugin.dll' (or one of its dependencies).
# Try using the full path with constructor syntax.'
# change ctypes.CDLL(find_lib(lib)) to ctypes.CDLL(find_lib(lib),winmode=0)
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
from ultralytics import YOLO


class ModelType(Enum):
    TRT = 0
    ENGINE = 1


class Yolov8TensorRTEngine(object):
    def __init__(self, model_path):
        # 判断输入的模型文件后缀
        _, ext = os.path.splitext(model_path)
        if ext == '.trt':
            self.model_type = ModelType.TRT
        elif ext == '.engine':
            self.model_type = ModelType.ENGINE
        else:
            raise ValueError(f'Unsupported model file extension: {ext}')

        if self.model_type == ModelType.TRT:
            # Todo: If you choose to use your .trt weight file, you should modify the class info here.
            self.class_num = 3
            self.class_name_list = ['ally', 'enemy', 'tag']
            print(f'Class number: {self.class_num}, Class name: {self.class_name_list}')

            self.cuda_context_for_multiple_threading = cuda.Device(0).make_context()
            self.mean = None
            self.std = None

            logger = trt.Logger(trt.Logger.WARNING)
            logger.min_severity = trt.Logger.Severity.ERROR
            runtime = trt.Runtime(logger)
            trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
            with open(model_path, "rb") as f:
                serialized_engine = f.read()
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
            self.context = engine.create_execution_context()
            self.inputs, self.outputs, self.bindings = [], [], []
            self.stream = cuda.Stream()
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype, mem_flags=cuda.host_alloc_flags.PORTABLE)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                if engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
        elif self.model_type == ModelType.ENGINE:
            self.model = YOLO(model_path, task='detect')

    def on_exit(self):
        print('Destroying TensorRTEngine...')
        if self.model_type == ModelType.TRT:
            self.cuda_context_for_multiple_threading.pop()

    def inference(self, origin_img, conf=0.5, end2end=True):
        if self.model_type == ModelType.TRT:
            img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
            data = self._trt_infer(img)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                                       np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5 + self.class_num)))[0]
                dets = self._trt_postprocess(predictions, ratio)

            targets = []
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                            :4], dets[:, 4], dets[:, 5]
                class_names = self.class_name_list
                boxes = final_boxes[final_scores > conf]
                classes = final_cls_inds[final_scores > conf]

                for box, cls_idx in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names[int(cls_idx)]
                    target_info = [class_name, (x1, y1), (x2, y2)]
                    targets.append(target_info)
            return targets
        elif self.model_type == ModelType.ENGINE:
            results = self.model(origin_img)
            output_results = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = result.names[cls]

                    if conf >= conf:
                        output_result = (class_name, (x1, y1), (x2, y2))
                        output_results.append(output_result)
            return output_results
        else:
            return []

    def _trt_infer(self, img):
        self.cuda_context_for_multiple_threading.push()
        temp_host_mem = np.ravel(img)
        np.copyto(self.inputs[0]['host'], temp_host_mem)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        self.cuda_context_for_multiple_threading.pop()
        return data

    @staticmethod
    def _trt_postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def _engine_infer(self, img):
        self.cuda_context_for_multiple_threading.push()
        temp_host_mem = np.ravel(img)
        np.copyto(self.inputs[0]['host'], temp_host_mem)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        self.cuda_context_for_multiple_threading.pop()
        return data

    @staticmethod
    def _engine_postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r
