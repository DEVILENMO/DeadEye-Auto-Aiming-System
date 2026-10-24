# coding: utf-8
# cython: language_level=3
"""
统一的检测输入预处理：letterbox 缩放 + 归一化 + HWC->CHW。

流水线约定：相机模块输出 RGB uint8 图像（BGR 由相机内部转换），
本模块输入 RGB，输出 float32 CHW blob，数值域 [0, 1]。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    """记录 letterbox 的几何参数，供后处理把框映射回原图坐标。"""

    scale: float  # 原图 -> 模型输入 的缩放比（原图 * scale = 模型尺寸）
    pad_x: float  # 水平方向两侧总 padding 的一半（居中填充）
    pad_y: float  # 垂直方向两侧总 padding 的一半

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        """模型输入坐标 -> 原图坐标。"""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


class Letterboxer:
    """
    预分配输出缓冲的 letterbox 预处理器。

    稳态推理路径上不产生新的 numpy 分配：
    - resize 结果直接写入预分配的 padded 缓冲；
    - CHW blob 复用同一块内存。
    """

    def __init__(self, input_h: int, input_w: int, pad_value: int = 114):
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self._pad_value = pad_value
        self._padded = np.full((self.input_h, self.input_w, 3), pad_value, dtype=np.uint8)
        self._blob = np.empty((1, 3, self.input_h, self.input_w), dtype=np.float32)

    def __call__(self, img_rgb: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        h, w = img_rgb.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_w = min(self.input_w, int(round(w * scale)))
        new_h = min(self.input_h, int(round(h * scale)))

        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if resized.dtype != np.uint8:
            resized = resized.astype(np.uint8)

        pad_x = (self.input_w - new_w) * 0.5
        pad_y = (self.input_h - new_h) * 0.5
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        # 重置 padding 区域（上一帧的图像可能残留）
        self._padded[:] = self._pad_value
        self._padded[y0:y0 + new_h, x0:x0 + new_w] = resized

        # HWC uint8 RGB -> CHW float32 [0,1]，写入预分配 blob
        np.divide(
            self._padded.transpose(2, 0, 1),
            255.0,
            out=self._blob[0],
            dtype=np.float32,
        )
        return self._blob, LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y)


def letterbox(img_rgb: np.ndarray, input_h: int, input_w: int, pad_value: int = 114):
    """无状态版本（一次性使用，如脚本/测试）。"""
    return Letterboxer(input_h, input_w, pad_value)(img_rgb)
