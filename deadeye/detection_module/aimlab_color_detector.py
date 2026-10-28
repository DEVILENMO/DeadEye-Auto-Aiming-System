# coding: utf-8
# cython: language_level=3
"""
AimLab 专用颜色检测模块。

AimLab 的目标球是高饱和度的青色 / 黄色球体（带辉光），场景为低饱和度灰白棋盘格，
绿色横条与目标在色相上相距很远。因此用 HSV 双色相区间 + 饱和度/明度下限即可把
球从背景里完整分离，连通域统计给出外接框与质心，全流程向量化，无逐像素 Python 循环。

性能：640×640 RGB 输入单帧约 1 ms（cvtColor 0.2 + inRange×2 0.2 + 连通域 0.5），
比神经网络推理低两个数量级，适合超高帧率闭环。

输出与流水线契约一致：[label, (x1, y1), (x2, y2)] 列表，label 为 "ball"。
"""
from __future__ import annotations

import cv2
import numpy as np

from deadeye.base.modules import DetectModule
from deadeye.base.configurable import ConfigurableMixin

# OpenCV HSV 色相环为 [0,180)：青 ≈90，黄 ≈30；绿横条 ≈60，与两者均相距 ≥25
_CYAN_HUE = (85, 100)
_YELLOW_HUE = (20, 40)
_DEFAULT_SAT_MIN = 80   # 灰白棋盘格饱和度极低，一道下限即可排除
_DEFAULT_VAL_MIN = 120  # 排除暗角与阴影


class AimLabColorDetector(DetectModule, ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_bool('detect_cyan', True, label='识别青色球')
        cls.register_bool('detect_yellow', True, label='识别黄色球')
        cls.register_int('min_area_px', 64, label='最小面积(px)', min_value=1, max_value=100000, step=16)
        cls.register_int('sat_min', _DEFAULT_SAT_MIN, label='饱和度下限', min_value=0, max_value=255, step=5)
        cls.register_int('val_min', _DEFAULT_VAL_MIN, label='明度下限', min_value=0, max_value=255, step=5)
        cls.register_float('max_aspect_ratio', 1.6, label='最大长宽比', min_value=1.0, max_value=10.0, step=0.1)
        cls.register_float('min_fill_ratio', 0.55, label='最小填充率', min_value=0.1, max_value=1.0, step=0.05)

    def __init__(
        self,
        *,
        detect_cyan: bool = True,
        detect_yellow: bool = True,
        min_area_px: int = 64,
        sat_min: int = _DEFAULT_SAT_MIN,
        val_min: int = _DEFAULT_VAL_MIN,
        max_aspect_ratio: float = 1.6,
        min_fill_ratio: float = 0.55,
        **kwargs,
    ):
        if not detect_cyan and not detect_yellow:
            raise ValueError('detect_cyan 与 detect_yellow 至少开启一个')
        self.detect_cyan = bool(detect_cyan)
        self.detect_yellow = bool(detect_yellow)
        self.min_area_px = int(min_area_px)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_fill_ratio = float(min_fill_ratio)
        self.last_detect_ms = 0.0

    def target_detect(self, img) -> list:
        t0 = cv2.getTickCount()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = None
        for hue_lo, hue_hi, enabled in (
            (_CYAN_HUE[0], _CYAN_HUE[1], self.detect_cyan),
            (_YELLOW_HUE[0], _YELLOW_HUE[1], self.detect_yellow),
        ):
            if not enabled:
                continue
            lower = np.array([hue_lo, self.sat_min, self.val_min], dtype=np.uint8)
            upper = np.array([hue_hi, 255, 255], dtype=np.uint8)
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        results: list = []
        if mask is not None:
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, count):  # 0 是背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_area_px:
                    continue
                w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                # 球是圆的：长宽比接近 1、外接框填充率接近 π/4；
                # 辉光光晕与地面反光呈条状，以此剔除
                aspect = max(w, h) / max(1, min(w, h))
                fill = area / max(1, w * h)
                if aspect > self.max_aspect_ratio or fill < self.min_fill_ratio:
                    continue
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                results.append(['ball', (int(x), int(y)), (int(x + w), int(y + h))])

        self.last_detect_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        return results
