# coding: utf-8
# cython: language_level=3
"""
AimLab 专用颜色检测模块。

AimLab 的目标球是高饱和度球体（带辉光），场景为低饱和度灰白棋盘格，绿色横条
与目标在色相上相距很远。检测器的三个颜色槽位与游戏内设置一一对应（默认/命中/
未命中目标颜色，蓝/金/蓝），按 hex 标定色相中心 ± 容差生成掩码，饱和度/明度
下限排除背景；连通域统计给出外接框与质心，全流程向量化，无逐像素 Python 循环。

性能：640×640 RGB 输入单帧约 1 ms（cvtColor 0.2 + inRange×2 0.2 + 连通域 0.5），
比神经网络推理低两个数量级，适合超高帧率闭环。

输出与流水线契约一致：[label, (x1, y1), (x2, y2)] 列表，label 为 "ball"。
"""
from __future__ import annotations

import cv2
import numpy as np

from deadeye.base.modules import DetectModule
from deadeye.base.configurable import ConfigurableMixin

# AimLab 颜色槽位默认值（与游戏内「默认/命中/未命中目标颜色」三项一一对应，
# 游戏内可自定义，UI 中按 hex 修改）
DEFAULT_TARGET_HEX = '#00FFFF'  # 蓝
HIT_TARGET_HEX = '#EDC709'      # 金
MISS_TARGET_HEX = '#00FFFF'     # 蓝

_DEFAULT_SAT_MIN = 80   # 灰白棋盘格饱和度极低，一道下限即可排除
_DEFAULT_VAL_MIN = 120  # 排除暗角与阴影


def _parse_hex_color(raw: str, fallback: str) -> tuple[int, int, int]:
    """'#RRGGBB' / 'RRGGBB' -> (r, g, b)；非法输入回退默认值。"""
    text = str(raw).strip().lstrip('#')
    if len(text) == 6:
        try:
            return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
        except ValueError:
            pass
    print(f'警告：颜色 {raw!r} 无法解析，回退 {fallback}')
    return _parse_hex_color(fallback, fallback)


def _rgb_hue(rgb: tuple[int, int, int]) -> int:
    """RGB -> OpenCV HSV 色相 [0,180)。"""
    pixel = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
    return int(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0, 0])


class AimLabColorDetector(DetectModule, ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_str('default_target_color', DEFAULT_TARGET_HEX, label='默认目标颜色')
        cls.register_str('hit_target_color', HIT_TARGET_HEX, label='命中目标颜色')
        cls.register_str('miss_target_color', MISS_TARGET_HEX, label='未命中目标颜色')
        cls.register_int('hue_tolerance', 10, label='色相容差', min_value=1, max_value=45, step=1)
        cls.register_bool('detect_default', True, label='识别默认目标')
        cls.register_bool('detect_hit', True, label='识别命中目标')
        cls.register_bool('detect_miss', True, label='识别未命中目标')
        cls.register_int('min_area_px', 64, label='最小面积(px)', min_value=1, max_value=100000, step=16)
        cls.register_int('sat_min', _DEFAULT_SAT_MIN, label='饱和度下限', min_value=0, max_value=255, step=5)
        cls.register_int('val_min', _DEFAULT_VAL_MIN, label='明度下限', min_value=0, max_value=255, step=5)
        cls.register_float('max_aspect_ratio', 1.6, label='最大长宽比', min_value=1.0, max_value=10.0, step=0.1)
        cls.register_float('min_fill_ratio', 0.55, label='最小填充率', min_value=0.1, max_value=1.0, step=0.05)

    def __init__(
        self,
        *,
        default_target_color: str = DEFAULT_TARGET_HEX,
        hit_target_color: str = HIT_TARGET_HEX,
        miss_target_color: str = MISS_TARGET_HEX,
        hue_tolerance: int = 10,
        detect_default: bool = True,
        detect_hit: bool = True,
        detect_miss: bool = True,
        min_area_px: int = 64,
        sat_min: int = _DEFAULT_SAT_MIN,
        val_min: int = _DEFAULT_VAL_MIN,
        max_aspect_ratio: float = 1.6,
        min_fill_ratio: float = 0.55,
        **kwargs,
    ):
        if not (detect_default or detect_hit or detect_miss):
            raise ValueError('三个颜色槽位的识别开关至少开启一个')
        self.detect_default = bool(detect_default)
        self.detect_hit = bool(detect_hit)
        self.detect_miss = bool(detect_miss)
        self.hue_tolerance = int(hue_tolerance)
        # 目标色 hex -> 色相中心；区间 = 中心 ± 容差（环绕处理见 _build_mask）
        self.slots = {
            'default': {'rgb': _parse_hex_color(default_target_color, DEFAULT_TARGET_HEX),
                        'enabled': self.detect_default},
            'hit': {'rgb': _parse_hex_color(hit_target_color, HIT_TARGET_HEX),
                    'enabled': self.detect_hit},
            'miss': {'rgb': _parse_hex_color(miss_target_color, MISS_TARGET_HEX),
                     'enabled': self.detect_miss},
        }
        for slot in self.slots.values():
            slot['hue'] = _rgb_hue(slot['rgb'])
        self.min_area_px = int(min_area_px)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_fill_ratio = float(min_fill_ratio)
        self.last_detect_ms = 0.0
        hues = ', '.join(f'{name} H{s["hue"]}' for name, s in self.slots.items() if s['enabled'])
        print(f'AimLabColorDetector: {hues}, tol ±{self.hue_tolerance}')

    def _build_mask(self, hsv: np.ndarray, hue_center: int) -> np.ndarray:
        """色相中心 ± 容差的掩码，处理色相环 [0,180) 的环绕。"""
        tol = self.hue_tolerance
        lo = (hue_center - tol) % 180
        hi = (hue_center + tol) % 180
        lower_s_v = np.array([0, self.sat_min, self.val_min], dtype=np.uint8)
        upper_s_v = np.array([179, 255, 255], dtype=np.uint8)
        if lo <= hi:
            lower = np.array([lo, self.sat_min, self.val_min], dtype=np.uint8)
            upper = np.array([hi, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        # 环绕：区间跨 0（如 175..5），拆两段
        part_lo = cv2.inRange(hsv, np.array([lo, *lower_s_v[1:]], dtype=np.uint8), upper_s_v)
        part_hi = cv2.inRange(hsv, lower_s_v, np.array([hi, *upper_s_v[1:]], dtype=np.uint8))
        return cv2.bitwise_or(part_lo, part_hi)

    def target_detect(self, img) -> list:
        t0 = cv2.getTickCount()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = None
        for slot in self.slots.values():
            if not slot['enabled']:
                continue
            part = self._build_mask(hsv, slot['hue'])
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        results: list = []
        if mask is not None:
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
                # 以质心为框中心（亚像素、对辉光变化稳定），尺寸仍取连通域外接框
                cx, cy = centroids[i]
                x1 = int(round(cx - w * 0.5))
                y1 = int(round(cy - h * 0.5))
                results.append(['ball', (x1, y1), (x1 + int(w), y1 + int(h))])

        self.last_detect_ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        return results
