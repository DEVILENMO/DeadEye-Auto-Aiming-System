# coding: utf-8
# cython: language_level=3
"""
屏幕截取模块（dxcam 替代方案）。

后端与优先级（AUTO 模式下依次尝试）：
    1. windows-capture : Windows.Graphics.Capture（Win10 1903+），Rust 实现，
                         维护活跃、速度最快；原生裁剪，回调线程推送最新帧。
    2. bettercam       : DXGI Desktop Duplication，dxcam 的社区维护分支，
                         API 与 dxcam 兼容，驱动侧直接输出 RGB。
    3. mss             : GDI BitBlt 兜底，任何 Windows 可用，CPU 占用高、速度慢。

统一输出约定：
    - 抓取屏幕中心 view_width x view_height 区域（屏幕不足时向内收缩）；
    - RGB、uint8、HWC 的 numpy 数组；
    - 无新帧时返回 None（DXGI/WGC 只在画面变化时产生新帧，调用方可循环重试）。
"""
import logging
import threading
from enum import Enum

import cv2
import numpy as np
import pyautogui
import win32api
import win32con
from mss import MSS as _Mss, mss as _mss_factory


class _WgcBackend:
    """windows-capture（Windows.Graphics.Capture）后端。

    推模式：库在独立线程回调 on_frame_arrived，这里只保留最新一帧；
    grab() 通过帧序号判断是否有新帧，转为拉模式语义。
    回调内 frame_buffer 是原生映射的零拷贝视图，回调返回后失效，必须就地拷贝。
    注意：monitor_index 从 1 开始（与 Windows 显示设置编号一致）。
    """

    name = 'windows-capture'

    def __init__(self, view_width, view_height, monitor_index=1):
        from windows_capture import WindowsCapture

        self.view_width = view_width
        self.view_height = view_height
        self._monitor_index = monitor_index
        self._lock = threading.Lock()
        self._latest_frame = None
        self._seq = 0
        self._last_seq = -1
        self._closed = False
        self._capture_control = None

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            monitor_index=monitor_index,
            # 0 = 不设最小帧间隔，帧率跟随显示器刷新率（默认 None 会被压到 60fps）
            minimum_update_interval=0,
        )

        @capture.event
        def on_frame_arrived(frame, capture_control):
            # 原生裁剪出中心视口再转换颜色，只对 view 大小的数据做拷贝
            left, top, vw, vh = self._calc_crop(frame.width, frame.height)
            buf = frame.crop(left, top, left + vw, top + vh).frame_buffer
            rgb = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB)
            with self._lock:
                self._latest_frame = rgb
                self._seq += 1

        @capture.event
        def on_closed():
            self._closed = True
            logging.getLogger('ScreenShotHelper Logger').warning(
                'windows-capture 会话已关闭（显示器断开或系统结束捕获）。')

        self._capture_control = capture.start_free_threaded()

    @property
    def screen_resolution(self):
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return pyautogui.size()
        h, w = frame.shape[:2]
        return w, h

    def _calc_crop(self, screen_w, screen_h):
        vw = min(self.view_width, screen_w)
        vh = min(self.view_height, screen_h)
        left = (screen_w - vw) // 2
        top = (screen_h - vh) // 2
        return left, top, vw, vh

    def grab(self):
        if self._closed:
            return None
        with self._lock:
            seq = self._seq
            frame = self._latest_frame
        if frame is None or seq == self._last_seq:
            return None
        self._last_seq = seq
        return frame

    def close(self):
        if self._capture_control is None:
            return
        try:
            self._capture_control.stop()
        except Exception:
            pass
        self._capture_control = None


def _bettercam_primary_res():
    """从 bettercam.output_info() 解析主显示器分辨率；失败返回 None。

    bettercam 1.x 已移除 output_res()，output_info 是仅有的公开接口。
    """
    import re

    import bettercam

    for line in bettercam.output_info().splitlines():
        if 'Primary:True' not in line:
            continue
        m = re.search(r'Res:\((\d+),\s*(\d+)\)', line)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


class _BetterCamBackend:
    """bettercam（DXGI Desktop Duplication）后端，dxcam 的维护分支。

    output_color='RGB' 让转换发生在驱动/C++ 侧，Python 每帧零转换成本。
    """

    name = 'bettercam'

    def __init__(self, view_width, view_height):
        import bettercam

        self._bettercam = bettercam
        self.view_width = view_width
        self.view_height = view_height
        res = _bettercam_primary_res()
        if res is None:
            raise RuntimeError('无法从 bettercam.output_info() 解析主显示器分辨率')
        self.resolution_x, self.resolution_y = res
        self._camera = bettercam.create(output_idx=0, output_color='RGB')
        self._region = self._calc_region()

    @property
    def screen_resolution(self):
        return self.resolution_x, self.resolution_y

    def _calc_region(self):
        width = min(self.view_width, self.resolution_x)
        height = min(self.view_height, self.resolution_y)
        left = (self.resolution_x - width) // 2
        top = (self.resolution_y - height) // 2
        # bettercam 的 region 为 (left, top, right, bottom)，右/下边界排除（numpy 切片风格）
        return left, top, left + width, top + height

    def grab(self):
        current = _bettercam_primary_res()
        if current is not None and current != (self.resolution_x, self.resolution_y):
            # DXGI 不支持热更新分辨率，屏幕变更后重建相机
            self.resolution_x, self.resolution_y = current
            self._camera.release()
            self._camera = self._bettercam.create(output_idx=0, output_color='RGB')
            self._region = self._calc_region()
        return self._camera.grab(region=self._region)

    def close(self):
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None


class _MssBackend:
    """mss（GDI BitBlt）后端，兼容性兜底。每次调用都强制重新截取，永远返回新帧。"""

    name = 'mss'

    def __init__(self, view_width, view_height):
        self.view_width = view_width
        self.view_height = view_height
        self.resolution_x, self.resolution_y = pyautogui.size()
        try:
            self._sct = _Mss()  # mss >= 10.2
        except TypeError:
            self._sct = _mss_factory()  # 旧版 mss 无 MSS 类
        self._region = self._calc_region()

    @property
    def screen_resolution(self):
        return self.resolution_x, self.resolution_y

    def _calc_region(self):
        width = min(self.view_width, self.resolution_x)
        height = min(self.view_height, self.resolution_y)
        left = (self.resolution_x - width) // 2
        top = (self.resolution_y - height) // 2
        return {'top': top, 'left': left, 'width': width, 'height': height}

    def grab(self):
        current = pyautogui.size()
        if current != (self.resolution_x, self.resolution_y):
            self.resolution_x, self.resolution_y = current
            self._region = self._calc_region()
        shot = self._sct.grab(self._region)
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2RGB)

    def close(self):
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None


class ScreenShotHelper:
    class CameraType(Enum):
        AUTO = 0
        WINDOWS_CAPTURE = 1
        BETTERCAM = 2
        MSS = 3

    class ImageColorMode(Enum):
        RGB = 0
        BGR = 1

    _BACKEND_FACTORIES = {
        CameraType.WINDOWS_CAPTURE: _WgcBackend,
        CameraType.BETTERCAM: _BetterCamBackend,
        CameraType.MSS: _MssBackend,
    }

    def __init__(self, img_width, img_height, camera_type=CameraType.AUTO):
        self.logger = logging.getLogger('ScreenShotHelper Logger')

        self.view_width = img_width
        self.view_height = img_height
        self.image_color_mode = self.ImageColorMode.RGB

        self.screen_resolution_x, self.screen_resolution_y = (
            win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
        print('Primary screen resolution:', self.screen_resolution_x, 'x', self.screen_resolution_y)

        candidates = self._resolve_candidates(camera_type)
        self.backend = None
        errors = []
        for candidate in candidates:
            factory = self._BACKEND_FACTORIES[candidate]
            try:
                if (candidate == self.CameraType.BETTERCAM
                        and not self._dda_resolution_matches()):
                    raise RuntimeError(
                        'DXGI 分辨率与系统分辨率不一致（显示缩放非 100% 时 DXGI 不可用）')
                self.backend = factory(self.view_width, self.view_height)
                print(f'Screen capture backend: {self.backend.name}')
                if (self.view_width > self.screen_resolution_x
                        or self.view_height > self.screen_resolution_y):
                    self.logger.warning(
                        f'视口 {self.view_width}x{self.view_height} 超过屏幕 '
                        f'{self.screen_resolution_x}x{self.screen_resolution_y}，已向内收缩。')
                break
            except Exception as e:
                errors.append(f'{candidate.name}: {type(e).__name__}: {e}')
                self.logger.warning(f'截屏后端 {candidate.name} 初始化失败，尝试下一个。')
        if self.backend is None:
            raise RuntimeError('所有截屏后端均初始化失败:\n  ' + '\n  '.join(errors))

    def _resolve_candidates(self, camera_type):
        if camera_type != self.CameraType.AUTO:
            return [camera_type]
        return [self.CameraType.WINDOWS_CAPTURE, self.CameraType.BETTERCAM, self.CameraType.MSS]

    def _dda_resolution_matches(self):
        try:
            res = _bettercam_primary_res()
            return res == (self.screen_resolution_x, self.screen_resolution_y)
        except Exception:
            return False

    def capture_screen_shot(self):
        """返回中心视口的 RGB 图像；无新帧时返回 None。"""
        return self.backend.grab()

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
