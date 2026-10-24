# coding: utf-8
# cython: language_level=3
from ctypes import windll

from deadeye.base.modules import BaseCamera
from deadeye.base.configurable import ConfigurableMixin
from deadeye.input_module.screenshot_helper import ScreenShotHelper


class SimpleScreenShotCamera(BaseCamera, ConfigurableMixin):
    """屏幕截取相机：捕获屏幕中心 view_range 区域，输出 RGB 图像。

    默认 AUTO 后端（windows-capture -> bettercam -> mss 依次降级），
    可通过 GUI 选择指定后端；抓取宽高默认 320x320，可在 GUI 中修改。
    """

    DEFAULT_VIEW_SIZE = (320, 320)

    BACKEND_MAP = {
        'auto': ScreenShotHelper.CameraType.AUTO,
        'windows_capture': ScreenShotHelper.CameraType.WINDOWS_CAPTURE,
        'bettercam': ScreenShotHelper.CameraType.BETTERCAM,
        'mss': ScreenShotHelper.CameraType.MSS,
    }

    @classmethod
    def declare_config(cls) -> None:
        cls.register_int(
            'view_width',
            cls.DEFAULT_VIEW_SIZE[0],
            label='抓取宽度(px)',
            min_value=64,
            max_value=7680,
        )
        cls.register_int(
            'view_height',
            cls.DEFAULT_VIEW_SIZE[1],
            label='抓取高度(px)',
            min_value=64,
            max_value=4320,
        )
        cls.register_choice(
            'capture_backend',
            'auto',
            label='截屏后端',
            choices=list(cls.BACKEND_MAP.keys()),
        )

    def __init__(self, view_range=None, *, capture_backend: str = 'auto',
                 view_width: int = 0, view_height: int = 0):
        user32 = windll.user32
        user32.SetProcessDPIAware()
        self.ori_resolution_x, self.ori_resolution_y = (
            user32.GetSystemMetrics(0),
            user32.GetSystemMetrics(1))
        print('Screen resolution:', self.ori_resolution_x, 'x', self.ori_resolution_y)

        # 优先级：显式 view_width/view_height > view_range > 类默认值；并钳制到屏幕内，
        # 保证 view_range 与实际输出帧一致（执行模块按 view_range 做屏幕坐标映射）
        width = int(view_width) or (view_range[0] if view_range else self.DEFAULT_VIEW_SIZE[0])
        height = int(view_height) or (view_range[1] if view_range else self.DEFAULT_VIEW_SIZE[1])
        width = max(1, min(width, self.ori_resolution_x))
        height = max(1, min(height, self.ori_resolution_y))
        self.view_range = (width, height)

        camera_type = self.BACKEND_MAP.get(capture_backend, ScreenShotHelper.CameraType.AUTO)
        self.screen_shot_camera = ScreenShotHelper(width, height, camera_type)

    def get_image(self):
        # ScreenShotHelper 统一输出 RGB，无需再做通道转换
        return self.screen_shot_camera.capture_screen_shot()

    def on_exit(self):
        self.screen_shot_camera.close()
