# coding: utf-8
# cython: language_level=3
import cv2
import win32api
import win32con
from ctypes import windll

import dxcam

from deadeye.base.modules import BaseCamera
from deadeye.input_module.screenshot_helper import ScreenShotHelper


class SimpleScreenShotCamera(BaseCamera):
    def __init__(self, view_range):
        super(SimpleScreenShotCamera).__init__()
        user32 = windll.user32
        user32.SetProcessDPIAware()
        self.ori_resolution_x, self.ori_resolution_y = (
            win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN))
        print('Screen resolution:', self.ori_resolution_x, 'x', self.ori_resolution_y)
        self.rel_resolution_x, self.rel_resolution_y = dxcam.output_res()[0]
        print('Scaled screen resolution:', self.rel_resolution_x, 'x', self.rel_resolution_y)
        self.screen_shot_camera = ScreenShotHelper(view_range[0], view_range[1], ScreenShotHelper.CameraType.DXCAM)

    def get_image(self):
        image = self.screen_shot_camera.capture_screen_shot()
        if image is not None and self.screen_shot_camera.image_color_mode == ScreenShotHelper.ImageColorMode.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
