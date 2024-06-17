# coding: utf-8
# cython: language_level=3
import logging
from enum import Enum

import dxcam
import numpy as np
import pyautogui
import win32api
import win32con
from mss import mss


class ScreenShotHelper:
    def __init__(self, img_width, img_height, camera_type):
        print('Device info:', dxcam.device_info().replace('\n', ' '))
        print('Output info:', dxcam.output_info().replace('\n', ' '))

        self.logger = logging.getLogger('ScreenShotHelper Logger')

        self.view_width = img_width
        self.view_height = img_height

        # 初始化分辨率
        print('system32 system resolution：', (
            win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN),
            win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)))
        print('dxcam system resolution:', dxcam.output_res()[0])
        print('pyautogui system resolution:', pyautogui.size())
        self.resolution_x, self.resolution_y = 0, 0

        self.camera_type = camera_type
        # 初始化dx相机
        self.dx_camera, self.dx_view_range = self.init_dx_camera()
        # 初始化mss相机
        self.mss_camera, self.mss_view_range = self.init_mss_camera()
        print('dxcam view range:', self.dx_view_range)
        print('mss view range:', self.mss_view_range)

        if dxcam.output_res()[0] != pyautogui.size():
            self.logger.error('dxcam detected resolution does not match the system resolution.')
            self.logger.error('To use the dxcam library, the system resolution scaling must be set to 100%.')
            self.logger.error('Switching to use the mss library.')
            if self.camera_type == self.CameraType.DXCAM:
                self.camera_type = self.CameraType.MSS

        if self.camera_type == self.CameraType.DXCAM:
            self.image_color_mode = self.ImageColorMode.BGR
        elif self.camera_type == self.CameraType.MSS:
            self.image_color_mode = self.ImageColorMode.BGR

        print('Camera type:', self.camera_type)
        print('Image color mode:', self.image_color_mode)

    class CameraType(Enum):
        DXCAM = 0
        MSS = 1

    class ImageColorMode(Enum):
        RGB = 0
        BGR = 1

    def capture_screen_shot(self):
        if self.camera_type == self.CameraType.DXCAM:
            return self.dx_capture_screen_shot()
        elif self.camera_type == self.CameraType.MSS:
            return self.mss_capture_screen_img()

    """
    dxcam
    """

    def init_dx_camera(self):
        return dxcam.create(), self.calculate_dx_view_range(self.view_width, self.view_height)

    def calculate_dx_view_range(self, width, height):
        # dxcam相机使用dxcam的分辨率接口
        self.resolution_x, self.resolution_y = dxcam.output_res()[0]
        left = (self.resolution_x - width) // 2
        top = (self.resolution_y - height) // 2
        right = left + width - 1
        down = top + height - 1
        return int(left), int(top), int(right), int(down)

    def update_dx_view_range(self):
        if (self.resolution_x, self.resolution_y) != dxcam.output_res()[0]:
            self.dx_view_range = self.calculate_dx_view_range(self.view_width, self.view_height)

    def dx_capture_screen_shot(self):
        self.update_dx_view_range()
        return self.dx_camera.grab(region=self.dx_view_range)

    """
    mss
    """

    def init_mss_camera(self):
        return mss(), self.calculate_mss_view_range(self.view_width, self.view_height)

    def calculate_mss_view_range(self, width, height):
        # use pyautogui resolution result for mss camera_thread
        self.resolution_x, self.resolution_y = pyautogui.size()
        left = (self.resolution_x - width) // 2
        top = (self.resolution_y - height) // 2
        return {'top': top, 'left': left, 'width': width, 'height': height}

    def update_mss_view_range(self):
        if self.mss_view_range is None or (self.resolution_x, self.resolution_y) != pyautogui.size():
            self.mss_view_range = self.calculate_mss_view_range(self.view_width, self.view_height)

    def mss_capture_screen_img(self):
        self.update_mss_view_range()
        screen_shot = self.mss_camera.grab(self.mss_view_range)
        return np.array(screen_shot)