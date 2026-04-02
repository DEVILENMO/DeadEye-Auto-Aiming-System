# coding: utf-8
# cython: language_level=3
import pyautogui

from deadeye.utils.math_tools import KalmanFilter


class Target:
    def __init__(self, label, index, left_top, right_bottom):
        self.label = label
        self.index = index
        self.left_top_kf = KalmanFilter(left_top[0], left_top[1])
        self.right_bottom_kf = KalmanFilter(right_bottom[0], right_bottom[1])
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.update_time = 0

    def update_position(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom

    def __repr__(self):
        return f'label: {self.label}, id: {self.index}, box: {self.left_top}, {self.right_bottom}'


class DetectModule:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def target_detect(self, img) -> list:
        raise NotImplementedError('Subclass must implement this method.')

    def on_exit(self):
        pass


class ExecutionModule:
    """
    执行模块：检测+追踪完成后，对当前帧与目标列表做后续动作（鼠标、写视频、电压输出等）。
    """

    def update_targets(self, image, target_list: list[Target]) -> None:
        raise NotImplementedError('Subclass must implement this method.')

    @staticmethod
    def calculate_view_range_start_pos(view_range) -> tuple[int, int]:
        screen_width, screen_height = pyautogui.size()
        left_top_x = int(screen_width / 2 - view_range[0] / 2)
        left_top_y = int(screen_height / 2 - view_range[1] / 2)
        return left_top_x, left_top_y

    def on_exit(self):
        pass


class MouseControlModule:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def click_left_button(self):
        raise NotImplementedError('Subclass must implement this method.')

    def move_mouse(self, x: int, y: int):
        raise NotImplementedError('Subclass must implement this method.')


class BaseCamera:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def get_image(self):
        raise NotImplementedError('Subclass must implement this method.')
