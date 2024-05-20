# coding: utf-8
# cython: language_level=3
import pyautogui

from MathTools import KalmanFilter


class Target:
    def __init__(self, label, index, left_top, right_bottom):
        self.label = label
        self.index = index
        self.left_top_kf = KalmanFilter(left_top[0], left_top[1])
        self.right_bottom_kf = KalmanFilter(right_bottom[0], right_bottom[1])
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.predict_time = 1

    def update_position(self, left_top, right_bottom):
        self.left_top = left_top
        self.right_bottom = right_bottom
        # predicted_left_top = self.left_top_kf.predict(left_top[0], left_top[1])
        # self.left_top = (0.5 * (predicted_left_top[0] + left_top[0]), 0.5 * (predicted_left_top[1] + left_top[1]))
        # predicted_right_bottom = self.right_bottom_kf.predict(right_bottom[0], right_bottom[1])
        # self.right_bottom = (0.5 * (predicted_right_bottom[0] + right_bottom[0]), 0.5 * (predicted_right_bottom[1] + right_bottom[1]))

    def __repr__(self):
        return f'label: {self.label}, id: {self.index}, box: {self.left_top}, {self.right_bottom}'


class DetectModule:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def target_detect(self, img) -> list:
        raise NotImplementedError('Subclass must implement this method.')

    def on_exit(self):
        """This function is called when the program is closed."""
        pass


class AutoAimModule:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def auto_shoot(self, target_list: list[Target]) -> None:
        raise NotImplementedError('Subclass must implement this method.')

    def auto_aim(self, target_list: list[Target]) -> tuple[int, int]:
        raise NotImplementedError('Subclass must implement this method.')

    @staticmethod
    def calculate_view_range_start_pos(view_range) -> tuple[int, int]:
        screen_width, screen_height = pyautogui.size()
        left_top_x = int(screen_width / 2 - view_range[0] / 2)
        left_top_y = int(screen_height / 2 - view_range[1] / 2)
        return left_top_x, left_top_y

    def on_exit(self):
        """This function is called when the program is closed."""
        pass


class MouseControlModule:
    def __init__(self):
        raise NotImplementedError('Subclass must implement this method.')

    def click_left_button(self):
        raise NotImplementedError('Subclass must implement this method.')

    def move_mouse(self, x: int, y: int):
        raise NotImplementedError('Subclass must implement this method.')
