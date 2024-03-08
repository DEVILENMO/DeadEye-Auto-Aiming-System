# coding: utf-8
# cython: language_level=3
from MathTools import KalmanFilter


class DetectModule:
    def __init__(self):
        pass

    def target_detect(self, img) -> list:
        raise NotImplementedError('Subclass must implement this method.')


class AutoAimModule:
    def __init__(self):
        pass

    def auto_shoot(self, target_list: list) -> None:
        raise NotImplementedError('Subclass must implement this method.')

    def auto_aim(self, target_list: list) -> bool:
        raise NotImplementedError('Subclass must implement this method.')


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
        predicted_left_top = self.left_top_kf.predict(left_top[0], left_top[1])
        self.left_top = (0.5 * (predicted_left_top[0] + left_top[0]), 0.5 * (predicted_left_top[1] + left_top[1]))
        predicted_right_bottom = self.right_bottom_kf.predict(right_bottom[0], right_bottom[1])
        self.right_bottom = (0.5 * (predicted_right_bottom[0] + right_bottom[0]), 0.5 * (predicted_right_bottom[1] + right_bottom[1]))
