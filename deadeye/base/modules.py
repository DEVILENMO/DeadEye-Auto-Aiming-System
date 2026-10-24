# coding: utf-8
# cython: language_level=3
import pyautogui

from deadeye.utils.math_tools import BoxKalmanFilter


class Target:
    """
    追踪目标：检测框由 BoxKalmanFilter 维护（等速模型，中心+尺寸）。

    生命周期字段由 TargetTracker 维护：
    - hits: 连续命中次数（达到 min_hits_to_confirm 后确认）
    - missed_frames: 连续漏检帧数（漏检期间由 KF 预测推进，超过上限删除）
    """

    def __init__(self, label, index, left_top, right_bottom):
        self.label = label
        self.index = index
        x1, y1 = left_top
        x2, y2 = right_bottom
        self.kf = BoxKalmanFilter((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)
        self.left_top = left_top
        self.right_bottom = right_bottom
        self.hits = 1
        self.missed_frames = 0
        self.update_time = 0

    def predict(self):
        """漏检帧：KF 先验推进，missed_frames +1。"""
        cx, cy, w, h = self.kf.predict()
        self._sync_box(cx, cy, w, h)
        self.missed_frames += 1

    def update_position(self, left_top, right_bottom):
        """命中帧：KF 修正，missed_frames 清零、hits +1。"""
        x1, y1 = left_top
        x2, y2 = right_bottom
        cx, cy, w, h = self.kf.update((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)
        self._sync_box(cx, cy, w, h)
        self.missed_frames = 0
        self.hits += 1
        self.update_time = 0

    def _sync_box(self, cx, cy, w, h):
        self.left_top = (int(round(cx - w * 0.5)), int(round(cy - h * 0.5)))
        self.right_bottom = (int(round(cx + w * 0.5)), int(round(cy + h * 0.5)))

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
    def get_image(self):
        raise NotImplementedError('Subclass must implement this method.')

    def on_exit(self):
        pass
