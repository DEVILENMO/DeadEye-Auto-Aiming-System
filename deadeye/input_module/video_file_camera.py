# coding: utf-8
import os

import cv2
import numpy as np

from deadeye.base.modules import BaseCamera
from deadeye.base.configurable import ConfigurableMixin


def _center_crop_or_resize(frame_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """将帧处理为 out_w x out_h，不足时先放大再中心裁剪。"""
    h, w = frame_bgr.shape[:2]
    if w < out_w or h < out_h:
        scale = max(out_w / w, out_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = frame_bgr.shape[:2]
    x1 = (w - out_w) // 2
    y1 = (h - out_h) // 2
    return frame_bgr[y1:y1 + out_h, x1:x1 + out_w].copy()


class VideoFileCamera(BaseCamera, ConfigurableMixin):
    """
    从单个视频文件读取帧；每帧中心裁剪/缩放到 view_range，输出 RGB（与截屏相机一致）。
    """

    @classmethod
    def declare_config(cls) -> None:
        cls.register_path(
            'video_path',
            '',
            label='视频文件',
            filetypes=[
                ('视频', '*.mp4 *.avi *.mov *.mkv *.webm *.m4v'),
                ('All files', '*.*'),
            ],
        )

    def __init__(self, view_range: tuple[int, int], *, video_path: str):
        video_path = os.path.abspath(video_path.strip())
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f'视频文件不存在: {video_path!r}')
        self._video_paths = [video_path]
        self._view_w, self._view_h = int(view_range[0]), int(view_range[1])
        self._path_index = 0
        self._cap = None
        self._reported_fps = 30.0
        self.stream_exhausted = False
        if not self._open_next_capture():
            self.stream_exhausted = True

    def _open_next_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        while self._path_index < len(self._video_paths):
            path = self._video_paths[self._path_index]
            self._path_index += 1
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f'无法打开视频，跳过: {path}')
                cap.release()
                continue
            self._cap = cap
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self._reported_fps = float(fps)
            print(f'视频输入: {path}, 报告 FPS: {self._reported_fps}')
            return True
        self._cap = None
        return False

    def get_reported_fps(self) -> float:
        return self._reported_fps

    def get_image(self):
        if self.stream_exhausted:
            return None
        if self._cap is None:
            self.stream_exhausted = True
            return None
        ok, frame_bgr = self._cap.read()
        if not ok:
            if not self._open_next_capture():
                self.stream_exhausted = True
                return None
            ok, frame_bgr = self._cap.read()
            if not ok:
                self.stream_exhausted = True
                return None
        cropped = _center_crop_or_resize(frame_bgr, self._view_w, self._view_h)
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    def on_exit(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
