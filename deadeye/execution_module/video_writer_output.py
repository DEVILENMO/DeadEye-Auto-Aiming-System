# coding: utf-8
import atexit
import os

import cv2
import numpy as np

from deadeye.base.modules import ExecutionModule, Target
from deadeye.base.configurable import ConfigurableMixin


class VideoWriterOutput(ExecutionModule, ConfigurableMixin):
    """
    将流水线中的 RGB 帧与追踪后的 target_list 画框与 ID 后写入视频。
    """

    @classmethod
    def declare_config(cls) -> None:
        cls.register_path('output_path', os.path.abspath(os.path.join('.', 'output', 'annotated.mp4')), label='输出视频路径')
        cls.register_choice('fourcc', 'mp4v', label='编码器', choices=['mp4v', 'XVID'])

    def __init__(
        self,
        output_path: str,
        fps: float | None = None,
        fourcc: str | None = None,
        input_video_path: str | None = None,
    ):
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._output_path = output_path
        self._fps = max(1.0, float(fps)) if fps is not None else 30.0
        self._fourcc_str = fourcc or self._infer_fourcc_from_path(output_path)
        self._writer = None
        self._size = None
        atexit.register(self.on_exit)
        if input_video_path:
            self._try_read_fps_from_video(input_video_path)

    @staticmethod
    def _infer_fourcc_from_path(output_path: str) -> str:
        _, ext = os.path.splitext(output_path)
        ext = ext.lower()
        if ext == '.avi':
            return 'XVID'
        if ext == '.mp4':
            return 'mp4v'
        return 'mp4v'

    def _try_read_fps_from_video(self, input_video_path: str):
        if not os.path.isfile(input_video_path):
            return
        cap = cv2.VideoCapture(input_video_path)
        try:
            if not cap.isOpened():
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self._fps = float(fps)
        finally:
            cap.release()

    def _ensure_writer(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        self._size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc_str)
        self._writer = cv2.VideoWriter(self._output_path, fourcc, self._fps, self._size)
        if not self._writer.isOpened():
            raise RuntimeError(f'无法创建视频写入器: {self._output_path}')

    def update_targets(self, image, target_list: list[Target]) -> None:
        if image is None:
            return
        if self._writer is None:
            self._ensure_writer(image)
        frame_rgb = np.ascontiguousarray(image)
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for target in target_list:
            if not isinstance(target, Target):
                continue
            lt = target.left_top
            rb = target.right_bottom
            cv2.rectangle(frame_bgr, lt, rb, (0, 255, 0), 2)
            label_text = f'ID:{target.index} {target.label}'
            cv2.putText(
                frame_bgr,
                label_text,
                (lt[0], max(0, lt[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        self._writer.write(frame_bgr)

    def on_exit(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f'视频已保存: {self._output_path}')
