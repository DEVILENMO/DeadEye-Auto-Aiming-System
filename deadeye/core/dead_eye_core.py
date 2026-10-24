# coding: utf-8
# cython: language_level=3
"""
DeadEye 核心流水线：相机线程 -> 检测线程 -> 追踪/执行线程。

追踪逻辑（匹配 + 卡尔曼 + 目标生命周期）委托给 TargetTracker，
参数（匹配 IoU 门控 / 漏检保留帧数 / 确认命中帧数）可通过构造函数或
ConfigurableMixin 表单配置。
"""
import threading
import time

from deadeye.base.configurable import ConfigurableMixin
from deadeye.base.modules import BaseCamera, DetectModule, ExecutionModule
from deadeye.core.target_tracker import TargetTracker


class DeadEyeCore(ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_float('match_iou_threshold', 0.3, label='匹配 IoU 门控', min_value=0.05, max_value=0.95, step=0.05)
        cls.register_int('max_missed_frames', 5, label='漏检保留帧数', min_value=0, max_value=60, step=1)
        cls.register_int('min_hits_to_confirm', 1, label='确认命中帧数', min_value=1, max_value=10, step=1)

    def __init__(
        self,
        camera: BaseCamera,
        detect_module: DetectModule,
        execution_modules: list[ExecutionModule] | None = None,
        *,
        tracker: TargetTracker | None = None,
        match_iou_threshold: float = 0.3,
        max_missed_frames: int = 5,
        min_hits_to_confirm: int = 1,
        verbose: bool = False,
    ):
        print('Initing DeadEye core system...')
        self.if_paused = True
        print('System is set to paused state while initing.')

        self.verbose = verbose  # 逐帧调试打印开关（高帧率下建议关闭）
        self.detect_module = detect_module
        self.execution_modules = execution_modules or []

        self.tracker = tracker or TargetTracker(
            match_iou_threshold=match_iou_threshold,
            max_missed_frames=max_missed_frames,
            min_hits_to_confirm=min_hits_to_confirm,
        )
        self.target_list = self.tracker.targets
        self.new_target_list = []

        self.image_update_time = None
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.fps_displayer = None

        self._last_frame_for_output = None

        self.program_continued = threading.Semaphore(0)
        self.update_image = threading.Semaphore(0)
        self.image_updated = threading.Semaphore(0)
        self.target_updated = threading.Semaphore(0)

        self.camera = camera
        self.image = None
        self.image_expired = True

        camera_thread = threading.Thread(target=self.camera_thread, args=())
        camera_thread.daemon = True
        camera_thread.start()
        print('Camera thread started.')
        target_detecting_therad = threading.Thread(target=self.target_detector, args=())
        target_detecting_therad.daemon = True
        target_detecting_therad.start()
        print('Target detecting thread started.')
        target_update_thread = threading.Thread(target=self.target_update_thread, args=())
        target_update_thread.daemon = True
        target_update_thread.start()
        print('Target update thread started.')

    def on_exit(self):
        if not self.if_paused:
            self.switch_pause_state()
        time.sleep(1)
        if hasattr(self.camera, 'on_exit'):
            self.camera.on_exit()
        for m in self.execution_modules:
            try:
                m.on_exit()
            except Exception:
                pass

    def switch_pause_state(self):
        self.if_paused = not self.if_paused
        if not self.if_paused:
            self.program_continued.release(2)
        return self.if_paused

    def camera_thread(self):
        while 1:
            self.program_continued.acquire()
            self.fps_timer = time.time()
            while 1:
                if not self.if_paused:
                    t0 = time.time()
                    image = self.camera.get_image()
                    if image is None:
                        if getattr(self.camera, 'stream_exhausted', False):
                            print('输入流已结束，自动暂停。')
                            self.if_paused = True
                            break
                        # 无新帧时短暂让出 CPU，避免忙等抢占 GIL 饿死截屏回调线程
                        time.sleep(0.001)
                        continue

                    self.image = image
                    self.image_updated.release()
                    self.image_update_time = time.time()
                    if self.verbose:
                        print(f'Screen shot time cost: {self.image_update_time - t0}s.')
                    self.update_image.acquire()
                else:
                    break

    def target_detector(self):
        while 1:
            self.image_updated.acquire()
            t0 = time.time()
            image_update_time = self.image_update_time
            image = self.image
            self.image = None
            self.update_image.release()

            if self.execution_modules and image is not None:
                self._last_frame_for_output = image.copy()
            else:
                self._last_frame_for_output = None

            self.new_target_list = self.detect_module.target_detect(image)
            self.target_updated.release()

            if self.verbose:
                print(f'Detected {len(self.new_target_list)} targets, '
                      f'time cost: {time.time() - t0}s.')
            current_time = time.time()
            if current_time - self.fps_timer >= 1:
                fps = self.fps_counter / (current_time - self.fps_timer)
                if self.fps_displayer:
                    self.fps_displayer.set(f"{round(fps)}")
                else:
                    print('FPS:', fps)
                self.fps_counter = 0
                self.fps_timer = current_time
            else:
                self.fps_counter += 1

    def target_update_thread(self):
        while 1:
            self.target_updated.acquire()
            frame_for_output = self._last_frame_for_output

            # 空检测帧也走 update：目标由 KF 预测推进并在 max_missed_frames 内保留
            self.target_list = self.tracker.update(self.new_target_list)

            if frame_for_output is not None:
                for m in self.execution_modules:
                    try:
                        m.update_targets(frame_for_output, list(self.target_list))
                    except Exception:
                        continue
