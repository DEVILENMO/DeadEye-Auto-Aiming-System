# coding: utf-8
# cython: language_level=3
import threading
import time

from scipy.optimize import linear_sum_assignment

from deadeye.base.modules import BaseCamera, DetectModule, ExecutionModule, Target


class DeadEyeCore:
    def __init__(
        self,
        camera: BaseCamera,
        detect_module: DetectModule,
        execution_modules: list[ExecutionModule] | None = None,
    ):
        print('Initing DeadEye core system...')
        self.if_paused = True
        print('System is set to paused state while initing.')

        self.detect_module = detect_module
        self.execution_modules = execution_modules or []

        self.target_list = []
        self.previous_targets_detected_time = None
        self.new_target_list = []
        self.target_num = 0

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
                        continue

                    self.image = image
                    self.image_updated.release()
                    self.image_update_time = time.time()
                    print(f'Screen shot time cost: {self.image_update_time - t0}s.')
                    self.update_image.acquire()
                else:
                    print('Paused.')
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
            print(f'Detected {len(self.new_target_list)} targets.')
            targets_detected_time = time.time()
            self.target_updated.release()

            target_detect_time_cost = time.time() - t0
            print(f'Detect time cost: {target_detect_time_cost}s.')
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

            if len(self.new_target_list):
                self.opt_targets()
            else:
                self.target_list.clear()

            if frame_for_output is not None:
                for m in self.execution_modules:
                    try:
                        m.update_targets(frame_for_output, list(self.target_list))
                    except Exception:
                        continue

    def hungarian_algorithm(self):
        matche_result = []
        previous_targets_num = len(self.target_list)
        targets_num = len(self.new_target_list)
        if previous_targets_num > 0 and targets_num > 0:
            distances = [[0] * targets_num for i in range(previous_targets_num)]
            for i in range(previous_targets_num):
                for j in range(targets_num):
                    if self.target_list[i].label != self.new_target_list[j][0]:
                        continue
                    distances[i][j] = abs(self.target_list[i].left_top[0] - self.new_target_list[j][1][0]) + \
                                      abs(self.target_list[i].left_top[1] - self.new_target_list[j][1][1]) + \
                                      abs(self.target_list[i].right_bottom[0] - self.new_target_list[j][2][0]) + \
                                      abs(self.target_list[i].right_bottom[1] - self.new_target_list[j][2][1])
            row_ind, col_ind = linear_sum_assignment(distances)
            for i in range(len(row_ind)):
                matche_result.append((row_ind[i], col_ind[i]))
        return matche_result

    def opt_targets(self):
        target_match_result_list = self.hungarian_algorithm()
        matched_previous_index_dict = {}
        matched_index_list = []
        for match_relation in target_match_result_list:
            matched_previous_index_dict[match_relation[0]] = match_relation[1]
            matched_index_list.append(match_relation[1])

        expired_target_list = []
        for index, target in enumerate(self.target_list):
            if index not in matched_previous_index_dict.keys():
                expired_target_list.append(target)
                continue
            target.update_position(self.new_target_list[matched_previous_index_dict[index]][1],
                                   self.new_target_list[matched_previous_index_dict[index]][2])
        for expired_target in expired_target_list:
            self.target_list.remove(expired_target)

        for index in range(0, len(self.new_target_list)):
            if index in matched_index_list:
                continue
            new_target = self.new_target_list[index]
            target = Target(new_target[0], self.target_num, new_target[1],
                            new_target[2])
            self.target_num = self.target_num + 1
            self.target_list.append(target)
