# coding: utf-8
# cython: language_level=3
import threading
import time

from scipy.optimize import linear_sum_assignment

from BaseModules import *


class DeadEyeCore:
    def __init__(self, camera: BaseCamera, detect_module: DetectModule, aim_module: AutoAimModule):
        print('Initing DeadEye core system...')
        self.if_paused = True
        print('System is set to paused state while initing.')

        # detector
        self.detect_module = detect_module

        # auto aim
        self.aim_module = aim_module
        self.if_auto_shoot = False  # auto shoot state
        self.if_auto_aim = False  # auto aim state

        # target data
        self.target_list = []
        self.previous_targets_detected_time = None
        self.new_target_list = []
        self.target_num = 0  # total number of targets

        # fps
        self.image_update_time = None
        self.fps_timer = time.time()
        self.fps_counter = 0
        self.fps_displayer = None

        # multiple threading
        self.program_continued = threading.Semaphore(0)
        self.update_image = threading.Semaphore(0)
        self.image_updated = threading.Semaphore(0)
        self.target_updated = threading.Semaphore(0)

        # camera
        self.camera = camera
        self.image = None
        self.image_expired = True

        # start threads
        camera_thread = threading.Thread(target=self.camera_thread, args=())
        camera_thread.daemon = True
        camera_thread.start()
        print('Camera thread started.')
        target_detecting_therad = threading.Thread(target=self.target_detector, args=())
        target_detecting_therad.daemon = True
        target_detecting_therad.start()
        print('Target detecting thread started.')
        auto_aiming_thread = threading.Thread(target=self.auto_aim, args=())
        auto_aiming_thread.daemon = True
        auto_aiming_thread.start()
        print('Auto aiming thread started.')

    def on_exit(self):
        if not self.if_paused:
            self.switch_pause_state()
        time.sleep(1)

    def switch_pause_state(self):
        self.if_paused = not self.if_paused
        if not self.if_paused:
            self.program_continued.release(2)
        return self.if_paused

    def switch_auto_shoot_state(self):
        self.if_auto_shoot = not self.if_auto_shoot
        return self.if_auto_shoot

    def switch_auto_aim_state(self):
        self.if_auto_aim = not self.if_auto_aim
        return self.if_auto_aim

    def camera_thread(self):
        while 1:
            self.program_continued.acquire()
            self.fps_timer = time.time()
            while 1:
                if not self.if_paused:
                    t0 = time.time()
                    image = self.camera.get_image()
                    if image is None:
                        continue

                    self.image = image
                    self.image_updated.release()
                    self.image_update_time = time.time()
                    print(f'Screen shot time cost: {self.image_update_time - t0}s.')
                    # print('Screen shot size:', image.shape)
                    # cv2.imshow('image', image)
                    # cv2.waitKey(0)
                    self.update_image.acquire()
                else:
                    print('Paused.')
                    break

    def target_detector(self):
        while 1:
            self.image_updated.acquire()
            t0 = time.time()
            image_update_time = self.image_update_time
            # print(f'Using image update at {t0 - image_update_time}s ago.')
            image = self.image
            self.image = None
            self.update_image.release()
            self.new_target_list = self.detect_module.target_detect(image)
            print(f'Detected {len(self.new_target_list)} targets.')
            # for target in self.new_target_list:
            #     print(target)
            targets_detected_time = time.time()
            self.target_updated.release()

            target_detect_time_cost = time.time() - t0
            print(f'Detect time cost: {target_detect_time_cost}s.')
            total_time_cost = targets_detected_time - image_update_time
            # print(f'Total time cost: {total_time_cost}.')
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

    def auto_aim(self):
        while 1:
            self.target_updated.acquire()
            if len(self.new_target_list):
                # 利用旧目标与当前目标进行目标位置的优化
                self.opt_targets()
            else:
                self.target_list.clear()
            if len(self.target_list):
                # 自动瞄准
                if self.if_auto_aim:
                    x_movement, y_movement = self.aim_module.auto_aim(self.target_list)
                if self.if_auto_shoot:
                    self.aim_module.auto_shoot(self.target_list)

    def hungarian_algorithm(self):
        # 匈牙利算法，用于目标匹配
        # 计算目标数量
        matche_result = []
        previous_targets_num = len(self.target_list)
        targets_num = len(self.new_target_list)
        if previous_targets_num > 0 and targets_num > 0:
            # 创建二维矩阵记录目标之间的距离
            distances = [[0] * targets_num for i in range(previous_targets_num)]
            for i in range(previous_targets_num):
                for j in range(targets_num):
                    # if label is not same, skip
                    if self.target_list[i].label != self.new_target_list[j][0]:
                        continue
                    # 分别计算目标左上点与右下点曼哈顿距离并求和
                    distances[i][j] = abs(self.target_list[i].left_top[0] - self.new_target_list[j][1][0]) + \
                                      abs(self.target_list[i].left_top[1] - self.new_target_list[j][1][1]) + \
                                      abs(self.target_list[i].right_bottom[0] - self.new_target_list[j][2][0]) + \
                                      abs(self.target_list[i].right_bottom[1] - self.new_target_list[j][2][1])
            # 使用匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(distances)
            # 记录匹配结果
            for i in range(len(row_ind)):
                matche_result.append((row_ind[i], col_ind[i]))
        return matche_result

    def opt_targets(self):
        # 采用匈牙利算法追踪物体后用卡尔曼滤波算法对目标真实位置进行预测
        # t0 = time.time()
        target_match_result_list = self.hungarian_algorithm()  # 匈牙利算法
        # 统计匹配情况
        matched_previous_index_dict = {}
        matched_index_list = []
        for match_relation in target_match_result_list:
            matched_previous_index_dict[match_relation[0]] = match_relation[1]
            matched_index_list.append(match_relation[1])

        # 清除丢失目标
        expired_target_list = []
        for index, target in enumerate(self.target_list):
            if index not in matched_previous_index_dict.keys():
                expired_target_list.append(target)
                continue
            target.update_position(self.new_target_list[matched_previous_index_dict[index]][1],
                                   self.new_target_list[matched_previous_index_dict[index]][2])
        for expired_target in expired_target_list:
            self.target_list.remove(expired_target)

        # 创建新目标
        for index in range(0, len(self.new_target_list)):
            if index in matched_index_list:
                continue
            # 为新目标建立对象
            new_target = self.new_target_list[index]
            target = Target(new_target[0], self.target_num, new_target[1],
                            new_target[2])
            self.target_num = self.target_num + 1
            self.target_list.append(target)
        # print('位置预测用时：', time.time() - t0, 's')
