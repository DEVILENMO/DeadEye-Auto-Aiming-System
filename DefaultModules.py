# coding: utf-8
# cython: language_level=3
import time

import pyautogui
import pydirectinput
from pynput import mouse

from BaseModules import *
from TensorRTEngine import *
from Yolov7Helper import *


class YoloDetector(DetectModule):
    def __init__(self, weight: str):
        super().__init__()
        if '.pt' in weight:
            detector = Yolov7Helper(weight)
            self.model_type = 'pt'
        else:
            detector = TensorRTEngine(weight)
            self.model_type = 'onnx'
        self.detector = detector

    def target_detect(self, img) -> list:
        if self.model_type == 'pt':
            pass
        else:
            pass
        return []


class DeadEyeAutoAimingModule(AutoAimModule):
    def __init__(self, view_range):
        super().__init__()
        self.view_range = view_range

        # mouse controller
        self.mouse_controller = mouse.Controller()

        # Auto aim settings
        self.auto_aim_range_x = 1.5
        self.auto_aim_range_y = 1.5
        self.aim_sensitive = 1

        # Auto shoot settings
        self.last_auto_shoot_time = time.time()

        # PID
        self.max_movement = 64  # max output movement on single direction, to avoid large movement
        # PID核心参数 0.75/0.15/0.01
        k_adjust = 1
        self.k_p = 0.75 * k_adjust  # 比例系数 主要移动
        self.k_i = 0.15 * k_adjust  # 积分系数 补充移动
        self.k_d = 0.01 * k_adjust  # 微分系数 抑制
        # 回弹强度
        self.rebound_strength = 0.01  # 作用于积分系数，用于消除瞄准回弹效果
        # PID计算变量
        # 前一次时间，用于计算dt
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

    def auto_shoot(self, target_list: list):
        # 自动扳机
        t = time.time()
        if t - self.last_auto_shoot_time < 0.25:
            return

        mouseX, mouseY = pyautogui.position()
        # print('鼠标位置：', mouseX, ',', mouseY)

        # print('当前共有：', len(targets), '个目标')
        for target in target_list:
            tag, left_top, right_bottom = target.label, target.left_top, target.right_bottom
            if tag != 0:
                continue
            # print(tag, (left_top[0] + self.view_range[0], left_top[1] + self.view_range[1]),
            #       (right_bottom[0] + self.view_range[0], right_bottom[1] + self.view_range[1]))
            width = right_bottom[0] - left_top[0]
            height = right_bottom[1] - left_top[1]
            if left_top[0] + self.view_range[0] + 0.25 * width <= mouseX <= right_bottom[0] + self.view_range[
                0] - 0.25 * width:
                if left_top[1] + self.view_range[1] <= mouseY <= right_bottom[1] + self.view_range[1] - 0.75 * height:
                    # windll.user32.BlockInput(1)
                    # self.mouse_controller.click(pynput.mouse.Button.right)
                    self.shoot()
                    break
            if left_top[0] + self.view_range[0] + 0.15 * width <= mouseX <= right_bottom[0] + self.view_range[
                0] - 0.15 * width:
                if left_top[1] + self.view_range[1] <= mouseY <= right_bottom[1] + self.view_range[1] - 0.5 * height:
                    # windll.user32.BlockInput(1)
                    # self.mouse_controller.click(pynput.mouse.Button.right)
                    self.shoot()
                    self.last_auto_shoot_time = t
                    break
        # print('瞄准计算用时：', time.time() - t)
        return

    def auto_aim(self, target_list: list):
        t = time.time()
        mouseX, mouseY = pyautogui.position()
        # print('鼠标位置：', mouseX, ',', mouseY)

        if not len(target_list):
            return False
        # print('检测到', len(targets), '个目标')
        rel_mouse_x = mouseX - self.view_range[0]
        rel_mouse_y = mouseY - self.view_range[1]
        # print('相对鼠标位置：', rel_mouse_x, rel_mouse_y)
        # 寻找最近目标的时候，最终距离减去目标长度宽度，这样可以避免小目标出现在大目标附近时，实际距离更远的小目标成为最近的目标
        nearest_target = min(target_list, key=lambda k: abs((k.left_top[0] + k.right_bottom[0]) / 2 - rel_mouse_x) +
                                                        abs((k.left_top[1] + k.right_bottom[1]) / 2 - rel_mouse_y) -
                                                        (k.right_bottom[0] - k.left_top[0] + k.right_bottom[1] -
                                                         k.left_top[1]))
        width = nearest_target.right_bottom[0] - nearest_target.left_top[0]
        height = nearest_target.right_bottom[1] - nearest_target.left_top[1]
        distance_x = (nearest_target.left_top[0] + nearest_target.right_bottom[0]) / 2 - rel_mouse_x
        distance_y = (nearest_target.left_top[1] + nearest_target.right_bottom[1]) / 2 - rel_mouse_y
        if distance_x > width * self.auto_aim_range_x or distance_y > height * self.auto_aim_range_y:
            return False
        # print('最近目标：', nearest_target)
        # 移动到最近的目标

        position_fixed = (round((nearest_target.left_top[0] + nearest_target.right_bottom[0]) * 0.5),
                          round((nearest_target.left_top[1] + nearest_target.right_bottom[1]) * 0.5 - 0.25 * height))
        x_r, y_r = self.calculate_mouse_movement_by_pid(position_fixed, (rel_mouse_x, rel_mouse_y))  # 计算鼠标移动

        # 鼠标操控部分
        pydirectinput.moveRel(int(x_r * self.aim_sensitive),
                              int(y_r * self.aim_sensitive),
                              duration=0.000, relative=True)

    def shoot(self):
        self.mouse_controller.click(mouse.Button.left)

    def set_pid_parameters(self, p=0.75, i=0.15, d=0.01, rebond_strength=0.01):
        self.k_p = p
        self.k_i = i
        self.k_d = d
        self.rebound_strength = rebond_strength

    def calculate_mouse_movement_by_pid(self, target_position, mouse_position, only_update_info=False):
        # PID控制算法
        if self.previous_time is not None and time.time() - self.previous_time > 0.5:
            # 消除残像
            self.previous_time = None

        if self.previous_time is None:
            # 初始化PID
            self.previous_time = time.time()
            self.x_integral_value = 0
            self.y_integral_value = 0
            self.previous_distance_x = 0
            self.previous_distance_y = 0

        # 绝对偏差
        current_time = time.time()
        distance_x = target_position[0] - mouse_position[0]
        if distance_x * self.previous_distance_x < 0:
            self.x_integral_value = self.x_integral_value * self.rebound_strength  # 降低积分量，减少回弹
        distance_y = target_position[1] - mouse_position[1]
        if distance_y * self.previous_distance_y < 0:
            self.y_integral_value = self.y_integral_value * self.rebound_strength  # 降低积分量，减少回弹
        # self.distance_list.append(distance_x)

        x_r, y_r = 0, 0  # 初始化返回值
        if not only_update_info:
            # P/比例
            x_p = self.k_p * distance_x
            y_p = self.k_p * distance_y

            # I/积分
            d_time = current_time - self.previous_time
            self.x_integral_value = self.x_integral_value + distance_x
            x_i = self.k_i * self.x_integral_value
            self.y_integral_value = self.y_integral_value + distance_y
            y_i = self.k_i * self.y_integral_value

            # D/微分
            if d_time != 0:
                derivative_x = (distance_x - self.previous_distance_x)
                x_d = self.k_d * derivative_x
                derivative_y = (distance_y - self.previous_distance_y)
                y_d = self.k_d * derivative_y
            else:
                x_d = 0
                y_d = 0

            # 结果
            x_r = x_p + x_i + x_d
            # print(x_p, x_i, x_d)
            y_r = y_p + y_i + y_d
            # print(y_p, y_i, y_d)

            # 极大值约束
            if abs(x_r) > self.max_movement:
                x_r = x_r / abs(x_r) * self.max_movement
            if abs(y_r) > self.max_movement:
                y_r = y_r / abs(y_r) * self.max_movement

        # 更新旧信息
        self.previous_time = current_time
        self.previous_distance_x = distance_x
        self.previous_distance_y = distance_y

        return x_r, y_r
