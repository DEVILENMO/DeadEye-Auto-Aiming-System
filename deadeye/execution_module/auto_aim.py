# coding: utf-8
# cython: language_level=3
import time

import pyautogui
from pynput import mouse

from deadeye.base.modules import ExecutionModule, MouseControlModule, Target
from deadeye.base.configurable import ConfigurableMixin


class SimpleMouseController(MouseControlModule):
    def __init__(self):
        super(SimpleMouseController).__init__()
        self.mc = mouse.Controller()

    def click_left_button(self):
        print('You are required to implement a mouse click function yourself')

    def move_mouse(self, x: int, y: int):
        print('You are required to implement a mouse move function yourself')


DEFAULT_P = 0.5
DEFEULT_I = 0.2
DEFAULT_D = 0.05
DEFAULT_R = 0.5


class DeadEyeAutoAimingModule(ExecutionModule, ConfigurableMixin):
    @classmethod
    def declare_config(cls) -> None:
        cls.register_bool('enable_auto_aim', False, label='启动自动瞄准')
        cls.register_bool('enable_auto_shoot', False, label='启动自动扳机')
        cls.register_float('auto_aim_range_x', 1.5, label='瞄准范围 X', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_float('auto_aim_range_y', 1.5, label='瞄准范围 Y', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_float('aim_sensitive', 1.0, label='灵敏度', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_int('max_movement', 320, label='最大移动', min_value=1, max_value=5000, step=1)
        cls.register_float('k_p', DEFAULT_P, label='PID Kp', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('k_i', DEFEULT_I, label='PID Ki', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('k_d', DEFAULT_D, label='PID Kd', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('rebound_strength', DEFAULT_R, label='回弹强度', min_value=0.0, max_value=1.0, step=0.01)

    def __init__(
        self,
        view_range,
        *,
        enable_auto_aim: bool = False,
        enable_auto_shoot: bool = False,
        auto_aim_range_x: float = 1.5,
        auto_aim_range_y: float = 1.5,
        aim_sensitive: float = 1.0,
        max_movement: int = 320,
        k_p: float = DEFAULT_P,
        k_i: float = DEFEULT_I,
        k_d: float = DEFAULT_D,
        rebound_strength: float = DEFAULT_R,
    ):
        super().__init__()
        self.view_range_start = self.calculate_view_range_start_pos(view_range)

        self.tracking_target_id = None

        self.mouse_controller = SimpleMouseController()

        self.auto_aim_range_x = float(auto_aim_range_x)
        self.auto_aim_range_y = float(auto_aim_range_y)
        self.aim_sensitive = float(aim_sensitive)

        self.last_auto_shoot_time = time.time()

        self.max_movement = int(max_movement)
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.rebound_strength = float(rebound_strength)
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

        self.enable_auto_aim = bool(enable_auto_aim)
        self.enable_auto_shoot = bool(enable_auto_shoot)

    def auto_shoot(self, target_list: list[Target]) -> None:
        t = time.time()
        if t - self.last_auto_shoot_time < 0.25:
            return

        mouse_x, mouse_y = pyautogui.position()

        for target in target_list:
            left_top, right_bottom = target.left_top, target.right_bottom
            width = right_bottom[0] - left_top[0]
            height = right_bottom[1] - left_top[1]
            if left_top[0] + self.view_range_start[0] + 0.25 * width <= mouse_x <= right_bottom[0] + \
                    self.view_range_start[0] - 0.25 * width:
                if left_top[1] + self.view_range_start[1] <= mouse_y <= right_bottom[1] + self.view_range_start[
                    1] - 0.75 * height:
                    self.shoot()
                    break
            if left_top[0] + self.view_range_start[0] + 0.15 * width <= mouse_x <= right_bottom[0] + \
                    self.view_range_start[0] - 0.15 * width:
                if left_top[1] + self.view_range_start[1] <= mouse_y <= right_bottom[1] + self.view_range_start[
                    1] - 0.5 * height:
                    self.shoot()
                    self.last_auto_shoot_time = t
                    break
        return

    def update_targets(self, image, target_list: list[Target]) -> None:
        # image 参数在该模块中不使用，保留以满足统一接口
        if not target_list:
            self.set_tracking_target_id(None)
            return
        if self.enable_auto_aim:
            self.auto_aim(target_list)
        if self.enable_auto_shoot:
            self.auto_shoot(target_list)

    def set_tracking_target_id(self, to_track_target_id: [int | None]):
        self.tracking_target_id = to_track_target_id
        self.clear_pid_history()

    def auto_aim(self, target_list: list[Target]) -> tuple[int, int]:
        mouse_x, mouse_y = pyautogui.position()

        if not len(target_list):
            self.set_tracking_target_id(None)
            return 0, 0
        rel_mouse_x = mouse_x - self.view_range_start[0]
        rel_mouse_y = mouse_y - self.view_range_start[1]

        aim_target = None
        if self.tracking_target_id is None:
            aim_target = self.find_closest_target(target_list, rel_mouse_x, rel_mouse_y)
            self.set_tracking_target_id(aim_target.index)
        else:
            for target in target_list:
                if target.index == self.tracking_target_id:
                    aim_target = target
                    break
            if aim_target is None:
                aim_target = self.find_closest_target(target_list, rel_mouse_x, rel_mouse_y)
                self.set_tracking_target_id(aim_target.index)

        width = aim_target.right_bottom[0] - aim_target.left_top[0]
        height = aim_target.right_bottom[1] - aim_target.left_top[1]
        distance_x = abs((aim_target.left_top[0] + aim_target.right_bottom[0]) / 2 - rel_mouse_x)
        distance_y = abs((aim_target.left_top[1] + aim_target.right_bottom[1]) / 2 - rel_mouse_y)
        if distance_x > width * self.auto_aim_range_x or distance_y > height * self.auto_aim_range_y:
            return 0, 0
        position_fixed = (round((aim_target.left_top[0] + aim_target.right_bottom[0]) * 0.5),
                          round((aim_target.left_top[1] + aim_target.right_bottom[1]) * 0.5))
        x_r, y_r = self.calculate_mouse_movement_by_pid(position_fixed, (rel_mouse_x, rel_mouse_y))
        x_r = int(x_r * self.aim_sensitive)
        y_r = int(y_r * self.aim_sensitive)

        self.mouse_controller.move_mouse(x_r, y_r)
        return x_r, y_r

    @staticmethod
    def find_closest_target(target_list: list[Target], mouse_pos_x: int, mouse_pos_y: int) -> Target:
        nearest_target = min(target_list, key=lambda k: abs((k.left_top[0] + k.right_bottom[0]) / 2 - mouse_pos_x) +
                                                        abs((k.left_top[1] + k.right_bottom[1]) / 2 - mouse_pos_y) -
                                                        (k.right_bottom[0] - k.left_top[0] + k.right_bottom[1] -
                                                         k.left_top[1]))
        return nearest_target

    def shoot(self):
        self.mouse_controller.click_left_button()

    def set_pid_parameters(self, p=None, i=None, d=None, rebond_strength=None):
        if p is not None:
            self.k_p = p
        if i is not None:
            self.k_i = i
        if d is not None:
            self.k_d = d
        if rebond_strength is not None:
            self.rebound_strength = rebond_strength

    def clear_pid_history(self):
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

    def calculate_mouse_movement_by_pid(self, target_position, mouse_position):
        if self.previous_time is not None and time.time() - self.previous_time > 0.5:
            self.clear_pid_history()

        current_time = time.time()
        distance_x = target_position[0] - mouse_position[0]
        if distance_x * self.previous_distance_x < 0:
            self.x_integral_value = self.x_integral_value * self.rebound_strength
        distance_y = target_position[1] - mouse_position[1]
        if distance_y * self.previous_distance_y < 0:
            self.y_integral_value = self.y_integral_value * self.rebound_strength

        x_p = self.k_p * distance_x
        y_p = self.k_p * distance_y

        if self.previous_time is not None:
            d_time = current_time - self.previous_time
            self.x_integral_value = self.x_integral_value + distance_x
            x_i = self.k_i * self.x_integral_value
            self.y_integral_value = self.y_integral_value + distance_y
            y_i = self.k_i * self.y_integral_value

            if d_time != 0:
                derivative_x = (distance_x - self.previous_distance_x) / d_time
                x_d = self.k_d * derivative_x
                derivative_y = (distance_y - self.previous_distance_y) / d_time
                y_d = self.k_d * derivative_y
            else:
                x_d = 0
                y_d = 0
        else:
            x_i = 0
            y_i = 0
            x_d = 0
            y_d = 0

        x_r = x_p + x_i + x_d
        y_r = y_p + y_i + y_d

        if abs(x_r) > self.max_movement:
            x_r = x_r / abs(x_r) * self.max_movement
        if abs(y_r) > self.max_movement:
            y_r = y_r / abs(y_r) * self.max_movement

        self.previous_time = current_time
        self.previous_distance_x = distance_x
        self.previous_distance_y = distance_y
        return x_r, y_r
