# coding: utf-8
# cython: language_level=3
import math
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


DEFAULT_P = 0.3
DEFEULT_I = 0.3
DEFAULT_D = 0.01
DEFAULT_R = 0.5
DEFAULT_MAX_STEP = 15.0


class DeadEyeAutoAimingModule(ExecutionModule, ConfigurableMixin):
    """
    PID 辅助瞄准执行模块。

    控制结构（取长补短后的完整形态）：
    - 误差整形：死区（默认 1.5px，可设 0.5px 做亚像素死区）消除锁定后的高频微颤；
      可选 atan 软饱和整形 shaped(e) = B·atan(A·(2|e|−dz))——小误差近似线性（斜率 atan_slope），
      大误差平滑封顶（上限 atan_ceiling，无 clamp 折角），"甩枪像人"的曲线来源；
    - 双轴 PID：X/Y 独立参数，Y 轴微分默认关闭（垂直框边缘抖动大，D 项易成噪声放大器）；
    - 增益调度：Kp 随目标速度连续放大（Kp_eff = Kp·(1 + schedule·|v|)），避免离散切换的抖振；
    - 速度前馈：out += vff · v_target，v_target 来自追踪层卡尔曼滤波的速度估计，
      直接补偿目标运动带来的相位滞后，比加大 Kp 更不易过冲；
    - 积分回弹：误差符号翻转时积分衰减（anti-windup），输出限幅 max_movement。
    """

    @classmethod
    def declare_config(cls) -> None:
        cls.register_bool('enable_auto_aim', False, label='启动自动瞄准')
        cls.register_bool('enable_auto_shoot', False, label='启动自动扳机')
        cls.register_float('auto_aim_range_x', 1.5, label='瞄准范围 X', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_float('auto_aim_range_y', 1.5, label='瞄准范围 Y', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_float('aim_sensitive', 1.0, label='灵敏度', min_value=0.1, max_value=10.0, step=0.1)
        cls.register_int('max_movement', 320, label='最大移动', min_value=1, max_value=5000, step=1)
        cls.register_float('max_step_per_frame', DEFAULT_MAX_STEP, label='输出限速(px/帧)', min_value=0.0, max_value=200.0, step=1.0)
        cls.register_float('k_p', DEFAULT_P, label='PID Kp', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('k_i', DEFEULT_I, label='PID Ki', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('k_d', DEFAULT_D, label='PID Kd(X轴)', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('k_d_y', 0.0, label='PID Kd(Y轴,默认关)', min_value=0.0, max_value=10.0, step=0.01)
        cls.register_float('deadzone_x', 1.5, label='死区 X(px)', min_value=0.0, max_value=50.0, step=0.5)
        cls.register_float('deadzone_y', 1.5, label='死区 Y(px)', min_value=0.0, max_value=50.0, step=0.5)
        cls.register_choice('shaping', 'linear', label='误差整形', choices=['linear', 'atan'])
        cls.register_float('atan_slope', 1.0, label='atan 整形斜率', min_value=0.1, max_value=5.0, step=0.1)
        cls.register_float('atan_ceiling', 300.0, label='atan 整形上限(px)', min_value=10.0, max_value=1000.0, step=10.0)
        cls.register_float('velocity_gain_schedule', 0.0, label='增益调度(每px/frame速度)', min_value=0.0, max_value=2.0, step=0.01)
        cls.register_float('velocity_feedforward_x', 0.0, label='速度前馈 X', min_value=0.0, max_value=2.0, step=0.01)
        cls.register_float('velocity_feedforward_y', 0.0, label='速度前馈 Y', min_value=0.0, max_value=2.0, step=0.01)
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
        max_step_per_frame: float = DEFAULT_MAX_STEP,
        k_p: float = DEFAULT_P,
        k_i: float = DEFEULT_I,
        k_d: float = DEFAULT_D,
        k_d_y: float = 0.0,
        deadzone_x: float = 1.5,
        deadzone_y: float = 1.5,
        shaping: str = 'linear',
        atan_slope: float = 1.0,
        atan_ceiling: float = 300.0,
        velocity_gain_schedule: float = 0.0,
        velocity_feedforward_x: float = 0.0,
        velocity_feedforward_y: float = 0.0,
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
        # 输出限速：输出即 px/帧（速度本身），按轴限幅构成甩枪速度天花板（0=关闭）
        self.max_step_per_frame = float(max_step_per_frame)
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.k_d_y = float(k_d_y)
        self.deadzone_x = float(deadzone_x)
        self.deadzone_y = float(deadzone_y)
        self.shaping = str(shaping) if str(shaping) in ('linear', 'atan') else 'linear'
        self.atan_slope = float(atan_slope)
        self.atan_ceiling = float(atan_ceiling)
        self.velocity_gain_schedule = float(velocity_gain_schedule)
        self.velocity_feedforward_x = float(velocity_feedforward_x)
        self.velocity_feedforward_y = float(velocity_feedforward_y)
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
        # 目标速度来自追踪层卡尔曼滤波（px/帧）：前馈与增益调度使用
        target_velocity = (float(aim_target.kf.x[4]), float(aim_target.kf.x[5]))
        x_r, y_r = self.calculate_mouse_movement_by_pid(position_fixed, (rel_mouse_x, rel_mouse_y), target_velocity)
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

    def set_pid_parameters(self, p=None, i=None, d=None, d_y=None, rebond_strength=None):
        if p is not None:
            self.k_p = p
        if i is not None:
            self.k_i = i
        if d is not None:
            self.k_d = d
        if d_y is not None:
            self.k_d_y = d_y
        if rebond_strength is not None:
            self.rebound_strength = rebond_strength

    def clear_pid_history(self):
        self.previous_time = None
        self.x_integral_value = 0
        self.y_integral_value = 0
        self.previous_distance_x = 0
        self.previous_distance_y = 0

    def _shape_error(self, error: float, deadzone: float) -> float:
        """
        误差整形：死区内归零；linear 模式原样通过；atan 模式软饱和。

        atan 模式：shaped(e) = B·atan(A·(2|e| − dz))·sign(e)
        - 小误差 ≈ atan_slope·(e − dz·sign(e))（近似线性，斜率可控）
        - 大误差平滑封顶于 atan_ceiling（无 clamp 折角）
        - 参数映射：B = 2C/π, A = slope·π/(4C)
        """
        magnitude = abs(error)
        if magnitude <= deadzone:
            return 0.0
        effective = 2.0 * magnitude - deadzone
        if self.shaping != 'atan':
            return error
        b = 2.0 * self.atan_ceiling / math.pi
        a = self.atan_slope * math.pi / (4.0 * self.atan_ceiling)
        return math.copysign(b * math.atan(a * effective), error)

    def calculate_mouse_movement_by_pid(self, target_position, mouse_position, target_velocity=None):
        """
        双轴 PID + 误差整形 + 连续增益调度 + 速度前馈。

        target_velocity: (vx, vy) px/帧（来自 Target 卡尔曼滤波），None 视为 (0, 0)。
        """
        vx, vy = target_velocity if target_velocity is not None else (0.0, 0.0)

        if self.previous_time is not None and time.time() - self.previous_time > 0.5:
            self.clear_pid_history()

        current_time = time.time()
        distance_x = self._shape_error(target_position[0] - mouse_position[0], self.deadzone_x)
        distance_y = self._shape_error(target_position[1] - mouse_position[1], self.deadzone_y)

        if distance_x * self.previous_distance_x < 0:
            self.x_integral_value = self.x_integral_value * self.rebound_strength
        if distance_y * self.previous_distance_y < 0:
            self.y_integral_value = self.y_integral_value * self.rebound_strength

        # 连续增益调度：目标越快 Kp 越大（无切换边界，不会抖振）
        kp_x = self.k_p * (1.0 + self.velocity_gain_schedule * abs(vx))
        kp_y = self.k_p * (1.0 + self.velocity_gain_schedule * abs(vy))
        x_p = kp_x * distance_x
        y_p = kp_y * distance_y

        if self.previous_time is not None:
            d_time = current_time - self.previous_time
            self.x_integral_value = self.x_integral_value + distance_x
            x_i = self.k_i * self.x_integral_value
            self.y_integral_value = self.y_integral_value + distance_y
            y_i = self.k_i * self.y_integral_value

            if d_time != 0:
                x_d = self.k_d * (distance_x - self.previous_distance_x) / d_time
                # Y 轴微分独立（默认 0）：垂直框边缘抖动大，D 项易成噪声放大器
                y_d = self.k_d_y * (distance_y - self.previous_distance_y) / d_time
            else:
                x_d = 0
                y_d = 0
        else:
            x_i = 0
            y_i = 0
            x_d = 0
            y_d = 0

        # 速度前馈：按目标速度直接补偿（相位超前），叠加在 PID 之后
        x_ff = self.velocity_feedforward_x * vx
        y_ff = self.velocity_feedforward_y * vy

        x_r = x_p + x_i + x_d + x_ff
        y_r = y_p + y_i + y_d + y_ff

        if abs(x_r) > self.max_movement:
            x_r = x_r / abs(x_r) * self.max_movement
        if abs(y_r) > self.max_movement:
            y_r = y_r / abs(y_r) * self.max_movement

        # 输出限速（甩枪速度上限，平滑由 atan 整形保证，这里只做硬顶）
        if self.max_step_per_frame > 0:
            cap = self.max_step_per_frame
            if abs(x_r) > cap:
                x_r = x_r / abs(x_r) * cap
            if abs(y_r) > cap:
                y_r = y_r / abs(y_r) * cap

        self.previous_time = current_time
        self.previous_distance_x = distance_x
        self.previous_distance_y = distance_y
        return x_r, y_r
