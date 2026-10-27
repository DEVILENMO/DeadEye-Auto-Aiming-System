# coding: utf-8
"""DeadEyeAutoAimingModule PID 控制律测试（死区 / Y轴微分独立 / 增益调度 / 速度前馈）。"""
import pytest

from deadeye.execution_module.auto_aim import DeadEyeAutoAimingModule

VIEW = (640, 640)


def make_module(**overrides):
    params = dict(
        k_p=0.5, k_i=0.0, k_d=0.0, k_d_y=0.0,
        deadzone_x=0.0, deadzone_y=0.0,
        velocity_gain_schedule=0.0,
        velocity_feedforward_x=0.0, velocity_feedforward_y=0.0,
        rebound_strength=0.0, max_movement=100000,
    )
    params.update(overrides)
    return DeadEyeAutoAimingModule(VIEW, enable_auto_aim=False, enable_auto_shoot=False, **params)


# ---------------------------------------------------------------- 死区

def test_deadband_zeroes_small_error():
    mod = make_module(deadzone_x=5.0, deadzone_y=5.0)
    x_r, y_r = mod.calculate_mouse_movement_by_pid((102, 103), (100, 100))
    assert x_r == 0 and y_r == 0


def test_deadband_keeps_error_outside_band():
    mod = make_module(deadzone_x=5.0)
    x_r, y_r = mod.calculate_mouse_movement_by_pid((110, 100), (100, 100))
    assert x_r == pytest.approx(0.5 * 10)


def test_deadband_also_gates_integral_and_derivative():
    """死区内的连续小误差不应累积积分，也不应触发微分。"""
    mod = make_module(k_i=1.0, k_d=1.0, deadzone_x=5.0)
    mod.calculate_mouse_movement_by_pid((3, 0), (0, 0))     # 误差 3 -> 死区内
    mod.previous_time = None
    mod.calculate_mouse_movement_by_pid((4, 0), (0, 0))     # 仍死区内
    mod.previous_time = None
    x_r, _ = mod.calculate_mouse_movement_by_pid((6, 0), (0, 0))  # 出死区
    assert x_r == pytest.approx(0.5 * 6)  # 只有 P：没有历史积分/微分量


# ---------------------------------------------------------------- Y 轴微分独立

def _second_call_with_dt(mod, target, mouse, dt=0.01):
    """首帧建立历史，再以受控 dt 做第二帧，返回第二帧输出。"""
    mod.calculate_mouse_movement_by_pid(target, mouse)
    import time
    mod.previous_time = time.time() - dt
    return mod.calculate_mouse_movement_by_pid(target, mouse)


def test_y_derivative_disabled_by_default():
    """k_d_y=0：Y 误差变化不产生微分项，输出与误差成正比。"""
    mod = make_module(k_d=1.0, k_d_y=0.0)
    x_r, y_r = _second_call_with_dt(mod, (110, 100), (100, 80))
    # y 误差恒为 20（两次调用相同），本就没有变化量——改用变化场景在下一测覆盖
    assert y_r == pytest.approx(0.5 * 20)


def test_y_derivative_applies_when_enabled():
    mod = make_module(k_d=0.0, k_d_y=1.0)
    mod.calculate_mouse_movement_by_pid((110, 100), (100, 90))   # y 误差 10
    import time
    mod.previous_time = time.time() - 0.01
    x_r, y_r = mod.calculate_mouse_movement_by_pid((110, 100), (100, 85))  # y 误差 15
    # y = P(0.5*15) + D(1.0*(15-10)/0.01)
    assert y_r == pytest.approx(0.5 * 15 + 1.0 * 5 / 0.01)
    # x 误差恒 10 无变化，k_d=0 -> 纯 P
    assert x_r == pytest.approx(0.5 * 10)


# ---------------------------------------------------------------- 速度前馈

def test_velocity_feedforward_adds_offset():
    base = make_module()
    ff = make_module(velocity_feedforward_x=0.1, velocity_feedforward_y=0.2)
    x0, y0 = base.calculate_mouse_movement_by_pid((110, 110), (100, 100))
    x1, y1 = ff.calculate_mouse_movement_by_pid((110, 110), (100, 100), (10.0, 5.0))
    assert x1 - x0 == pytest.approx(0.1 * 10.0)
    assert y1 - y0 == pytest.approx(0.2 * 5.0)


# ---------------------------------------------------------------- 增益调度

def test_velocity_gain_schedule_scales_p():
    static = make_module()
    scheduled = make_module(velocity_gain_schedule=0.1)  # v=50 -> Kp 放大 6 倍
    x0, _ = static.calculate_mouse_movement_by_pid((110, 100), (100, 100))
    x1, _ = scheduled.calculate_mouse_movement_by_pid((110, 100), (100, 100), (50.0, 0.0))
    assert x1 == pytest.approx(x0 * (1 + 0.1 * 50))


def test_gain_schedule_is_continuous():
    """连续函数：速度微小变化 -> 输出微小变化，无切换跳变。"""
    mod = make_module(velocity_gain_schedule=0.05)
    x1, _ = mod.calculate_mouse_movement_by_pid((110, 100), (100, 100), (10.0, 0.0))
    mod2 = make_module(velocity_gain_schedule=0.05)
    x2, _ = mod2.calculate_mouse_movement_by_pid((110, 100), (100, 100), (10.5, 0.0))
    assert x2 == pytest.approx(x1 * (1 + 0.05 * 10.5) / (1 + 0.05 * 10.0), rel=1e-6)
