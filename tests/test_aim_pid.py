# coding: utf-8
"""DeadEyeAutoAimingModule PID 控制律测试（死区 / Y轴微分独立 / 增益调度 / 速度前馈）。"""
import pytest
import numpy as np

from deadeye.execution_module.auto_aim import DeadEyeAutoAimingModule

VIEW = (640, 640)


def make_module(**overrides):
    params = dict(
        k_p=0.5, k_i=0.0, k_d=0.0, k_d_y=0.0,
        deadzone_x=0.0, deadzone_y=0.0,
        shaping='linear', atan_slope=1.0, atan_ceiling=300.0,
        velocity_gain_schedule=0.0,
        velocity_feedforward_x=0.0, velocity_feedforward_y=0.0,
        rebound_strength=0.0, max_movement=100000, max_step_per_frame=0.0,
        output_smoothing=0.0,
    )
    params.update(overrides)
    return DeadEyeAutoAimingModule(VIEW, enable_auto_aim=False, enable_auto_shoot=False, **params)


# ---------------------------------------------------------------- 死区

def test_deadband_zeroes_small_error():
    mod = make_module(deadzone_x=5.0, deadzone_y=5.0)
    x_r, y_r = mod.calculate_mouse_movement_by_pid((102, 103), (100, 100))
    assert x_r == 0 and y_r == 0


def test_deadband_keeps_error_outside_band():
    """削减式死区：死区外输出与超出部分成正比（边界连续，无踢）。"""
    mod = make_module(deadzone_x=5.0)
    x_r, y_r = mod.calculate_mouse_movement_by_pid((110, 100), (100, 100))
    assert x_r == pytest.approx(0.5 * 5)  # 10px 误差、死区 5 -> 有效 5


def test_deadband_boundary_is_continuous():
    """边界两侧无跳变：越界 1px 的输出远小于死区宽度本身（对比硬边界会全额踢出）。"""
    mod = make_module(deadzone_x=20.0)
    x_in, _ = mod.calculate_mouse_movement_by_pid((120, 100), (100, 100))   # 误差 = 死区
    mod.clear_pid_history()
    x_out1, _ = mod.calculate_mouse_movement_by_pid((121, 100), (100, 100))  # 越界 1px
    mod.clear_pid_history()
    x_out2, _ = mod.calculate_mouse_movement_by_pid((122, 100), (100, 100))  # 越界 2px
    assert x_in == 0
    assert x_out1 == pytest.approx(0.5 * 1)
    assert x_out2 == pytest.approx(0.5 * 2)


def test_deadband_also_gates_integral_and_derivative():
    """死区内的连续小误差不应累积积分，也不应触发微分。"""
    mod = make_module(k_i=1.0, k_d=1.0, deadzone_x=5.0)
    mod.calculate_mouse_movement_by_pid((3, 0), (0, 0))     # 误差 3 -> 死区内
    mod.previous_time = None
    mod.calculate_mouse_movement_by_pid((4, 0), (0, 0))     # 仍死区内
    mod.previous_time = None
    x_r, _ = mod.calculate_mouse_movement_by_pid((6, 0), (0, 0))  # 出死区 1px
    assert x_r == pytest.approx(0.5 * 1)  # 只有 P：没有历史积分/微分量


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


# ---------------------------------------------------------------- atan 软饱和整形

def test_atan_shaping_zero_below_deadzone():
    mod = make_module(shaping='atan', deadzone_x=0.5)
    x_r, _ = mod.calculate_mouse_movement_by_pid((100.4, 100), (100, 100))
    assert x_r == 0


def test_atan_shaping_small_error_near_linear():
    """小误差区间 atan(x)≈x：输出接近线性模式（斜率=1 时几乎一致）。"""
    linear = make_module(shaping='linear', deadzone_x=0.0)
    atan_mod = make_module(shaping='atan', deadzone_x=0.0, atan_slope=1.0, atan_ceiling=300.0)
    x_lin, _ = linear.calculate_mouse_movement_by_pid((102.0, 100), (100, 100))
    x_atan, _ = atan_mod.calculate_mouse_movement_by_pid((102.0, 100), (100, 100))
    assert x_atan == pytest.approx(x_lin, rel=0.01)


def test_atan_shaping_saturates_at_ceiling():
    """大误差软饱和：输出封顶于 Kp·atan_ceiling，且平滑逼近（无 clamp 折角）。"""
    mod = make_module(shaping='atan', deadzone_x=0.0, atan_ceiling=300.0)
    x_huge, _ = mod.calculate_mouse_movement_by_pid((1e6, 100), (100, 100))
    assert x_huge == pytest.approx(0.5 * 300.0, rel=0.001)
    x_big, _ = mod.calculate_mouse_movement_by_pid((2000.0, 100), (100, 100))
    assert x_big < x_huge  # 单调
    assert x_big > 0.5 * 300.0 * 0.9  # 大误差已接近上限


def test_atan_shaping_monotonic():
    mod = make_module(shaping='atan', deadzone_x=0.5)
    outputs = []
    for e in (1.0, 3.0, 10.0, 50.0, 200.0, 1000.0):
        x_r, _ = mod.calculate_mouse_movement_by_pid((100 + e, 100), (100, 100))
        outputs.append(x_r)
    assert outputs == sorted(outputs) and all(v > 0 for v in outputs)


# ---------------------------------------------------------------- 输出限速

def test_step_limit_caps_flick_speed():
    """限速生效：大误差输出被截到 max_step_per_frame，小误差不受影响。"""
    mod = make_module(max_step_per_frame=15.0)
    x_big, _ = mod.calculate_mouse_movement_by_pid((10000, 100), (100, 100))
    assert x_big == pytest.approx(15.0)
    x_small, _ = mod.calculate_mouse_movement_by_pid((112, 100), (100, 100))
    assert x_small == pytest.approx(0.5 * 12)


def test_step_limit_zero_disables_cap():
    mod = make_module(max_step_per_frame=0.0)
    x_r, _ = mod.calculate_mouse_movement_by_pid((10000, 100), (100, 100))
    assert x_r == pytest.approx(0.5 * 9900)


# ---------------------------------------------------------------- 默认参数

def test_default_gains_match_tuned_preset():
    """默认增益：Kp 维持、Ki 按时间归一化（每秒）、Kd 默认关、积分输出限幅、限速 15px/帧。"""
    defaults = {f.key: f.default for f in DeadEyeAutoAimingModule.get_config_spec()}
    assert defaults['k_p'] == pytest.approx(0.3)
    assert defaults['k_i'] == pytest.approx(5.0)
    assert defaults['k_d'] == pytest.approx(0.0)
    assert defaults['k_d_y'] == pytest.approx(0.0)
    assert defaults['deadzone_x'] == pytest.approx(0.5)
    assert defaults['deadzone_y'] == pytest.approx(0.5)
    assert defaults['deadzone_box_ratio'] == pytest.approx(0.0)
    assert defaults['max_integral_output'] == pytest.approx(30.0)
    assert defaults['max_step_per_frame'] == pytest.approx(15.0)


def test_integral_normalized_by_time_not_frame_count():
    """积分按 Σe·dt 归一化：同样误差持续 dt 秒，I = Ki·e·dt，与帧数无关。"""
    import time as _time
    mod = make_module(k_i=2.0)
    mod.calculate_mouse_movement_by_pid((110, 100), (100, 100))       # 首帧建立历史
    mod.previous_time = _time.time() - 0.2                             # dt = 0.2s（低于 0.5s 过期保护）
    x_r, _ = mod.calculate_mouse_movement_by_pid((110, 100), (100, 100))
    # I = 2.0 * (10*0.2) = 4，P = 0.5*10 = 5，D=0 -> 总 9
    assert x_r == pytest.approx(0.5 * 10 + 2.0 * 10 * 0.2)


def test_integral_output_clamped():
    """积分输出限幅：长时间同号误差下 I 项封顶，不再无界增长。"""
    import time as _time
    mod = make_module(k_i=100.0, max_integral_output=30.0)
    mod.calculate_mouse_movement_by_pid((500, 100), (100, 100))
    for _ in range(5):  # 累计 5×0.2s 的 400px 误差 -> 未限幅将达 100*400 = 40000
        mod.previous_time = _time.time() - 0.2
        mod.calculate_mouse_movement_by_pid((500, 100), (100, 100))
    assert mod.last_pid_debug['i'][0] == pytest.approx(30.0)


# ---------------------------------------------------------------- 死区随框比例

def test_proportional_deadband_scales_with_box(monkeypatch):
    """大框的死区按比例放大：200px 框、比例 0.03 -> 有效死区 6px。"""
    import deadeye.execution_module.auto_aim as aa
    monkeypatch.setattr(aa.pyautogui, 'position', lambda: (0, 0))

    from deadeye.base.modules import Target
    mod = DeadEyeAutoAimingModule(
        (640, 640), enable_auto_aim=True, enable_auto_shoot=False,
        k_p=0.5, output_smoothing=0.0, max_movement=100000, max_step_per_frame=0,
        deadzone_x=1.5, deadzone_y=1.5, deadzone_box_ratio=0.03,
        auto_aim_range_x=10.0, auto_aim_range_y=10.0,
    )
    mod.view_range_start = (0, 0)  # mock 鼠标坐标即视口坐标
    # 框 (100,100)-(300,300)，中心 (200,200)；鼠标在 (0,0) 视口坐标
    target = Target('ball', 0, (100, 100), (300, 300))
    mod.tracking_target_id = 0
    x_r, _ = mod.auto_aim([target])
    # 误差 200px，超出死区 6px，输出 = P(0.5*200) + I
    assert abs(x_r) > 50
    assert mod.last_pid_debug['e_shaped'][0] == pytest.approx(194.0)


def test_proportional_deadband_silences_small_offset_on_big_box(monkeypatch):
    """大球上偏移 5px（旧死区会响应）应在比例死区内静默。"""
    import deadeye.execution_module.auto_aim as aa
    monkeypatch.setattr(aa.pyautogui, 'position', lambda: (0, 0))

    from deadeye.base.modules import Target
    mod = DeadEyeAutoAimingModule(
        (640, 640), enable_auto_aim=True, enable_auto_shoot=False,
        k_p=0.5, output_smoothing=0.0, max_movement=100000,
        deadzone_x=1.5, deadzone_y=1.5, deadzone_box_ratio=0.03,
        auto_aim_range_x=10.0, auto_aim_range_y=10.0,
    )
    mod.view_range_start = (0, 0)  # mock 鼠标坐标即视口坐标
    # 200px 框：有效死区 6px。把鼠标放在 (5,0)：误差 5px < 6 -> 零输出
    monkeypatch.setattr(aa.pyautogui, 'position', lambda: (195, 0))
    target = Target('ball', 0, (100, 100), (300, 300))
    mod.tracking_target_id = 0
    x_r, _ = mod.auto_aim([target])
    assert x_r == 0


def test_proportional_deadband_keeps_min_on_small_box(monkeypatch):
    """小球回到死区下限：40px 框比例项 1.2 < 下限 1.5。"""
    import deadeye.execution_module.auto_aim as aa
    from deadeye.base.modules import Target
    mod = DeadEyeAutoAimingModule(
        (640, 640), enable_auto_aim=True, enable_auto_shoot=False,
        k_p=0.5, output_smoothing=0.0, max_movement=100000,
        deadzone_x=1.5, deadzone_y=1.5, deadzone_box_ratio=0.03,
        auto_aim_range_x=10.0, auto_aim_range_y=10.0,
    )
    mod.view_range_start = (0, 0)  # mock 鼠标坐标即视口坐标
    # 40px 框中心 (200,200)，鼠标 (0,0)：误差 200px >> 死区，看输出非零即可
    monkeypatch.setattr(aa.pyautogui, 'position', lambda: (0, 0))
    target = Target('ball', 0, (180, 180), (220, 220))
    mod.tracking_target_id = 0
    x_r, _ = mod.auto_aim([target])
    assert x_r != 0
    # 小框上 2px 误差仍应静默（比例项未超过下限）
    monkeypatch.setattr(aa.pyautogui, 'position', lambda: (199, 0))
    mod.clear_pid_history()
    mod.tracking_target_id = 0
    target2 = Target('ball', 0, (180, 180), (220, 220))
    x_r2, _ = mod.auto_aim([target2])
    assert x_r2 == 0

# ---------------------------------------------------------------- 输出平滑

def test_output_smoothing_damps_alternating_jitter():
    """交替 ±e 的噪声经 EMA 后幅度大幅衰减：稳态幅度 = e·(1-s)/s。"""
    mod = make_module(output_smoothing=0.75)
    outs = []
    for sign in (1, -1) * 12:
        x_r, _ = mod.calculate_mouse_movement_by_pid((100 + sign * 10, 100), (100, 100))
        outs.append(x_r)
    # 稳态振幅 = 10*(1-0.75)/0.75 ≈ 3.3（未平滑为 5），且相邻帧差远小于未平滑
    assert abs(outs[-1]) < 5.0
    diffs = np.abs(np.diff(outs))
    assert diffs.max() < 5.0


def test_output_smoothing_preserves_constant_tracking():
    """恒定误差下 EMA 输出与无平滑完全一致：平滑只衰减变化，不拖慢稳定跟踪。"""
    smooth = make_module(output_smoothing=0.6)
    raw = make_module(output_smoothing=0.0)
    for _ in range(10):
        s = smooth.calculate_mouse_movement_by_pid((160, 100), (100, 100))[0]
        r = raw.calculate_mouse_movement_by_pid((160, 100), (100, 100))[0]
        assert s == pytest.approx(r)  # 恒 30：EMA 对常值透明


def test_output_smoothing_damps_step_transition():
    """误差突降时过渡被阻尼：平滑版逐帧渐变，无平滑立即跳变。"""
    smooth = make_module(output_smoothing=0.6)
    raw = make_module(output_smoothing=0.0)
    s_vals, r_vals = [], []
    for e in (60, 60, 10, 10, 10):
        s_vals.append(smooth.calculate_mouse_movement_by_pid((100 + e, 100), (100, 100))[0])
        r_vals.append(raw.calculate_mouse_movement_by_pid((100 + e, 100), (100, 100))[0])
    assert r_vals[2] == pytest.approx(5.0)          # 无平滑立即跳到 0.5*10
    assert s_vals[2] == pytest.approx(0.4 * 5 + 0.6 * 30)  # 平滑版渐变 = 20
    assert 5.0 < s_vals[2] < 30.0


def test_output_smoothing_resets_on_target_change():
    mod = make_module(output_smoothing=0.9)
    mod.calculate_mouse_movement_by_pid((200, 100), (100, 100))
    mod.clear_pid_history()
    assert mod._smoothed_x is None  # 换目标后平滑状态清零，不携带旧速度
    x_r, _ = mod.calculate_mouse_movement_by_pid((110, 100), (100, 100))
    assert x_r == pytest.approx(0.5 * 10)
