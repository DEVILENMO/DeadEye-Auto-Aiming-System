# coding: utf-8
"""截屏输入后端测试（需要真实 Windows 桌面会话）。

覆盖 ScreenShotHelper 的三个后端（windows-capture / bettercam / mss）
与 AUTO 降级链：各后端应能在超时内产出一帧中心视口 RGB 图像并干净关闭。
"""
import sys
import time

import numpy as np
import pytest

from deadeye.input_module.screenshot_helper import ScreenShotHelper

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='截屏后端仅在 Windows 桌面会话可用')

VIEW = (640, 640)
FIRST_FRAME_TIMEOUT = 5.0


def _grab_first_frame(camera_type):
    """返回 (helper, 首帧或 None)；不抛初始化异常，交由调用方决定 skip 还是失败。"""
    try:
        helper = ScreenShotHelper(VIEW[0], VIEW[1], camera_type)
    except Exception as e:
        return None, e
    deadline = time.time() + FIRST_FRAME_TIMEOUT
    img = None
    while time.time() < deadline and img is None:
        img = helper.capture_screen_shot()
        if img is None:
            time.sleep(0.01)
    return helper, img


@pytest.mark.parametrize('camera_type_name', ['WINDOWS_CAPTURE', 'BETTERCAM', 'MSS'])
def test_backend_produces_rgb_frame(camera_type_name):
    helper, result = _grab_first_frame(ScreenShotHelper.CameraType[camera_type_name])
    if helper is None:
        pytest.skip(f'后端 {camera_type_name} 在当前环境不可用（生产环境会自动降级）: {result}')
    try:
        assert result is not None, f'{camera_type_name} 在 {FIRST_FRAME_TIMEOUT}s 内没有产出帧'
        assert result.shape == (VIEW[1], VIEW[0], 3), f'帧形状异常: {result.shape}'
        assert result.dtype == np.uint8
    finally:
        helper.close()


def test_auto_backend_resolves():
    helper, img = _grab_first_frame(ScreenShotHelper.CameraType.AUTO)
    try:
        assert helper is not None and helper.backend is not None
        assert img is not None, 'AUTO 后端在超时内没有产出帧'
    finally:
        if helper is not None:
            helper.close()


def test_view_larger_than_screen_is_clamped():
    try:
        helper = ScreenShotHelper(100000, 100000, ScreenShotHelper.CameraType.AUTO)
    except Exception as e:
        pytest.skip(f'环境不可用: {e}')
    try:
        img = None
        deadline = time.time() + FIRST_FRAME_TIMEOUT
        while time.time() < deadline and img is None:
            img = helper.capture_screen_shot()
            if img is None:
                time.sleep(0.01)
        assert img is not None
        screen_w, screen_h = helper.screen_resolution_x, helper.screen_resolution_y
        assert img.shape == (min(100000, screen_h), min(100000, screen_w), 3)
    finally:
        helper.close()


# ---------------------------------------------------------------- 相机层

def _first_camera_frame(camera):
    img = None
    deadline = time.time() + FIRST_FRAME_TIMEOUT
    while time.time() < deadline and img is None:
        img = camera.get_image()
        if img is None:
            time.sleep(0.01)
    return img


def test_camera_default_view_is_320():
    from deadeye.input_module import SimpleScreenShotCamera

    camera = SimpleScreenShotCamera()
    try:
        assert camera.view_range == (320, 320)
        img = _first_camera_frame(camera)
        assert img is not None
        assert img.shape == (320, 320, 3)
    finally:
        camera.on_exit()


def test_camera_custom_view_size():
    from deadeye.input_module import SimpleScreenShotCamera

    camera = SimpleScreenShotCamera(view_width=240, view_height=180)
    try:
        assert camera.view_range == (240, 180)
        img = _first_camera_frame(camera)
        assert img is not None
        assert img.shape == (180, 240, 3)
    finally:
        camera.on_exit()
