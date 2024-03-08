import os
import sys
from pathlib import Path
import threading
from pynput import keyboard

yolov7_path = Path(__file__).resolve().parent / 'Yolov7'
sys.path.append(str(yolov7_path))
from DeadEyeCore import DeadEyeCore
from DefaultModules import *


# 键盘监听
def start_keyboard_listener():
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    # keyboard_listener.join()
    print('Keyboard listener thread inited.')


def on_press(key):
    try:
        # print(key.char)
        # t0 = time.time()
        global dead_eye
        return
    except AttributeError:
        return


def on_release(key):
    try:
        # print(key.char)
        # t0 = time.time()
        global dead_eye
        if key.char == 'p':
            pause()
        if key.char == 'o':
            # 停止程序
            stop_program()
        return
    except AttributeError:
        return


# 暂停/继续功能
def pause():
    global program_continued
    if dead_eye.switch_pause_state():
        print('Paused.')
    else:
        program_continued.release()
        print('Continue working.')


# 结束程序
def stop_program():
    print('CLOSING...')
    global botClosed, dead_eye
    botClosed.release()
    os._exit(0)


if __name__ == '__main__':
    view_range = (640, 640)
    print('Initing detect module...')
    detect_module = YoloDetector('./weights/yolov7-tiny.pt')  # set your model file here, .pt or .onnx
    print('Initing auto aiming module...')
    auto_aiming_module = DeadEyeAutoAimingModule(view_range)
    dead_eye = DeadEyeCore(detect_module, auto_aiming_module, view_range)

    # 多线程变量
    program_continued = threading.Semaphore(0)
    botClosed = threading.Semaphore(0)

    print('Starting keyboard listener thread...')
    start_keyboard_listener()

    botClosed.acquire()  # main thread holds on here
