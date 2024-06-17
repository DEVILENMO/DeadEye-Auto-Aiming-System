import os
import sys
from enum import Enum
import threading
from pynput import keyboard, mouse
import tkinter as tk
from tkinter import ttk

from DeadEyeCore import DeadEyeCore
from DefaultModules import *

# UI文字
UI_TEXT = {
    'en': {
        'title': 'DeadEye Auto Aiming System',
        'auto_aim': 'Auto Aim',
        'auto_shoot': 'Auto Shoot',
        'language': 'Language',
        'pause': 'Pause',
        'continue': 'Continue',
        'running': 'Running',
        'paused': 'Paused',
        'exit': 'Exit',
        'fps': 'FPS: '
    },
    'zh': {
        'title': 'DeadEye辅助瞄准系统',
        'auto_aim': '辅助瞄准',
        'auto_shoot': '自动扳机',
        'language': '语言',
        'pause': '暂停',
        'continue': '继续',
        'running': '运行中',
        'paused': '已暂停',
        'exit': '退出',
        'fps': '帧率：'
    }
}


class DeadEyeUI(tk.Tk):
    def __init__(self, dead_eye_core: DeadEyeCore):
        super().__init__()

        self.dead_eye_core = dead_eye_core
        self.title(UI_TEXT['en']['title'])
        self.geometry('250x250')  # 调整窗口大小以适应新增的按钮

        # language setting
        self.lang = 'en'

        # FPS label
        self.fps_text_var = tk.StringVar(value=UI_TEXT['en']['fps'])  # 添加用于显示FPS文本的变量
        self.fps_value_var = tk.StringVar(value='0')  # 添加用于显示FPS数值的变量
        self.dead_eye_core.fps_displayer = self.fps_value_var
        self.fps_text_label = None  # 用于存储FPS文本标签的变量
        self.fps_value_label = None  # 用于存储FPS数值标签的变量

        # auto aim checkbutton
        self.auto_aim_checkbutton = None
        self.auto_aim_var = tk.BooleanVar(value=self.dead_eye_core.if_auto_aim)
        # auto shoot checkbutton
        self.auto_shoot_checkbutton = None
        self.auto_shoot_var = tk.BooleanVar(value=self.dead_eye_core.if_auto_shoot)
        # language select board
        self.language_var = tk.StringVar(value='English')
        # pause button
        self.pause_button = None
        # pause state label
        self.pause_label = None
        if not self.dead_eye_core.if_paused:
            self.pause_var = tk.StringVar(value=UI_TEXT['en']['running'])
            self.pause_button_var = tk.StringVar(value=UI_TEXT['en']['pause'])
        else:
            self.pause_var = tk.StringVar(value=UI_TEXT['en']['paused'])
            self.pause_button_var = tk.StringVar(value=UI_TEXT['en']['continue'])
        # exit button
        self.exit_button = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.hot_keys = \
            {
                keyboard.KeyCode.from_char('p'): (self.Function.PAUSE, self.KeyEventType.RELEASE),
                keyboard.KeyCode.from_char('o'): (self.Function.EXIT, self.KeyEventType.RELEASE),
                mouse.Button.left: (self.Function.TOGGLE_AUTO_AIM, None)
            }
        print('Starting keyboard listener thread...')
        self.start_keyboard_listener()
        print('Starting mouse listener thread...')
        self.start_mouse_listener()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=(10, 5))
        main_frame.grid(row=0, column=0, sticky='nsew')

        # FPS Text Label
        self.fps_text_label = ttk.Label(main_frame, textvariable=self.fps_text_var)
        self.fps_text_label.grid(row=0, column=1, padx=(5, 0), pady=5, sticky='e')
        # FPS Value Label
        self.fps_value_label = ttk.Label(main_frame, textvariable=self.fps_value_var)
        self.fps_value_label.grid(row=0, column=2, padx=(0, 5), pady=5, sticky='w')

        # Auto Aim Checkbutton
        self.auto_aim_checkbutton = ttk.Checkbutton(main_frame, text=UI_TEXT['en']['auto_aim'], variable=self.auto_aim_var,
                                    command=self.toggle_auto_aim)
        self.auto_aim_checkbutton.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        # Auto Shoot Checkbutton
        self.auto_shoot_checkbutton = ttk.Checkbutton(main_frame, text=UI_TEXT['en']['auto_shoot'], variable=self.auto_shoot_var,
                                      command=self.toggle_auto_shoot)
        self.auto_shoot_checkbutton.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        # Language Combobox
        language_frame = ttk.Labelframe(main_frame, text=UI_TEXT['en']['language'], padding=(10, 5))
        language_frame.grid(row=2, column=0, padx=5, pady=(10, 5), sticky='ew')
        language_combo = ttk.Combobox(language_frame, textvariable=self.language_var, values=['English', '简体中文'],
                                      state='readonly', width=15)
        language_combo.current(0)
        language_combo.bind('<<ComboboxSelected>>', self.change_language)
        language_combo.grid(row=0, column=0)

        # Pause Status Label
        self.pause_label = ttk.Label(main_frame, textvariable=self.pause_var)
        self.pause_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')

        # Pause/Continue Button
        self.pause_button = ttk.Button(main_frame, textvariable=self.pause_button_var, command=self.toggle_pause)
        self.pause_button.grid(row=4, column=0, padx=5, pady=5, sticky='w')

        # Exit Button
        self.exit_button = ttk.Button(main_frame, text=UI_TEXT['en']['exit'], command=self.exit_program)
        self.exit_button.grid(row=5, column=0, padx=5, pady=5, sticky='w')

    # Button functions
    def change_language(self, event):
        self.lang = 'zh' if self.language_var.get() == '简体中文' else 'en'

        self.title(UI_TEXT[self.lang]['title'])
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for c in child.winfo_children():
                    if isinstance(c, ttk.Checkbutton):
                        if c == self.auto_aim_checkbutton:
                            c.configure(text=UI_TEXT[self.lang]['auto_aim'])
                        elif c == self.auto_shoot_checkbutton:
                            c.configure(text=UI_TEXT[self.lang]['auto_shoot'])
                    elif isinstance(c, ttk.Labelframe):
                        c.configure(text=UI_TEXT[self.lang]['language'])
                    elif isinstance(c, ttk.Label):
                        if c == self.fps_text_label:
                            self.fps_text_var.set(UI_TEXT[self.lang]['fps'])
                        else:
                            if not self.dead_eye_core.if_paused:
                                self.pause_var.set(UI_TEXT[self.lang]['running'])
                            else:
                                self.pause_var.set(UI_TEXT[self.lang]['paused'])
                    elif isinstance(c, ttk.Button):
                        if c == self.pause_button:
                            if not self.dead_eye_core.if_paused:
                                self.pause_button_var.set(UI_TEXT[self.lang]['pause'])
                            else:
                                self.pause_button_var.set(UI_TEXT[self.lang]['continue'])
                        elif c == self.exit_button:
                            c.configure(text=UI_TEXT[self.lang]['exit'])

    def toggle_auto_aim(self):
        self.dead_eye_core.switch_auto_aim_state()
        self.auto_aim_var.set(self.dead_eye_core.if_auto_aim)

    def toggle_auto_shoot(self):
        self.dead_eye_core.switch_auto_shoot_state()

    def toggle_pause(self):
        if not pause():
            self.pause_var.set(UI_TEXT[self.lang]['running'])
            self.pause_button_var.set(UI_TEXT[self.lang]['pause'])
        else:
            self.pause_var.set(UI_TEXT[self.lang]['paused'])
            self.pause_button_var.set(UI_TEXT[self.lang]['continue'])

    def exit_program(self):
        self.on_closing()

    def on_closing(self):
        stop_program()

    # Keyboard and mouse event
    class Function(Enum):
        PAUSE = 0
        EXIT = 1
        TOGGLE_AUTO_AIM = 2

    class KeyEventType(Enum):
        PRESS = 0
        RELEASE = 1

    def start_keyboard_listener(self):
        keyboard_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        keyboard_listener.daemon = True
        keyboard_listener.start()
        print('Keyboard listener thread inited.')

    def start_mouse_listener(self):
        mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
        mouse_listener.daemon = True
        mouse_listener.start()
        print('Mouse listener thread inited.')

    def on_key_press(self, key):
        try:
            self.handle_hotkeys(key, self.KeyEventType.PRESS)
            return
        except AttributeError:
            return

    def on_key_release(self, key):
        try:
            self.handle_hotkeys(key, self.KeyEventType.RELEASE)
            return
        except AttributeError:
            return

    def on_mouse_click(self, x, y, button, pressed):
        if pressed:
            self.handle_hotkeys(button, self.KeyEventType.PRESS)
        else:
            self.handle_hotkeys(button, self.KeyEventType.RELEASE)

    def handle_hotkeys(self, key, key_event_type: KeyEventType):
        if key in self.hot_keys.keys():
            function, event_type = self.hot_keys[key]
            if function == self.Function.PAUSE and key_event_type == event_type:
                self.toggle_pause()
            if function == self.Function.EXIT and key_event_type == event_type:
                self.after(0, self.exit_program)
            if function == self.Function.TOGGLE_AUTO_AIM:
                self.toggle_auto_aim()


# Pause/Continue
def pause():
    global dead_eye, program_continued
    if dead_eye.switch_pause_state():
        print('Paused.')
        return True
    else:
        program_continued.release()
        print('Continue working.')
        return False


# 结束程序
def stop_program():
    print('CLOSING...')
    global dead_eye, detect_module, auto_aiming_module, botClosed
    dead_eye.on_exit()
    detect_module.on_exit()
    auto_aiming_module.on_exit()
    botClosed.release()
    sys.exit(0)


if __name__ == '__main__':
    view_range = (640, 640)
    print('Initing detect module...')
    detect_module = YoloDetector('./weights/apex_v8s.engine')  # set your model file here, .pt .trt or .engine
    print('Initing auto aiming module...')
    auto_aiming_module = DeadEyeAutoAimingModule(view_range)
    print('Initing camera module...')
    camera_module = SimpleScreenShotCamera(view_range)
    dead_eye = DeadEyeCore(camera_module, detect_module, auto_aiming_module)

    # 多线程变量
    program_continued = threading.Semaphore(0)
    botClosed = threading.Semaphore(0)

    print('Starting UI...')
    ui = DeadEyeUI(dead_eye)
    ui.mainloop()

    botClosed.acquire()  # main thread holds on here
