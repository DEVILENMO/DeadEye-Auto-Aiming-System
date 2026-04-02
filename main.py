import os
import sys
import threading
from enum import Enum
import importlib
import inspect
import pkgutil

from pynput import keyboard, mouse
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog

from deadeye.core import DeadEyeCore
from deadeye.detection_module import YoloDetector
from deadeye.execution_module import DeadEyeAutoAimingModule, VideoWriterOutput
from deadeye.input_module import SimpleScreenShotCamera, VideoFileCamera
from deadeye.base.modules import BaseCamera, DetectModule, ExecutionModule
from deadeye.base.configurable import ConfigurableMixin

# UI文字
UI_TEXT = {
    'en': {
        'title': 'DeadEye Auto Aiming System',
        'basic_settings': 'Basic settings',
        'startup_config': 'Pipeline configuration',
        'detect_module': 'Detector',
        'input_module': 'Input',
        'exec_module': 'Execution',
        'module_params_fmt': '{name} parameters',
        'btn_choose': 'Browse…',
        'dialog_choose_file': 'Choose file',
        'dialog_choose_path': 'Choose path',
        'auto_aim': 'Auto Aim',
        'auto_shoot': 'Auto Shoot',
        'language': 'Language',
        'pause': 'Pause',
        'continue': 'Continue',
        'running': 'Running',
        'paused': 'Paused',
        'exit': 'Exit',
        'fps': 'FPS: ',
        'start_run': 'Start',
        'idle': 'Not started',
        'status_already_running': 'Already running',
        'status_model_invalid': 'Invalid model path: choose a model in detector settings',
        'status_init_detect': 'Initializing detector: {name}…',
        'status_init_input': 'Initializing input: {name}…',
        'status_init_exec': 'Initializing execution: {name}…',
        'status_starting_core': 'Starting core…',
        'status_start_failed': 'Startup failed: {error}',
    },
    'zh': {
        'title': 'DeadEye辅助瞄准系统',
        'basic_settings': '基本设置',
        'startup_config': '启动配置',
        'detect_module': '检测模块',
        'input_module': '输入模块',
        'exec_module': '执行模块',
        'module_params_fmt': '{name} 参数',
        'btn_choose': '选择',
        'dialog_choose_file': '选择文件',
        'dialog_choose_path': '选择路径',
        'auto_aim': '辅助瞄准',
        'auto_shoot': '自动扳机',
        'language': '语言',
        'pause': '暂停',
        'continue': '继续',
        'running': '运行中',
        'paused': '已暂停',
        'exit': '退出',
        'fps': '帧率：',
        'start_run': '开始运行',
        'idle': '未启动',
        'status_already_running': '已在运行中',
        'status_model_invalid': '模型路径无效：请在检测模块参数中选择模型文件',
        'status_init_detect': '初始化检测模块：{name}…',
        'status_init_input': '初始化输入模块：{name}…',
        'status_init_exec': '初始化执行模块：{name}…',
        'status_starting_core': '启动核心线程…',
        'status_start_failed': '启动失败：{error}',
    },
}


def _discover_subclasses_in_package(package_name: str, base_class: type) -> dict[str, type]:
    """
    扫描 package_name 及其子模块，收集继承 base_class 的类。
    会跳过导入失败的模块（例如缺少可选依赖）。
    """
    result: dict[str, type] = {}
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return result

    def collect_from_module(mod):
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if obj is base_class:
                    continue
                if issubclass(obj, base_class):
                    result[obj.__name__] = obj
            except Exception:
                continue

    collect_from_module(package)
    if not hasattr(package, '__path__'):
        return result

    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        try:
            mod = importlib.import_module(module_info.name)
        except Exception:
            continue
        collect_from_module(mod)

    return dict(sorted(result.items(), key=lambda kv: kv[0].lower()))


class DeadEyeUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')

        self.dead_eye_core: DeadEyeCore | None = None
        self.detect_module = None
        self.camera_module = None
        self.bot_closed = threading.Semaphore(0)

        self.title(UI_TEXT['en']['title'])
        self.geometry('520x1040')
        self.minsize(480, 1000)

        self.lang = 'en'

        self.status_var = tk.StringVar(value=UI_TEXT['en']['idle'])
        self.fps_text_var = tk.StringVar(value=UI_TEXT['en']['fps'])
        self.fps_value_var = tk.StringVar(value='0')
        self.fps_text_label = None
        self.fps_value_label = None

        self.input_module_var = tk.StringVar(value='')
        self.detect_module_var = tk.StringVar(value='YoloDetector')
        self.exec_module_var = tk.StringVar(value='DeadEyeAutoAimingModule')

        self.detect_config_frame = None
        self.detect_config_vars: dict[str, tk.Variable] = {}
        self.detect_config_widgets: list[ctk.CTkBaseClass] = []
        self.exec_config_frame = None
        self.exec_config_vars: dict[str, tk.Variable] = {}
        self.exec_config_widgets: list[ctk.CTkBaseClass] = []
        self.input_config_frame = None
        self.input_config_vars: dict[str, tk.Variable] = {}
        self.input_config_widgets: list[ctk.CTkBaseClass] = []

        self.auto_aim_checkbutton = None
        self.auto_aim_var = tk.BooleanVar(value=False)
        self.auto_shoot_checkbutton = None
        self.auto_shoot_var = tk.BooleanVar(value=False)
        self.aim_runtime_frame = None
        self.language_var = tk.StringVar(value='English')
        self.language_menu = None
        self.language_title_label = None
        self.run_button = None
        self.run_button_var = tk.StringVar(value=UI_TEXT['en']['start_run'])
        self.exit_button = None

        self.input_menu = None
        self.detect_menu = None
        self.exec_menu = None
        self._title_basic = None
        self._label_startup_config = None
        self._label_detect_module = None
        self._label_input_module = None
        self._label_exec_module = None

        self.camera_class_map = _discover_subclasses_in_package('deadeye.input_module', BaseCamera)
        self.detect_class_map = _discover_subclasses_in_package('deadeye.detection_module', DetectModule)
        self.execution_class_map = _discover_subclasses_in_package('deadeye.execution_module', ExecutionModule)
        if not self.camera_class_map:
            self.camera_class_map = {'SimpleScreenShotCamera': SimpleScreenShotCamera}
        if not self.detect_class_map:
            self.detect_class_map = {'YoloDetector': YoloDetector}
        if not self.execution_class_map:
            self.execution_class_map = {
                'DeadEyeAutoAimingModule': DeadEyeAutoAimingModule,
                'VideoWriterOutput': VideoWriterOutput,
            }
        if 'SimpleScreenShotCamera' in self.camera_class_map:
            self.input_module_var.set('SimpleScreenShotCamera')
        if 'YoloDetector' in self.detect_class_map:
            self.detect_module_var.set('YoloDetector')
        if 'DeadEyeAutoAimingModule' in self.execution_class_map:
            self.exec_module_var.set('DeadEyeAutoAimingModule')

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

    def _t(self, key: str, **kwargs) -> str:
        template = UI_TEXT[self.lang][key]
        return template.format(**kwargs) if kwargs else template

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, corner_radius=10)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)
        main_frame.grid_columnconfigure(0, weight=1)

        basic_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        basic_frame.grid(row=0, column=0, padx=10, pady=(10, 8), sticky='ew')
        basic_frame.grid_columnconfigure(1, weight=1)

        self._title_basic = ctk.CTkLabel(basic_frame, text=self._t('basic_settings'))
        self._title_basic.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 6), sticky='w')
        self.language_title_label = ctk.CTkLabel(basic_frame, text=self._t('language'))
        self.language_title_label.grid(row=1, column=0, padx=10, pady=(4, 10), sticky='w')
        self.language_menu = ctk.CTkOptionMenu(
            basic_frame,
            values=['English', '简体中文'],
            variable=self.language_var,
            command=self.change_language,
            width=160,
        )
        self.language_menu.grid(row=1, column=1, padx=10, pady=(4, 10), sticky='w')

        config_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        config_frame.grid(row=1, column=0, padx=10, pady=(0, 8), sticky='ew')
        config_frame.grid_columnconfigure(1, weight=1)

        self._label_startup_config = ctk.CTkLabel(config_frame, text=self._t('startup_config'))
        self._label_startup_config.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 6), sticky='w')

        self._label_detect_module = ctk.CTkLabel(config_frame, text=self._t('detect_module'))
        self._label_detect_module.grid(row=1, column=0, padx=10, pady=6, sticky='w')
        self.detect_menu = ctk.CTkOptionMenu(
            config_frame,
            values=list(self.detect_class_map.keys()),
            variable=self.detect_module_var,
            command=self._on_detect_module_changed,
            width=200,
        )
        self.detect_menu.grid(row=1, column=1, padx=10, pady=6, sticky='w')

        self.detect_config_frame = ctk.CTkFrame(config_frame, corner_radius=8)
        self.detect_config_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=(6, 0), sticky='ew')
        self.detect_config_frame.grid_columnconfigure(1, weight=1)

        self._label_input_module = ctk.CTkLabel(config_frame, text=self._t('input_module'))
        self._label_input_module.grid(row=3, column=0, padx=10, pady=6, sticky='w')
        self.input_menu = ctk.CTkOptionMenu(
            config_frame,
            values=list(self.camera_class_map.keys()),
            variable=self.input_module_var,
            command=self._on_input_module_changed,
            width=200,
        )
        self.input_menu.grid(row=3, column=1, padx=10, pady=6, sticky='w')

        self.input_config_frame = ctk.CTkFrame(config_frame, corner_radius=8)
        self.input_config_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=(6, 0), sticky='ew')
        self.input_config_frame.grid_columnconfigure(1, weight=1)

        self._label_exec_module = ctk.CTkLabel(config_frame, text=self._t('exec_module'))
        self._label_exec_module.grid(row=5, column=0, padx=10, pady=6, sticky='w')
        self.exec_menu = ctk.CTkOptionMenu(
            config_frame,
            values=list(self.execution_class_map.keys()),
            variable=self.exec_module_var,
            command=lambda v: self._on_exec_module_changed(),
            width=200,
        )
        self.exec_menu.grid(row=5, column=1, padx=10, pady=6, sticky='w')

        self.exec_config_frame = ctk.CTkFrame(config_frame, corner_radius=8)
        self.exec_config_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=(6, 10), sticky='ew')
        self.exec_config_frame.grid_columnconfigure(1, weight=1)

        run_control_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        run_control_frame.grid(row=2, column=0, padx=10, pady=(0, 8), sticky='ew')
        run_control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(run_control_frame, textvariable=self.status_var).grid(row=0, column=0, padx=(10, 6), pady=(10, 6), sticky='w')
        self.fps_text_label = ctk.CTkLabel(run_control_frame, textvariable=self.fps_text_var, width=56, anchor='e')
        self.fps_text_label.grid(row=0, column=2, padx=(8, 4), pady=(10, 6), sticky='e')
        self.fps_value_label = ctk.CTkLabel(run_control_frame, textvariable=self.fps_value_var, width=40, anchor='w')
        self.fps_value_label.grid(row=0, column=3, padx=(0, 10), pady=(10, 6), sticky='w')

        self.run_button = ctk.CTkButton(
            run_control_frame,
            textvariable=self.run_button_var,
            command=self._on_run_or_pause_click,
            width=200,
        )
        self.run_button.grid(row=1, column=0, columnspan=4, padx=10, pady=(0, 10), sticky='ew')

        control_frame = ctk.CTkFrame(main_frame, corner_radius=10)
        control_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky='ew')
        control_frame.grid_columnconfigure(0, weight=1)

        self.aim_runtime_frame = ctk.CTkFrame(control_frame, corner_radius=8)
        self.aim_runtime_frame.grid(row=0, column=0, padx=10, pady=(10, 6), sticky='ew')
        self.aim_runtime_frame.grid_columnconfigure(0, weight=1)

        self.auto_aim_checkbutton = ctk.CTkCheckBox(
            self.aim_runtime_frame,
            text=UI_TEXT['en']['auto_aim'],
            variable=self.auto_aim_var,
            command=self.toggle_auto_aim,
            state='disabled',
        )
        self.auto_aim_checkbutton.grid(row=0, column=0, padx=10, pady=(10, 4), sticky='w')

        self.auto_shoot_checkbutton = ctk.CTkCheckBox(
            self.aim_runtime_frame,
            text=UI_TEXT['en']['auto_shoot'],
            variable=self.auto_shoot_var,
            command=self.toggle_auto_shoot,
            state='disabled',
        )
        self.auto_shoot_checkbutton.grid(row=1, column=0, padx=10, pady=(4, 10), sticky='w')

        self.exit_button = ctk.CTkButton(control_frame, text=UI_TEXT['en']['exit'], command=self.exit_program, fg_color='#8B3A3A', hover_color='#6B2E2E')
        self.exit_button.grid(row=1, column=0, padx=10, pady=(6, 10), sticky='ew')

        self._on_detect_module_changed(self.detect_module_var.get())
        self._on_input_module_changed(self.input_module_var.get())
        self._on_exec_module_changed()
        self._refresh_aim_runtime_visibility()
        self._refresh_run_control_state()

    def change_language(self, choice: str):
        self.lang = 'zh' if choice == '简体中文' else 'en'

        self.title(UI_TEXT[self.lang]['title'])
        self.fps_text_var.set(UI_TEXT[self.lang]['fps'])
        if self._title_basic is not None:
            self._title_basic.configure(text=self._t('basic_settings'))
        if self._label_startup_config is not None:
            self._label_startup_config.configure(text=self._t('startup_config'))
        if self._label_detect_module is not None:
            self._label_detect_module.configure(text=self._t('detect_module'))
        if self._label_input_module is not None:
            self._label_input_module.configure(text=self._t('input_module'))
        if self._label_exec_module is not None:
            self._label_exec_module.configure(text=self._t('exec_module'))
        if self.auto_aim_checkbutton is not None:
            self.auto_aim_checkbutton.configure(text=UI_TEXT[self.lang]['auto_aim'])
        if self.auto_shoot_checkbutton is not None:
            self.auto_shoot_checkbutton.configure(text=UI_TEXT[self.lang]['auto_shoot'])
        if self.language_title_label is not None:
            self.language_title_label.configure(text=UI_TEXT[self.lang]['language'])
        self._on_detect_module_changed(self.detect_module_var.get())
        self._on_input_module_changed(self.input_module_var.get())
        self._on_exec_module_changed()
        self._refresh_run_control_state()

    def _refresh_run_control_state(self):
        if self.dead_eye_core is None:
            self.status_var.set(UI_TEXT[self.lang]['idle'])
            self.run_button_var.set(UI_TEXT[self.lang]['start_run'])
        elif self.dead_eye_core.if_paused:
            self.status_var.set(UI_TEXT[self.lang]['paused'])
            self.run_button_var.set(UI_TEXT[self.lang]['continue'])
        else:
            self.status_var.set(UI_TEXT[self.lang]['running'])
            self.run_button_var.set(UI_TEXT[self.lang]['pause'])
        self.exit_button.configure(text=UI_TEXT[self.lang]['exit'])

    def _find_deadeye_auto_aiming_module(self):
        if self.dead_eye_core is None:
            return None
        for m in self.dead_eye_core.execution_modules:
            if isinstance(m, DeadEyeAutoAimingModule):
                return m
        return None

    def toggle_auto_aim(self):
        mod = self._find_deadeye_auto_aiming_module()
        if mod is None:
            return
        mod.enable_auto_aim = not mod.enable_auto_aim
        self.auto_aim_var.set(mod.enable_auto_aim)

    def toggle_auto_shoot(self):
        mod = self._find_deadeye_auto_aiming_module()
        if mod is None:
            return
        mod.enable_auto_shoot = not mod.enable_auto_shoot
        self.auto_shoot_var.set(mod.enable_auto_shoot)

    def toggle_pause(self):
        if self.dead_eye_core is None:
            return
        if self.dead_eye_core.switch_pause_state():
            print('Paused.')
        else:
            print('Continue working.')
        self._refresh_run_control_state()

    def _on_run_or_pause_click(self):
        if self.dead_eye_core is None:
            self.start_pipeline()
        else:
            self.toggle_pause()

    def on_closing(self):
        self.exit_program()

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
        if self.dead_eye_core is None:
            return
        if key in self.hot_keys.keys():
            function, event_type = self.hot_keys[key]
            if function == self.Function.PAUSE and key_event_type == event_type:
                self.toggle_pause()
            if function == self.Function.EXIT and key_event_type == event_type:
                self.after(0, self.exit_program)
            if function == self.Function.TOGGLE_AUTO_AIM:
                self.toggle_auto_aim()

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def _on_exec_module_changed(self):
        self._rebuild_exec_config_panel()
        self._refresh_aim_runtime_visibility()

    def _refresh_aim_runtime_visibility(self):
        if self.aim_runtime_frame is None:
            return
        should_show = False
        name = self.exec_module_var.get().strip()
        if name:
            exec_cls = self.execution_class_map.get(name)
            if exec_cls is not None and issubclass(exec_cls, ConfigurableMixin):
                keys = [f.key for f in exec_cls.get_config_spec()]
                if ('enable_auto_aim' in keys) or ('enable_auto_shoot' in keys):
                    should_show = True
        if should_show:
            self.aim_runtime_frame.grid()
        else:
            self.aim_runtime_frame.grid_remove()

    def _clear_dynamic_panel(self, widgets: list[ctk.CTkBaseClass], vars_dict: dict[str, tk.Variable]):
        for w in widgets:
            try:
                w.destroy()
            except Exception:
                pass
        widgets.clear()
        vars_dict.clear()

    def _rebuild_dynamic_panel(self, parent_frame, module_cls, vars_dict: dict[str, tk.Variable], widgets: list[ctk.CTkBaseClass]):
        self._clear_dynamic_panel(widgets, vars_dict)
        if parent_frame is None:
            return
        if module_cls is None or not issubclass(module_cls, ConfigurableMixin):
            parent_frame.grid_remove()
            return
        fields = module_cls.get_config_spec()
        if not fields:
            parent_frame.grid_remove()
            return

        parent_frame.grid()
        parent_frame.grid_columnconfigure(0, weight=0)
        parent_frame.grid_columnconfigure(1, weight=1)
        parent_frame.grid_columnconfigure(2, weight=0)
        parent_frame.grid_columnconfigure(3, weight=1)

        title = ctk.CTkLabel(
            parent_frame,
            text=self._t('module_params_fmt', name=module_cls.__name__),
        )
        title.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 6), sticky='w')
        widgets.append(title)

        row = 1
        col = 0
        for field in fields:
            if field.field_type == 'bool':
                var = tk.BooleanVar(value=bool(field.default))
                widget = ctk.CTkCheckBox(parent_frame, text=field.label, variable=var)
                widget.grid(row=row, column=col, padx=10, pady=4, sticky='w')
                widgets.append(widget)
                vars_dict[field.key] = var
                col += 1
                if col > 1:
                    col = 0
                    row += 1
                continue

            var = tk.StringVar(value=str(field.default))
            vars_dict[field.key] = var

            label = ctk.CTkLabel(parent_frame, text=field.label)
            label.grid(row=row, column=0, padx=10, pady=4, sticky='w')
            widgets.append(label)

            if field.field_type == 'choice' and field.choices:
                menu = ctk.CTkOptionMenu(parent_frame, values=field.choices, variable=var, width=180)
                menu.grid(row=row, column=1, padx=10, pady=4, sticky='w')
                widgets.append(menu)
            elif field.field_type == 'path':
                entry = ctk.CTkEntry(parent_frame, textvariable=var)
                entry.grid(row=row, column=1, padx=10, pady=4, sticky='ew')
                widgets.append(entry)

                def make_choose_func(ui, key_name: str, filetypes):
                    def _choose():
                        if filetypes:
                            p = filedialog.askopenfilename(
                                title=ui._t('dialog_choose_file'),
                                initialdir=os.path.abspath('.'),
                                filetypes=filetypes,
                            )
                        else:
                            p = filedialog.askopenfilename(
                                title=ui._t('dialog_choose_path'),
                                initialdir=os.path.abspath('.'),
                            )
                        if p:
                            vars_dict[key_name].set(p)
                    return _choose

                btn = ctk.CTkButton(
                    parent_frame,
                    text=self._t('btn_choose'),
                    width=70,
                    command=make_choose_func(self, field.key, field.path_filetypes),
                )
                btn.grid(row=row, column=2, padx=10, pady=4, sticky='e')
                widgets.append(btn)
            else:
                entry = ctk.CTkEntry(parent_frame, textvariable=var, width=180)
                entry.grid(row=row, column=1, padx=10, pady=4, sticky='w')
                widgets.append(entry)

            row += 1

    def _on_input_module_changed(self, input_value: str):
        camera_cls = self.camera_class_map.get(input_value)
        self._rebuild_dynamic_panel(self.input_config_frame, camera_cls, self.input_config_vars, self.input_config_widgets)

    def _on_detect_module_changed(self, detect_value: str):
        detect_cls = self.detect_class_map.get(detect_value)
        self._rebuild_dynamic_panel(
            self.detect_config_frame, detect_cls, self.detect_config_vars, self.detect_config_widgets
        )

    def _rebuild_exec_config_panel(self):
        frame = self.exec_config_frame
        name = self.exec_module_var.get().strip()
        widgets = self.exec_config_widgets
        vars_dict = self.exec_config_vars
        if frame is None:
            return
        self._clear_dynamic_panel(widgets, vars_dict)
        if not name:
            frame.grid_remove()
            return
        exec_cls = self.execution_class_map.get(name)
        if exec_cls is None:
            frame.grid_remove()
            return
        if not issubclass(exec_cls, ConfigurableMixin):
            frame.grid_remove()
            return
        fields = exec_cls.get_config_spec()
        if not fields:
            frame.grid_remove()
            return
        self._rebuild_dynamic_panel(frame, exec_cls, vars_dict, widgets)

    @staticmethod
    def _instantiate_execution_module(exec_cls: type, view_range: tuple[int, int], raw_config: dict[str, object]):
        if issubclass(exec_cls, ConfigurableMixin):
            config = exec_cls.parse_config(raw_config)
        else:
            config = {}
        try:
            return exec_cls(view_range, **config)
        except TypeError:
            try:
                return exec_cls(**config)
            except TypeError:
                return exec_cls(view_range=view_range, **config)

    def start_pipeline(self):
        if self.dead_eye_core is not None:
            self._set_status(self._t('status_already_running'))
            return

        input_module_name = self.input_module_var.get().strip()
        detect_module_name = self.detect_module_var.get().strip()

        try:
            view_range = (640, 640)

            detect_cls = self.detect_class_map.get(detect_module_name) or YoloDetector
            raw_detect_config: dict[str, object] = {}
            for k, v in self.detect_config_vars.items():
                raw_detect_config[k] = v.get()
            if issubclass(detect_cls, ConfigurableMixin):
                detect_config = detect_cls.parse_config(raw_detect_config)
            else:
                detect_config = {}
            if detect_cls is YoloDetector:
                model_path = str(detect_config.get('model_path', '')).strip()
                if not model_path or not os.path.isfile(model_path):
                    self._set_status(self._t('status_model_invalid'))
                    return
            self._set_status(self._t('status_init_detect', name=detect_cls.__name__))
            self.detect_module = detect_cls(**detect_config)

            camera_cls = self.camera_class_map.get(input_module_name) or SimpleScreenShotCamera
            raw_input_config: dict[str, object] = {}
            for k, v in self.input_config_vars.items():
                raw_input_config[k] = v.get()
            if issubclass(camera_cls, ConfigurableMixin):
                input_config = camera_cls.parse_config(raw_input_config)
            else:
                input_config = {}
            self._set_status(self._t('status_init_input', name=camera_cls.__name__))
            try:
                self.camera_module = camera_cls(view_range, **input_config)
            except TypeError:
                self.camera_module = camera_cls(view_range=view_range, **input_config)

            execution_modules: list[ExecutionModule] = []
            exec_name = self.exec_module_var.get().strip()
            exec_cls = self.execution_class_map.get(exec_name)
            if exec_cls is None:
                exec_cls = DeadEyeAutoAimingModule
            raw_exec_config: dict[str, object] = {}
            for k, v in self.exec_config_vars.items():
                raw_exec_config[k] = v.get()
            self._set_status(self._t('status_init_exec', name=exec_cls.__name__))
            execution_modules.append(
                self._instantiate_execution_module(exec_cls, view_range, raw_exec_config)
            )

            self._set_status(self._t('status_starting_core'))
            self.dead_eye_core = DeadEyeCore(
                self.camera_module,
                self.detect_module,
                execution_modules=execution_modules,
            )
            self.dead_eye_core.fps_displayer = self.fps_value_var

            if self.dead_eye_core.if_paused:
                self.dead_eye_core.switch_pause_state()

            self.auto_aim_checkbutton.configure(state='normal')
            self.auto_shoot_checkbutton.configure(state='normal')

            aim_mod = self._find_deadeye_auto_aiming_module()
            if aim_mod is not None:
                self.auto_aim_var.set(aim_mod.enable_auto_aim)
                self.auto_shoot_var.set(aim_mod.enable_auto_shoot)

            self._refresh_run_control_state()
        except Exception as e:
            self._set_status(self._t('status_start_failed', error=str(e)))
            return

    def exit_program(self):
        print('CLOSING...')
        try:
            if self.dead_eye_core is not None:
                self.dead_eye_core.on_exit()
        finally:
            try:
                if self.detect_module is not None:
                    self.detect_module.on_exit()
            except Exception:
                pass
        self.bot_closed.release()
        sys.exit(0)


if __name__ == '__main__':
    print('Starting UI...')
    ui = DeadEyeUI()
    ui.mainloop()
    ui.bot_closed.acquire()
