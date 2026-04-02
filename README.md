# DeadEye 辅助瞄准系统 / DeadEye Auto Aiming System

<img src="DeadEye.png" alt="DeadEye" style="width: 100%; height: auto;"/>

DeadEye 目标追踪系统是一款高性能的图像辅助目标追踪工具。作为示例，提供了在FPS游戏中的目标追踪以及对应的瞄准效果演示。示例程序通过分析屏幕截图，检测并追踪目标，最后通过模拟操作实现辅助瞄准和自动扳机功能。不仅如此，我还将这个系统应用在光学激光控制和监控探头的目标追踪中，虽然这些领域的示例程序没有提供，但用户可以通过简单修改相关模块，将这个系统应用于不同领域。

DeadEye target tracking system is a high-performance image-assisted target tracking tool. As an example, it provides a demonstration of target tracking and corresponding aiming effects in FPS games. The sample program analyzes screenshots, detects and tracks targets, and finally implements assisted aiming and automatic trigger functions through simulated operations. Additionally, I have applied this system to optical laser control and monitoring target tracking. Although sample programs for these fields are not provided, users can apply this system in different domains by simply modifying the corresponding modules.

## 近期更新（2026）

- **流水线**：`DeadEyeCore`（`deadeye/core/dead_eye_core.py`）中数据流为：输入源取帧 → `DetectModule.target_detect` → 匈牙利匹配与卡尔曼更新 `Target` 列表 → 依次调用各 **`ExecutionModule.update_targets(frame, target_list)`**。
- **目录规范**：业务模块目录统一为 `*_module`：`deadeye/input_module`、`deadeye/detection_module`、`deadeye/execution_module`（原 `input` / `detection` / `aim` / `output` 已调整；`VideoWriterOutput` 已并入 `execution_module`）。
- **执行模块抽象**：瞄准与写视频等后处理统一继承 **`ExecutionModule`**，在检测与追踪完成后对当前帧与 `target_list` 调用 `update_targets()`；核心类 **`DeadEyeCore`** 使用 `execution_modules` 列表按顺序执行。
- **移除 NoAimModule**：不再需要空瞄准占位类；默认可选 `DeadEyeAutoAimingModule` 或 `VideoWriterOutput` 等。
- **图形界面**：顶部「基本设置」卡片（语言）；「启动配置」内含检测 / 输入 / 执行模块及动态参数（YOLO **模型路径**在检测模块参数中配置）；「运行控制」卡片内为状态、**FPS** 与 **开始运行 / 暂停 / 继续** 合一按钮；窗口默认高度加大以便参数区完整显示。
- **视频输入**：`VideoFileCamera` 改为选择**单个视频文件**（`video_path`），不再扫描整个文件夹。
- **多语言**：界面新增文案已纳入语言字典，切换中英文会同步更新卡片标题、状态提示、文件对话框与动态表单中的「选择」等文本。
- **UI 组合**：各阶段模块通过 `main.py` 扫描 `*_module` 包与 `ConfigurableMixin` 动态生成参数表单。

## Recent Updates (2026)

- **Pipeline**: In `DeadEyeCore` (`deadeye/core/dead_eye_core.py`), the flow is: grab a frame from the input camera → `DetectModule.target_detect` → Hungarian matching + Kalman updates on `Target` → each **`ExecutionModule.update_targets(frame, target_list)`** in order.
- **Package layout**: Module folders are named `*_module`: `deadeye/input_module`, `deadeye/detection_module`, `deadeye/execution_module` (replacing the old `input` / `detection` / `aim` / `output` layout; `VideoWriterOutput` now lives under `execution_module`).
- **Execution abstraction**: Post-detection steps (aim assist, video export, etc.) share the **`ExecutionModule`** base class and implement `update_targets()`; **`DeadEyeCore`** runs a list `execution_modules` in order.
- **NoAimModule removed**: No empty “no-op” aim class; defaults point to real modules such as `DeadEyeAutoAimingModule` or `VideoWriterOutput`.
- **UI**: A top **Basic settings** card (language); **Pipeline configuration** for detector / input / execution with dynamic fields (**model path** is part of detector settings); **Run control** combines status, **FPS**, and a single **Start / Pause / Resume** button; taller default window for form layout.
- **Video input**: `VideoFileCamera` uses a **single file** (`video_path`), not a directory of videos.
- **Localization**: New UI strings go through the language dictionary so titles, status messages, file dialogs, and browse buttons update with English / 简体中文.
- **UI composition**: `main.py` discovers classes under `*_module` packages and builds `ConfigurableMixin` forms for their settings.

## 特性

- **高速截图**: 使用 ScreenShotHelper 类快速截取屏幕图像，并自动适配不同的屏幕分辨率。
- **目标检测**: 目标检测 YoloDetector 类继承于 DetectModule 基类，基于 Yolov8 目标检测神经网络，支持基于 `.pt` 权重文件的直接检测或使用 TensorRT 部署后的 `.trt` / `.engine` 模型进行精确的目标检测。
- **目标追踪**: 结合匈牙利算法和卡尔曼滤波优化目标追踪的精确性和稳定性。
- **执行模块**: `DeadEyeAutoAimingModule`、`VideoWriterOutput` 等继承 **`ExecutionModule`**，在追踪更新后对帧与目标列表执行瞄准、写带框视频等动作；辅助瞄准使用 PID 控制算法。
- **易于扩展**: 可在 `input_module`、`detection_module`、`execution_module` 中按基类扩展；鼠标等行为由执行模块内 `MouseControlModule` 等实现，按需自行接入硬件或驱动。
- **多语言UI**: 具有简单直观的用户界面,支持英文和简体中文两种语言,用户可以方便地切换语言或者增加新的语言支持。

## Features

- **High-speed Screenshot**: Uses the ScreenShotHelper class to quickly capture screen images and automatically adapts to different screen resolutions.
- **Target Detection**: The target detection class YoloDetector is derived from the base class DetectModule. It is based on the Yolov8 target detection neural network and supports direct detection using `.pt` weight files or precise target detection using deployed TensorRT models with `.trt` / `.engine` formats.
- **Target Tracking**: Combines the Hungarian algorithm and Kalman filter to optimize the accuracy and stability of target tracking.
- **Execution modules**: Classes such as `DeadEyeAutoAimingModule` and `VideoWriterOutput` inherit **`ExecutionModule`** and run after tracking updates (aim assist, annotated video export, etc.); aim assist uses a PID controller.
- **Easy to Extend**: Extend `input_module`, `detection_module`, and `execution_module` via their base classes; mouse or other actuation is wired inside execution modules (e.g. `MouseControlModule` patterns) for your own hardware.
- **Multi-language UI**: Features a simple and intuitive user interface that supports both English and Simplified Chinese. Users can easily switch between languages or add support for new languages.


## 技术路线

### 输入模块 (BaseCamera) — `deadeye/input_module`
作为示例实现了快速的截图相机 `SimpleScreenShotCamera` 以及单文件视频源 `VideoFileCamera`（选择**一个**视频路径）。用户可继承 `BaseCamera` 实现 `get_image()`，通过 `main.py` 的 UI 或自行组装 `DeadEyeCore` 使用。
- **自动分辨率计算**: 自动检测屏幕分辨率并调整截图尺寸适应不同大小的窗口需求。
- **高速截图**: 采用 `dxcam` 或 `mss` 实现高速截图功能。
- **定制扩展**：你只需要继承 `BaseCamera` 类并且实现 `get_image()` 函数返回恰当的图像就可以自定义你的相机了。


### 目标检测模块 (DetectModule) — `deadeye/detection_module`
作为示例实现了基于 Yolov8 的 `YoloDetector` 类（支持 `ConfigurableMixin`，模型路径等在模块参数中配置）。用户可实现自己的检测类，保证输出格式与现有流水线一致即可。
- **YoloDetector 类**: 可以使用 `.pt` 模型进行直接检测，也支持使用 `.trt` / `.engine` 模型的基于 TensorRT 的部署方式进行高效检测。

### 目标追踪
- **目标类 (Target)**: 基于检测结果，使用匈牙利算法进行帧间目标匹配和编号，此外使用卡尔曼滤波算法对目标位置进行预测和优化，以实现平滑的追踪效果。

### 执行模块 (ExecutionModule) — `deadeye/execution_module`
示例包括 `DeadEyeAutoAimingModule`（PID 辅助瞄准）与 `VideoWriterOutput`（画框并写出视频）。均继承 **`ExecutionModule`**，实现 `update_targets(image, target_list)`。
- **定制扩展**: 可新增继承 `ExecutionModule` 的类，放入 `execution_module` 包后由 UI 自动发现（需满足扫描规则）。

### 鼠标控制模块（MouseControlModule）
- **SimpleMouseController**: 鼠标控制类实例， `DeadEyeAutoAimingModule` 类会在 `__init__` 时实例化这个类并用这个类来控制鼠标。用户需要自己实现这个类里控制鼠标两个函数 `click_left_button` 以及 `move_mouse` ，分别对应点击鼠标左键以及移动鼠标。
- **定制扩展**: 用户可以自由实现这个类，即可轻松使用树莓派或其他设备来控制鼠标，此外，作者并没有为用户实现这个类的两个控制鼠标的函数功能以及其他硬件操控功能，用户编程实现后对应的法律风险需要用户自己承担。

## Technological Path

### Input Module (BaseCamera) — `deadeye/input_module`
Example implementations include `SimpleScreenShotCamera` for fast screen capture and `VideoFileCamera` for a **single** video file path. Inherit `BaseCamera`, implement `get_image()`, and wire it through the `main.py` UI or by constructing `DeadEyeCore` manually.
- **Automatic Resolution Calculation**: Automatically detects screen resolution and adjusts the screenshot size to accommodate different window requirements.
- **High-Speed Screenshot**: Utilizes `dxcam` or `mss` to achieve high-speed screenshot functionality.
- **Custom Extensions**: To customize your camera, simply inherit from the `BaseCamera` class and implement the `get_image()` function to return the appropriate image.

### Target Detection Module (DetectModule) — `deadeye/detection_module`
`YoloDetector` is based on Yolov8 and supports `ConfigurableMixin` (model path and other options in module settings). You may provide your own detector class as long as its output matches the pipeline contract.
- **YoloDetector Class**: Can use `.pt` models for direct detection and also supports efficient detection using `.trt` / `.engine` models through TensorRT deployment.

### Target Tracking
- **Inter-Frame Matching**: Utilizes the Hungarian algorithm for consistent target matching across frames, ensuring reliable tracking.
- **Position Prediction and Optimization**: Employs Kalman filter algorithms to predict and optimize target positions, resulting in smoother tracking and better performance.

### Execution Module (ExecutionModule) — `deadeye/execution_module`
Examples include `DeadEyeAutoAimingModule` (PID aim assist) and `VideoWriterOutput` (draw boxes and write video). All inherit **`ExecutionModule`** and implement `update_targets(image, target_list)`.
- **Custom Extensions**: Add new subclasses under `execution_module` so the UI discovery can pick them up (subject to the package scan rules).

### Mouse control module (MouseControlModule)
- **SimpleMouseController**: Mouse control class instance. The `DeadEyeAutoAimingModule` class will instantiate this class during `__init__` and use this class to control the mouse. Users need to implement the two mouse control functions in this class, `click_left_button` and `move_mouse`, which correspond to clicking the left mouse button and moving the mouse respectively.
- **Customized Extension**: Users can freely implement this class, and can easily use Raspberry Pi or other devices to control the mouse. In addition, the author has not implemented the two mouse control functions and other hardware of this class for users. The corresponding legal risks after the control function and user programming are implemented need to be borne by the user himself.

## 使用指南

### 运行程序

1. 执行 `main.py` 启动图形界面：在「基本设置」中切换语言；在「启动配置」中选择检测 / 输入 / 执行模块并填写参数（如 YOLO 模型路径、视频文件路径等）；在「运行控制」查看状态与 FPS，使用 **开始运行 / 暂停 / 继续** 控制流水线。仍可使用快捷键暂停/继续、退出，以及运行时切换辅助瞄准、自动扳机等选项。
2. 程序运行中可以使用以下默认快捷键：
   - `P` : 暂停/继续程序
   - `O` : 完全结束程序
   - `鼠标左键` : 按下时开启自动瞄准

## Usage Guide

### Running the Program

1. Run `main.py` to open the UI. Use **Basic settings** (language), **Pipeline configuration** (detector / input / execution modules and fields such as YOLO model path and video file path), then **Run control** (status, FPS, Start / Pause / Resume). Hotkeys still pause/resume/exit; runtime toggles include auto-aim and auto-trigger when available.
2. While the program is running, you can use the following default hotkeys:
   - `P` : Pause/Resume the program
   - `O` : Completely exit the program
   - `Left Mouse Button` : Enable auto-aiming while pressed

### 效果演示

以下动图展示了此项目可提供的辅助瞄准效果：

### Demonstration

The following animation demonstrates the aiming assist effect provided by this project:

![辅助瞄准效果演示 / Aiming Assist Effect Demonstration](./effect_test.gif)

### 注意事项

- **注1**：此项目是从之前损坏的项目代码中整合而来，部分功能缺失。
- **注2**：本项目是一个使用屏幕图像基于目标检测网络进行辅助瞄准的程序，仅作为课程项目，供学习和研究测试使用。严禁将本项目用于游戏作弊、盈利等任何可能涉嫌违法的用途。本程序不提供对鼠标、键盘等输入设备的任何直接控制或互动的代码。其他使用者需要在严格遵守AGPL-3.0许可协议的前提下，合法合规地借鉴本仓库的源代码，不得进行任何违法行为。如有任何违法行为发生，与本项目作者无关，作者保留依法追究相关责任人法律责任的权利。
- **注3**：2024.4.25，项目已修复了绝大多数错误，并且更换目标检测模块至 Yolov8 版本，使用 .pt 以及 .trt 权重文件进行目标检测并且辅助瞄准测试无误。
- **注4**：2026 年起对包结构、执行模块抽象与图形界面有较大调整（模块化目录、`ExecutionModule`、`main.py` UI 布局与多语言文案等），细节见上文 **「近期更新（2026）」**；与旧教程或截图不一致时以当前仓库代码为准。

### Precautions

- **Note 1**: This project is integrated from previously damaged project code and some functions were lost.
- **Note 2**: This project is a program that uses screen images based on an object detection network for assisted aiming, solely as a course project for learning and research testing purposes. It is strictly prohibited to use this project for any potentially illegal purposes such as game cheating or profiteering. This program does not provide any code for direct control or interaction with input devices such as mice and keyboards. Other users must strictly comply with the AGPL-3.0 license agreement and legally and compliantly reference the source code of this repository, and must not engage in any illegal activities. If any illegal activities occur, they are unrelated to the author of this project, and the author reserves the right to legally pursue the legal responsibility of relevant responsible persons.
- **Note 3**: As of April 25, 2024, the project has fixed most of the errors and has updated the target detection module to the Yolov8 version. It now uses the .pt and .trt weight file for target detection, and the assisted aiming has been tested without issues.
- **Note 4**: From 2026 onward, package layout, the execution-module abstraction, and the UI were significantly refactored—see **Recent Updates (2026)** above. If older posts mention paths like `aim` / `input` / `output` or names like `AutoAimModule`, prefer the current repository.

### 常见问题

1. TensorRT类

   `FileNotFoundError: Could not find: nvinfer.dll. Is it on your PATH?`

   `FileNotFoundError: Could not find: cudnn64_8.dll. Is it on your PATH?`

   `Could not find module 'C:\...\nvinfer_plugin.dll`

   解决方式在 `deadeye/detection_module/tensorrt_engine.py` 及相关环境配置说明中查阅（若仍引用旧文件名，请以仓库内实际路径为准）。

2. dxcam类

   `AttributeError: module 'dxcam' has no attribute 'output_res'. Did you mean: 'output_info'?`

   请使用以下代码替换 `dxcam` 中的 `__init__` 文件
   ```python
   import weakref
   import time
   from dxcam.dxcam import DXCamera, Output, Device
   from dxcam.util.io import (
      enum_dxgi_adapters,
      get_output_metadata,
   )


   class Singleton(type):
      _instances = {}

      def __call__(cls, *args, **kwargs):
         if cls not in cls._instances:
               cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
         else:
               print(f"Only 1 instance of {cls.__name__} is allowed.")

         return cls._instances[cls]


   class DXFactory(metaclass=Singleton):

      _camera_instances = weakref.WeakValueDictionary()

      def __init__(self) -> None:
         p_adapters = enum_dxgi_adapters()
         self.devices, self.outputs = [], []
         for p_adapter in p_adapters:
               device = Device(p_adapter)
               p_outputs = device.enum_outputs()
               if len(p_outputs) != 0:
                  self.devices.append(device)
                  self.outputs.append([Output(p_output) for p_output in p_outputs])
         self.output_metadata = get_output_metadata()

      def create(
         self,
         device_idx: int = 0,
         output_idx: int = None,
         region: tuple = None,
         output_color: str = "RGB",
         max_buffer_len: int = 64,
      ):
         device = self.devices[device_idx]
         if output_idx is None:
               # Select Primary Output
               output_idx = [
                  idx
                  for idx, metadata in enumerate(
                     self.output_metadata.get(output.devicename)
                     for output in self.outputs[device_idx]
                  )
                  if metadata[1]
               ][0]
         instance_key = (device_idx, output_idx)
         if instance_key in self._camera_instances:
               print(
                  "".join(
                     (
                           f"You already created a DXCamera Instance for Device {device_idx}--Output {output_idx}!\n",
                           "Returning the existed instance...\n",
                           "To change capture parameters you can manually delete the old object using `del obj`.",
                     )
                  )
               )
               return self._camera_instances[instance_key]

         output = self.outputs[device_idx][output_idx]
         output.update_desc()
         camera = DXCamera(
               output=output,
               device=device,
               region=region,
               output_color=output_color,
               max_buffer_len=max_buffer_len,
         )
         self._camera_instances[instance_key] = camera
         time.sleep(0.1)  # Fix for https://github.com/ra1nty/DXcam/issues/31
         return camera

      def device_info(self) -> str:
         ret = ""
         for idx, device in enumerate(self.devices):
               ret += f"Device[{idx}]:{device}\n"
         return ret

      def output_info(self) -> str:
         ret = ""
         for didx, outputs in enumerate(self.outputs):
               for idx, output in enumerate(outputs):
                  ret += f"Device[{didx}] Output[{idx}]: "
                  ret += f"Res:{output.resolution} Rot:{output.rotation_angle}"
                  ret += f" Primary:{self.output_metadata.get(output.devicename)[1]}\n"
         return ret

      def output_res(self):
         res = []
         for didx, outputs in enumerate(self.outputs):
               for idx, output in enumerate(outputs):
                  res.append(output.resolution)
         return res

      def clean_up(self):
         for _, camera in self._camera_instances.items():
               camera.release()


   __factory = DXFactory()


   def create(
      device_idx: int = 0,
      output_idx: int = None,
      region: tuple = None,
      output_color: str = "RGB",
      max_buffer_len: int = 64,
   ):
      return __factory.create(
         device_idx=device_idx,
         output_idx=output_idx,
         region=region,
         output_color=output_color,
         max_buffer_len=max_buffer_len,
      )


   def device_info():
      return __factory.device_info()


   def output_info():
      return __factory.output_info()

   def output_res():
      return __factory.output_res()
   ```

### Frequently Asked Questions

1. TensorRT related problems

   `FileNotFoundError: Could not find: nvinfer.dll. Is it on your PATH?`

   `FileNotFoundError: Could not find: cudnn64_8.dll. Is it on your PATH?`

   `Could not find module 'C:\...\nvinfer_plugin.dll`

   See comments and environment notes near `deadeye/detection_module/tensorrt_engine.py` (legacy filenames in old docs may differ from the current tree).

2. DxCam related problems

   'AttributeError: module 'dxcam' has no attribute 'output_res'. Did you mean: 'output_info'?'

   Please replace the `__init__` file in `dxcam` with the above python code.


## 使用须知

欢迎您使用本仓库中的文件。在使用本仓库中的任何文件之前，请仔细阅读以下使用须知：

1. **版权声明**：本仓库中的原创性文件所有权利均归仓库所有者所有，未经许可不得用于商业用途。

2. **引用与借鉴**：如果您在自己的项目中借鉴、引用或使用了本仓库中的任何文件，请务必在您的项目文档（如README文件）中明确注明出处，并提供指向本仓库的链接。建议的引用格式如下：

   本项目中使用了 [DeadEyeAutoAimingSystem](https://github.com/DEVILENMO/DeadEyeAutoAimingSystem) 仓库中的部分文件，特此表示感谢。

3. **再次分发**：如果您需要再次分发本仓库中的文件，请确保在分发时包含本使用须知，并明确注明原始仓库的链接。

4. **问题反馈**：如果您在使用本仓库中的文件时遇到任何问题，欢迎通过 Issue 或 Pull Request 的方式反馈给我们。我们会尽快处理并给予回复。

## Usage Notice

Welcome to use the files in this repository. Before using any files in this repository, please read the following usage notice carefully:

1. **Copyright Statement**: The original files in this repository are owned by the repository owner. Commercial use is not allowed without permission.

2. **Citation and Reference**: If you reference, cite, or use any files from this repository in your own project, please make sure to clearly indicate the source in your project documentation (such as the README file) and provide a link to this repository. The recommended citation format is as follows:

   This project uses some files from the [DeadEyeAutoAimingSystem](https://github.com/DEVILENMO/DeadEyeAutoAimingSystem) repository. We would like to express our gratitude.

3. **Redistribution**: If you need to redistribute the files from this repository, please ensure that you include this usage notice and clearly state the link to the original repository when distributing.

4. **Feedback**: If you encounter any problems while using the files in this repository, please feel free to provide feedback to us through Issues or Pull Requests. We will handle and respond as soon as possible.

## 参考项目

在开发本项目的过程中,我参考了以下优秀的开源项目。在此对这些项目的贡献者表示感谢!

- [YOLOv7](https://github.com/WongKinYiu/yolov7): YOLOv7 的官方实现。
- [YOLOv8](https://github.com/ultralytics/ultralytics): YOLOv8 的官方实现。
- [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series): 一个全面的 TensorRT 部署 YOLO 系列模型的项目。
- [wang2024yolov10](https://github.com/THU-MIG/yolov10): YOLOv10 的官方实现。

## Reference Projects

During the development of this project, I referenced the following excellent open-source projects. I would like to express our gratitude to the contributors of these projects!

- [YOLOv7](https://github.com/WongKinYiu/yolov7): The official implementation of YOLOv7.
- [YOLOv8](https://github.com/ultralytics/ultralytics): The official implementation of YOLOv8.
- [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series): A comprehensive project for deploying YOLO series models with TensorRT.
- [wang2024yolov10](https://github.com/THU-MIG/yolov10): The official implementation of YOLOv10.

## 开源许可

本项目采用 GNU Affero General Public License v3.0 (AGPLv3) 开源许可证。在使用、修改或分发本项目代码时,你必须同意并遵守 AGPLv3 的所有条款。

请查看 [LICENSE](LICENSE.txt) 文件以获取完整的许可证文本。

## Open Source License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3). By using, modifying, or distributing the code in this project, you agree to comply with all the terms and conditions of AGPLv3.

Please refer to the [LICENSE](LICENSE.txt) file for the full text of the license.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVILENMO/DeadEye-Auto-Aiming-System&type=Date)](https://www.star-history.com/#DEVILENMO/DeadEye-Auto-Aiming-System&Date)