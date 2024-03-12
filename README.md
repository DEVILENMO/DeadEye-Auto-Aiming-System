# DeadEye 辅助瞄准系统 / DeadEye Auto Aiming System

DeadEye 辅助瞄准系统是一款高性能的图像辅助瞄准工具，旨在通过图像处理和目标追踪技术提高射击精度。该系统通过分析屏幕截图，检测并追踪目标，最后通过模拟操作实现辅助瞄准和自动扳机功能。

The DeadEye Aiming Assist System is a high-performance image aiming tool designed to improve shooting accuracy through image processing and target tracking technology. The system analyzes screen captures to detect and track targets, and ultimately facilitates aiming and triggers actions through simulated operations.

## 特性

- **高速截图**: 使用 ScreenShotHelper 类快速截取屏幕图像，并自动适配不同的屏幕分辨率。
- **目标检测**: 目标检测 YoloDetector 类继承于 DetectModule 基类，基于 Yolov8 目标检测神经网络，支持基于.pt权重文件的直接检测或使用 TensorRT 部署后的 .onnx /.trt 模型进行精确的目标检测（目前只测试了 .pt 模型，确认可以正常使用）。
- **目标追踪**: 结合匈牙利算法和卡尔曼滤波优化目标追踪的精确性和稳定性。
- **辅助瞄准**: 辅助瞄准模块 DeadEyeAutoAimingModule 类继承于 AutoAimModule 基类，利用 PID 控制算法实现平滑的辅助瞄准。
- **易于扩展**: 用户可以根据需求定制或扩展自己的目标检测模块或辅助瞄准模块。

## Features

- **High-speed Screenshot**: Uses the ScreenShotHelper class to quickly capture screen images and automatically adapts to different screen resolutions.
- **Target Detection**: The target detection class YoloDetector is derived from the base class DetectModule. It is based on the Yolov8 target detection neural network and supports direct detection using .pt weight files or precise target detection using deployed TensorRT models with .onnx or .trt formats (currently, only the .pt model has been tested and confirmed to work properly).
- **Target Tracking**: Combines the Hungarian algorithm and Kalman filter to optimize the accuracy and stability of target tracking.
- **Assist Aiming**: The DeadEyeAutoAimingModule class, inheriting from the AutoAimModule base class, uses PID control algorithms to achieve smooth assist aiming.
- **Easy to Extend**: Users can customize or extend their own target detection modules or assist aiming modules according to their needs.


## 技术路线

### 截图模块 (ScreenShotHelper) / Screenshot Module (ScreenShotHelper)

- **自动分辨率计算**: 自动检测屏幕分辨率并调整截图尺寸适应不同大小的窗口需求。
- **高速截图**: 采用 `dxcam` 或 `mss` 实现高速截图功能。


### 目标检测模块 (DetectModule)

- **YoloDetector 类**: 可以使用 `.pt` 模型进行直接检测，也支持使用 `.onnx / .trt` 模型的基于 TensorRT 的部署方式进行高效检测。

### 目标追踪

- **目标类 (Target)**: 基于检测结果，使用匈牙利算法进行帧间目标匹配和编号，此外使用卡尔曼滤波算法对目标位置进行预测和优化，以实现平滑的追踪效果。

### 瞄准模块 (AutoAimModule)

- **DeadEyeAutoAimingModule**: 基础的辅助瞄准模块类，实现了核心的瞄准功能。
- **定制扩展**: 允许用户继承或修改基类，创建个性化的辅助瞄准模块。

## Technological Path

### Screenshot Module (ScreenShotHelper)
- **Automatic Resolution Calculation**: Automatically detects screen resolution and adjusts the screenshot size to accommodate different window sizes.
- **High-Speed Screenshots**: Uses advanced libraries like `dxcam` or `mss` to capture high-speed screenshots with minimal performance impact.

### Target Detection Module (YoloDetector)
- **Model Compatibility**: Supports direct detection using `.pt` model files, offering flexibility in model training and deployment.
- **Efficient Detection**: Incorporates `.onnx` models with TensorRT for efficient and accurate target detection, optimizing for lower latency and higher throughput.

### Target Tracking
- **Inter-Frame Matching**: Utilizes the Hungarian algorithm for consistent target matching across frames, ensuring reliable tracking.
- **Position Prediction and Optimization**: Employs Kalman filter algorithms to predict and optimize target positions, resulting in smoother tracking and better performance.

### Aiming Module (DeadEyeAutoAimingModule)
- **Core Aiming Functions**: Implements the fundamental features necessary for assisted aiming, such as target locking and trajectory adjustment.
- **Customization and Extension**: Designed with extensibility in mind, allowing users to inherit from or modify the base class to create customized aiming modules tailored to specific requirements.

## 使用指南

### 运行程序

1. 执行 `main.py` 文件启动程序。
2. 程序运行中可以使用以下快捷键：
   - `P`: 暂停/继续程序
   - `O`: 完全结束程序

## Usage Guide

### Running the Program

1. Run the `main.py` file to start the program.
2. During program execution, the following shortcuts can be used:
   - `P`: Pause/continue the program
   - `O`: Completely terminate the program

### 效果演示

以下动图展示了此项目可提供的辅助瞄准效果：

### Demonstration

The following animation demonstrates the aiming assist effect provided by this project:

![辅助瞄准效果演示 / Aiming Assist Effect Demonstration](./effect_test.gif)

### 注意事项

- **注1**：此项目是从之前损坏的项目代码中整合而来，尚未进行完整的重新测试。可能存在未知的错误或问题，请在使用时注意。
- **注2**：项目仅供学习和测试用途，严禁用于游戏作弊或任何违反游戏使用规则的行为。
- **注3**：2024.3.12，项目已修复了绝大多数错误，并且更换目标检测模块至 Yolov8 版本，使用 .pt 权重文件进行目标检测并且辅助瞄准测试无误，暂未测试 .onnx 及 .trt 权重文件的推理流程。

### Precautions

- **Note 1**: This project is integrated from previously damaged project code and has not been completely retested. There may be unknown errors or issues, so please be cautious when using it.
- **Note 2**: The project is for learning and testing purposes only, and it is strictly forbidden to use it for game cheating or any behavior that violates game usage rules.
- **Note 3**: As of March 12, 2024, the project has fixed most of the errors and has updated the target detection module to the Yolov8 version. It now uses the .pt weight file for target detection, and the assisted aiming has been tested without issues. The inference process for .onnx and .trt weight files has not been tested yet.