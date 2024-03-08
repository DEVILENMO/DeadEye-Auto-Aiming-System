# DeadEye 辅助瞄准系统

DeadEye 辅助瞄准系统是一款高性能的图像辅助瞄准工具，旨在通过图像处理和目标追踪技术提高射击精度。该系统通过分析屏幕截图，检测并追踪目标，最后通过模拟操作实现辅助瞄准和自动扳机功能。

The DeadEye Aiming Assist System is a high-performance image aiming tool designed to improve shooting accuracy through image processing and target tracking technology. The system analyzes screen captures to detect and track targets, and ultimately facilitates aiming and triggers actions through simulated operations.

## 特性

- **高速截图**: 使用 ScreenShotHelper 类快速截取屏幕图像，并自动适配不同的屏幕分辨率。
- **目标检测**: 目标检测 YoloDetector 类继承于 DetectModule 基类，基于 Yolov7 目标检测神经网络，支持基于.pt权重文件的直接检测或使用 TensorRT 部署后的 .onnx 模型进行精确的目标检测。
- **目标追踪**: 结合匈牙利算法和卡尔曼滤波优化目标追踪的精确性和稳定性。
- **辅助瞄准**: 辅助瞄准模块 DeadEyeAutoAimingModule 类继承于 AutoAimModule 基类，利用 PID 控制算法实现平滑的辅助瞄准。
- **易于扩展**: 用户可以根据需求定制或扩展自己的目标检测模块或辅助瞄准模块。

## Features

- **High-speed Screenshot**: Uses the ScreenShotHelper class to quickly capture screen images and automatically adapts to different screen resolutions.
- **Target Detection**: The YoloDetector class, inheriting from the DetectModule base class, is based on the Yolov7 target detection neural network, supporting direct detection with .pt weight files or precise target detection using .onnx models deployed with TensorRT.
- **Target Tracking**: Combines the Hungarian algorithm and Kalman filter to optimize the accuracy and stability of target tracking.
- **Assist Aiming**: The DeadEyeAutoAimingModule class, inheriting from the AutoAimModule base class, uses PID control algorithms to achieve smooth assist aiming.
- **Easy to Extend**: Users can customize or extend their own target detection modules or assist aiming modules according to their needs.

## 技术路线

请按照原有的格式继续将其他部分的中文描述翻译成英文并添加到 Markdown 文件中。

...

## 使用指南

### 运行程序

1. 执行 `main.py` 文件启动程序。
2. 程序运行中可以使用以下快捷键：
   - `P`: 暂停/继续程序
   - `O`: 完全结束程序

### Usage Guide

#### Running the Program

1. Run the `main.py` file to start the program.
2. During program execution, the following shortcuts can be used:
   - `P`: Pause/continue the program
   - `O`: Completely terminate the program

### 效果演示

以下动图展示了此项目可提供的辅助瞄准效果：

![辅助瞄准效果演示](./effect_test.gif)

### Demonstration

The following animation demonstrates the aiming assist effect provided by this project:

![Aiming Assist Effect Demonstration](./effect_test.gif)

### 注意事项

- **注1**：此项目是从之前损坏的项目代码中整合而来，尚未进行完整的重新测试。可能存在未知的错误或问题，请在使用时注意。
- **注2**：项目仅供学习和测试用途，严禁用于游戏作弊或任何违反游戏使用规则的行为。

### Precautions

- **Note 1**: This project is integrated from previously damaged project code and has not been completely retested. There may be unknown errors or issues, so please be cautious when using it.
- **Note 2**: The project is for learning and testing purposes only, and it is strictly forbidden to use it for game cheating or any behavior that violates game usage rules.