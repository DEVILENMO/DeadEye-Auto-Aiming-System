# DeadEye 通用视觉目标定位与追踪系统框架 / DeadEye: Universal Visual Target Localization & Tracking Framework

<img src="res/DeadEye.png" alt="DeadEye" style="width: 100%; height: auto;"/>

DeadEye 是一个**通用的视觉目标定位与追踪系统框架**：模块化流水线「输入 → 检测 → 追踪 → 执行」，检测层以可插拔后端覆盖 YOLOv5/v7/v8/v11/v12/26 的完整 CUDA 部署链（ONNX Runtime CUDA EP / TensorRT + CUDA Graph，RTX 5090 实测端到端 789–1200 FPS），追踪层提供 IoU 匈牙利匹配、卡尔曼滤波与可配置的目标生命周期，执行层把定位结果对接到任意下游动作。

作为示例，仓库提供了 FPS 游戏屏幕目标追踪与辅助瞄准的效果演示（通过分析屏幕截图检测并追踪目标，再由模拟操作实现辅助瞄准与自动扳机）。同样的框架我已应用在光学激光控制和监控探头追踪上。这两个领域的示例程序未随仓库提供，替换或扩展对应模块之后，DeadEye 可以承担任何需要看住目标、持续跟随的任务。

DeadEye is a **universal visual target localization & tracking framework**: a modular pipeline (*input → detection → tracking → execution*) where the detection layer ships pluggable backends covering the full CUDA deployment chain of YOLOv5/v7/v8/v11/v12/26 (ONNX Runtime CUDA EP / TensorRT + CUDA Graph, measured 789–1200 FPS end-to-end on an RTX 5090), the tracking layer provides IoU-cost Hungarian matching, Kalman filtering and a configurable target lifecycle, and the execution layer connects localization results to arbitrary downstream actions.

As a showcase, the repository includes an FPS screen-tracking & aim-assist demo (detect and track targets from screen captures, then drive simulated aiming/trigger actions). I have also deployed the same framework to optical laser control and surveillance-camera tracking. Those samples are not included here; after swapping or extending the corresponding modules, DeadEye fits any task that must watch a scene and keep following its targets.

## 整体架构 / Architecture

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 输入模块             │   │ 检测模块             │   │ 目标追踪             │   │ 执行模块             │
│ BaseCamera           │ → │ YoloDetector         │ → │ TargetTracker        │ → │ ExecutionModule      │
│ 截屏 / 视频          │   │ 多后端推理           │   │ 匹配 + KF + 生命周期 │   │ 瞄准 PID / 写视频    │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

- **线程模型**（`deadeye/core/dead_eye_core.py`）：相机、检测、追踪/执行三条线程以信号量逐帧交接，检测慢于输入时自动丢帧保最新。
- **数据契约**：检测输出 `[class_name, (x1, y1), (x2, y2)]`（原图坐标）；追踪输出 `Target`（KF 平滑框 + 稳定 ID + 速度）；执行模块拿到「当前帧 + Target 列表」做任意下游动作。
- **实测端到端**（TRT 引擎检测 + 追踪 + 执行层消费目标坐标，无磁盘 I/O）：**约 730 FPS**；执行层换为标注视频写出（mp4v 编码落盘）后降至约 635 FPS，编码与磁盘写入的开销约占 15%。纯检测线程 789–1200 FPS（见检测模块）。

- **Thread model** (`deadeye/core/dead_eye_core.py`): camera, detection and tracking/execution threads hand off frames via semaphores; when detection is slower than input, older frames are dropped in favor of the newest.
- **Data contract**: detections are `[class_name, (x1, y1), (x2, y2)]` in original-image coordinates; tracking yields `Target` objects (KF-smoothed box + stable ID + velocity); execution modules receive (frame, target list).
- **Measured end-to-end** (TRT detection + tracking + coordinate-consuming execution, no disk I/O): **≈730 FPS**; swapping the execution stage for annotated video writing (mp4v encode to disk) drops it to ≈635 FPS, a ~15% cost from encoding and disk writes. Detection thread alone: 789–1200 FPS (see the detection module).

## 输入模块（BaseCamera）· `deadeye/input_module` / Input Module

统一接口：继承 `BaseCamera` 实现 `get_image()`，返回 **RGB uint8 HWC** 图像。内置屏幕截屏相机（`SimpleScreenShotCamera`，视口取屏幕中心）与单视频文件源（`VideoFileCamera`，默认指向 `res/videos/` 示例视频）。

### 截屏后端：原理与实测 / Capture backends: principles & measurements（`screenshot_helper.py`）

三个后端按 `windows-capture → bettercam → mss` 自动降级（UI 可指定）/ Automatic fallback chain `windows-capture → bettercam → mss` (or pin one in the UI):

| 后端 Backend | 原理 Principle | 有效帧率 Effective FPS* | 单次 grab Per grab | 特点 Notes |
|---|---|---|---|---|
| windows-capture | Windows.Graphics.Capture（Rust 绑定）：独立线程回调推送新帧，回调内原生裁剪中心视口 + 零拷贝转换，Python 侧仅保留最新帧<br>Windows.Graphics.Capture (Rust bindings): callback thread pushes new frames, native crop + conversion in the callback, newest frame kept on the Python side | **300 FPS** | 0.004 ms | 帧率受屏幕刷新率约束、内容驱动；静止画面返回 None 属正常<br>refresh-capped, content-driven; `None` on static screens is expected |
| bettercam | DXGI Desktop Duplication（dxcam 维护分支）：`output_color='RGB'` 使颜色转换在驱动/C++ 侧完成，Python 每帧零转换<br>DXGI Desktop Duplication (maintained dxcam fork): conversion moved into the driver/C++ side | **428 FPS** | 1.3 ms | 速度最快；要求显示缩放 100%，分辨率热变更自动重建<br>fastest; needs 100% display scaling, rebuilds on resolution change |
| mss | GDI BitBlt：每次调用全量抓取<br>GDI BitBlt, full grab per call | **194 FPS** | 4.4 ms | 兼容性兜底（RDP/虚拟机）；CPU 占用最高<br>compatibility fallback (RDP/VMs); highest CPU usage |

\* 实测条件：2560×1440 @ 300Hz 主屏、640×640 中心视口、鼠标持续移动制造画面变化、3 秒计数。WGC 帧率严格等于刷新率上限；bettercam 可超过刷新率（部分场景含重复帧）；mss 为主动抓取率。/ Measured on a 2560×1440 @ 300Hz display, 640×640 center viewport, continuous cursor motion, 3 s window.

Unified interface: subclass `BaseCamera` and implement `get_image()` returning **RGB uint8 HWC** images. Two inputs ship built-in: the screen-capture camera (`SimpleScreenShotCamera`, center viewport) and a single-file video source (`VideoFileCamera`, defaulting to the sample under `res/videos/`).

## 检测模块（DetectModule）· `deadeye/detection_module` / Detection Module

`YoloDetector` 按模型后缀自动分发到三个推理后端；预处理（letterbox，预分配缓冲稳态零分配）与后处理（输出头自动解码 + 向量化 NMS）对所有后端共享。

### 推理后端 / Inference backends

| 后端 Backend | 模型格式 Format | 原理 Principle |
|---|---|---|
| UltralyticsBackend | `.pt` | torch 直接推理训练权重（v8/v11/v12/26），启动预热消除首帧卡顿<br>runs training weights directly via torch, with a startup warm-up that removes first-frame stalls |
| OnnxRuntimeBackend | `.onnx` | CUDA EP 自适应（无 GPU 回退 CPU）；IOBinding 输入绑定预处理复用缓冲、输出预分配固定 host 数组，稳态**零内存分配**（每次调用 = 1×H2D + 计算 + 1×D2H）<br>CUDA EP with CPU fallback; IOBinding binds the input to the reused letterbox buffer and pre-allocates outputs, giving zero steady-state allocation (each call = 1×H2D + compute + 1×D2H) |
| TensorRTBackend | `.engine/.plan/.trt` | TRT 10/11 tensor-name API + cuda-python 直连驱动；pinned host / device 显存全预分配、非默认流；**CUDA Graph** 把「H2D→推理→D2H」整链捕获为单图，每帧一次 graph launch<br>TRT 10/11 tensor-name API driven via cuda-python; pinned host and device buffers fully pre-allocated on a non-default stream; **CUDA Graph** captures the whole H2D→inference→D2H chain into one graph, replayed with a single launch per frame |

### 输出头自动解码 / Output-head auto-decoding

- v5 / v7：`[anchors, 5+nc]`（objectness，conf = obj × cls）
- v8 / v11 / v12：`[4+nc, anchors]` 解码头（无 objectness）
- v10 / v26：`[max_det, 6]` 端到端输出，免 NMS

优先使用 Ultralytics 导出元数据（类别名 / imgsz / end2end）消歧，无元数据时按「anchor 维 = 较大维」启发式判断。后处理实测：v8/v11 头（含 NMS）0.08–0.16 ms，v26 端到端头仅 0.01–0.02 ms。

Ultralytics export metadata (names/imgsz/end2end) disambiguates first; without metadata, a larger-dimension-is-anchors heuristic applies. Measured post-processing: 0.08–0.16 ms for v8/v11 heads (with NMS), 0.01–0.02 ms for the v26 end-to-end head (NMS-free).

### 专用检测器：AimLab 颜色检测 / Dedicated detector: AimLab color detection（`aimlab_color_detector.py`）

针对 AimLab 目标球（高饱和青 / 黄色圆球）的专用检测器，作为 YoloDetector 之外的第二选择出现在 UI 检测模块列表中。原理：RGB 转 HSV 后按青（H 85–100）与黄（H 20–40）两个色相区间做 `inRange`，饱和度 / 明度下限排除灰白棋盘格背景，绿色横条（H≈60）因色相相距较远被天然排除；`connectedComponentsWithStats` 给出外接框，再按圆形度过滤（长宽比 ≤1.6、填充率 ≥0.55）剔除辉光光晕与地面反光条。

实测：640×640 单帧约 1.8 ms（约为神经网络后端的千分之一量级），真实截图上精确检出单一球体。可配置项：青 / 黄开关、最小面积、饱和度 / 明度下限、圆形度阈值。

A dedicated detector for AimLab's saturated cyan / yellow target balls, listed alongside YoloDetector in the UI. Principle: RGB→HSV, two hue-band `inRange` masks (cyan H 85–100, yellow H 20–40), saturation/value floors exclude the gray checkered background and the green bar (H≈60) falls far outside both bands; `connectedComponentsWithStats` yields boxes, then a circularity filter (aspect ≤1.6, fill ≥0.55) rejects glow halos and floor reflections. Measured ≈1.8 ms per 640×640 frame; single-ball screenshots detected exactly.

### 性能基准 / Benchmarks（RTX 5090，端到端含预处理与后处理，200 帧均值 / end-to-end, mean of 200 frames）

| 模型 Model | ONNX Runtime CPU | ONNX Runtime CUDA | TensorRT + CUDA Graph |
|---|---|---|---|
| YOLOv11s @ 256 | 168 FPS (5.47 ms) | 188 FPS (4.99 ms) | **1200 FPS (0.83 ms)** |
| YOLOv11s @ 320 | 125 FPS (7.50 ms) | 243 FPS (4.12 ms) | **1198 FPS (0.83 ms)** |
| YOLO26s @ 256 | 162 FPS (6.17 ms) | 227 FPS (4.40 ms) | **1005 FPS (1.00 ms)** |
| YOLO26s @ 320 | 126 FPS (7.97 ms) | 177 FPS (5.65 ms) | **789 FPS (1.27 ms)** |

> 结论：batch=1 的小模型，瓶颈落在 kernel launch 开销上；CUDA Graph 把每帧合并为一次 launch，是收益最大的一项（推理 0.63–1.11 ms）。FP16 权重转换把体积减半，吞吐变化在 ±2% 以内。<br>Takeaway: batch-1 small models bottleneck on kernel-launch overhead; merging each frame into a single graph launch is the biggest win (0.63–1.11 ms inference). FP16 conversion halves the file size and moves throughput by ±2%.

## 目标追踪（TargetTracker）· `deadeye/core/target_tracker.py` / Target Tracking

- **帧间匹配**：匈牙利算法（`scipy.linear_sum_assignment`）求解**向量化 IoU 代价矩阵**（尺度无关，角点曼哈顿距离的替代），叠加**类别门控 + IoU 门控**（跨类别/重叠不足不参与关联）；匹配基于 **KF 预测框**进行，运动目标的关联因此更稳。
- **卡尔曼滤波**：每目标一个 8 维等速 KF（中心 + 尺寸 + 速度，numpy 实现，`BoxKalmanFilter`）。它平滑观测噪声，在漏检期间推进目标位置，**速度状态**则直接交给执行层做前馈。
- **目标生命周期**（UI「追踪参数」面板可调）：`漏检保留帧数`（默认 5，期间 ID 不变）、`确认命中帧数`（默认 1，抑制单帧误检）、`匹配 IoU 门控`（默认 0.3）。三者共同消除了单帧漏检造成的 ID 跳变。
- **开销**：关联为微秒级（IoU 矩阵向量化 + 线性分配），远低于单帧检测耗时。

- **Association**: Hungarian assignment (`scipy.linear_sum_assignment`) over a **vectorized IoU cost matrix** (scale-invariant, replacing corner Manhattan distance) with **class + IoU gating**; association runs on **KF-predicted boxes**, which keeps moving targets matched steadily.
- **Kalman filter**: one 8-state constant-velocity KF per target (`BoxKalmanFilter`, numpy): it smooths noisy boxes, advances targets through misses, and exposes **target velocity** to the execution layer.
- **Lifecycle** (tunable in the UI Tracking panel): missed-frame retention (default 5, stable IDs), min-hits confirmation (default 1), match IoU gate (default 0.3). Together they eliminate single-miss ID churn.
- **Cost**: microsecond-scale association, negligible against per-frame detection.

## 执行模块（ExecutionModule）· `deadeye/execution_module` / Execution Module

统一接口 `update_targets(frame, target_list)`；内置辅助瞄准（`DeadEyeAutoAimingModule`）与标注视频写出（`VideoWriterOutput`：画框 + ID，默认输出 `output/annotated.mp4`）。

### 辅助瞄准控制律 / Aim control law

```
e'      = shape(e)                          # 死区（默认 1.5px，可设 0.5px 亚像素）
                                              atan 模式: B·atan(A·(2|e|−dz))：小误差近似线性，大误差平滑封顶
Kp_eff  = Kp · (1 + schedule · |v_target|)   # 连续增益调度（v 来自追踪层 KF）
out     = Kp_eff·e' + Ki·Σe' + Kd_axis·Δe' + vff·v_target
          → 积分回弹 anti-windup → max_movement 安全限幅 → 输出限速（px/帧）
```

| 参数 Parameter | 默认 Default | 说明 Description |
|---|---|---|
| Kp / Ki | 0.3 / 5.0 每秒 | Ki 按时间归一化，积分另设输出限幅（默认 30px），瞄准环频率从 60 到 300Hz 变化时手感保持一致<br>Ki normalized by time with a clamped integral output, so feel stays consistent from 60 to 300Hz loops |
| Kd（X）/ Kd（Y） | 0 / 0 | 微分默认双轴关闭：高频检测抖动除以毫秒级 dt 会被放大成随机输出<br>both derivatives off by default; per-pixel detection noise divided by a millisecond dt becomes random output |
| 死区 Deadband | 1.5 px | 消除锁定后的高频微颤；设 0.5px 即亚像素死区<br>kills the high-frequency tremor when locked; 0.5px gives a sub-pixel deadband |
| 误差整形 Shaping | linear（可选 atan） | atan 软饱和：斜率 `atan_slope`、上限 `atan_ceiling`，甩枪平滑封顶无折角<br>atan soft saturation with slope `atan_slope` and cap `atan_ceiling`; flicks saturate smoothly |
| 输出限速 Slew cap | 15 px/帧 | 输出即 px/帧（速度本身），按轴限幅构成甩枪速度天花板<br>the output is px/frame (speed itself); the per-axis cap forms the flick-speed ceiling |
| 输出平滑 Smoothing | 0.6 | 输出端 EMA：恒定跟踪全透传，只衰减 ±1px 级的符号交替噪声，消除锁定后的微颤（0 关闭）<br>output-side EMA: transparent for constant tracking, attenuates alternating +-1px noise that causes lock jitter (0 disables) |
| 速度前馈 / 增益调度 Feed-forward / scheduling | 0 / 0 | KF 速度补偿运动相位滞后；Kp 随目标速度连续放大（无切换抖振）<br>Kalman velocity compensates motion phase lag; Kp scales continuously with target speed |

Unified `update_targets(frame, target_list)` interface; ships the aim assist (`DeadEyeAutoAimingModule`) and the annotated-video writer (`VideoWriterOutput`, default `output/annotated.mp4`).

The aim control law combines a deadband (terminal-jitter kill, sub-pixel capable), optional atan soft-saturation shaping (near-linear for small errors, smoothly capped for flicks), a dual-axis PID with the Y derivative off by default, continuous velocity-based gain scheduling, Kalman-velocity feed-forward, integral-rebound anti-windup and a px/frame output slew cap. Defaults mirror battle-tested tuning (see table above: Kp=Ki=0.3 integral-dominant, Kd=0.01/0, deadzone 1.5px, slew cap 15px/frame). All parameters are auto-generated UI fields.

## CUDA 部署与工具链 / CUDA Deployment & Toolchain

Windows 下无需系统级 CUDA 配置：`deadeye/detection_module/cuda_env.py` 启动时自动注册 torch/lib 与 pip 安装 NVIDIA 包中的 CUDA DLL（CUDA 13 运行库由 torch 捆绑提供）。

```bash
python tools/export_yolo.py weights/model.pt --format onnx --imgsz 320   # pt -> ONNX
python tools/convert_fp16.py weights/                                    # FP16 混合精度转换（体积减半，带加载自检）
python tools/build_engines.py weights/*.onnx                             # 构建 TRT 引擎（timing cache + 元数据 sidecar）
python tools/benchmark.py weights/ --engine --cpu                        # 后端基准对比
python -m pytest tests/ -q                                               # 单元测试（53 项）
```

- **引擎缓存**：sidecar JSON 记录类别名 / imgsz / end2end / 精度 / TRT 版本与 ONNX mtime，加载引擎即得全部解码参数，过期自动重建；TRT 11 精度跟随模型，FP16 引擎自动优先 `.fp16.onnx` 构建。
- **引擎不入库**（`.engine` 已 ignore）：引擎绑定 GPU 架构与 TRT 版本，克隆后用 `build_engines.py` 本机重建；自备模型放 `weights/`（默认路径锚定仓库根，`deadeye/paths.py`）。

No system-level CUDA setup needed on Windows: `cuda_env.py` auto-registers CUDA DLLs from torch/lib and pip-installed NVIDIA packages at startup. Engines are cached with a metadata sidecar (staleness-aware rebuilds); FP16 engines build from the `.fp16.onnx` source automatically. Engines are tied to your GPU and TRT version, so rebuild locally and place your own models under `weights/` (default paths anchored to the repo root via `deadeye/paths.py`).

## 使用指南 / Usage Guide

1. 执行 `main.py` 启动图形界面：「启动配置」选择检测 / 输入 / 执行模块并填写参数（模型路径、置信度 / IoU、推理设备、追踪参数、控制律参数等在各模块参数区）；「运行控制」查看状态与 FPS，开始运行 / 暂停 / 继续。
2. 默认快捷键：`P` 暂停/继续；`O` 退出；`鼠标左键` 按住开启自动瞄准。
3. 环境安装见 `requirements.txt`（宽松约束）或 `requirements-freeze.txt`（精确复现基准环境：Python 3.12 + torch 2.11 cu130 + TRT 11.4 + onnxruntime-gpu 1.31）。

1. Run `main.py` to open the UI: pick detector / input / execution modules and their parameters (model path, conf/IoU, device, tracking and control-law settings); Run control shows status and FPS.
2. Default hotkeys: `P` pause/resume, `O` exit, hold left mouse button to enable auto-aim.
3. Install via `requirements.txt` (loose) or `requirements-freeze.txt` (exact reproduction of the benchmark environment).

### 效果演示 / Demonstration

![辅助瞄准效果演示 / Aiming Assist Effect Demonstration](./res/effect_test.gif)

带标注的示例输出见 `output/annotated.mp4`（检测 + 追踪 ID + 画框的全流水线产物）。See `output/annotated.mp4` for a sample annotated output of the full pipeline.

## 常见问题 / FAQ

1. TensorRT / CUDA 类

   `FileNotFoundError: Could not find: nvinfer.dll. Is it on your PATH?` / `...cublasLt64_13.dll / cudnn64_9.dll which is missing`

   现代栈（TensorRT 10/11 + onnxruntime-gpu 1.2x）不需要手动配置系统 PATH：pip 安装 `tensorrt cuda-python onnxruntime-gpu` 即可（CUDA 13 DLL 由 torch 捆绑），程序启动时 `cuda_env.py` 自动注册 DLL 搜索路径。若仍缺失，确认依赖装在**运行所用环境**内；旧版 TensorRT 8.6 与 RTX 50 系显卡不兼容。精确复现见 `requirements-freeze.txt`。

   The modern stack needs no manual PATH setup: `pip install tensorrt cuda-python onnxruntime-gpu` and the DLL bootstrap handles the rest; make sure packages live in the environment you run with. Legacy TensorRT 8.6 is incompatible with RTX 50-series GPUs.

2. 截屏后端

   旧版 `dxcam` 的 `AttributeError: module 'dxcam' has no attribute 'output_res'` 系其停止维护所致，已移除，改为 `windows-capture`（默认）→ `bettercam` → `mss` 自动降级链，UI 输入模块参数中可指定后端。

   - `windows-capture` 需 Windows 10 1903+；Windows 10 上系统强制显示捕获黄边框（Win11 可 `draw_border=False` 去除）。
   - bettercam（DXGI）要求显示缩放 100%，否则自动切换 `mss`。
   - DXGI/WGC 只在画面变化时产帧，静止画面 `get_image()` 返回 `None` 属正常行为。

   The legacy `dxcam` error came from the package being unmaintained; it was removed in favor of the `windows-capture` → `bettercam` → `mss` fallback chain (selectable in the UI). WGC requires Win10 1903+ (yellow border on Win10); bettercam requires 100% display scaling; DXGI/WGC produce frames only on content changes, so `None` on a static screen is expected.

## 注意事项 / Precautions

- **注1**：早期版本由作者人工设计架构并编码实现，后因代码损坏丢失，现版本基于 AI 辅助重构后开源；各模块原理与实测指标见上文模块章节。
- **注2**：本项目是一个通用的目标检测追踪框架，禁止用于违法用途。严禁将本项目用于游戏作弊、盈利等任何可能涉嫌违法的用途。本程序不提供对鼠标、键盘等输入设备的任何直接控制或互动的代码。其他使用者需要在严格遵守AGPL-3.0许可协议的前提下，合法合规地借鉴本仓库的源代码，不得进行任何违法行为。如有任何违法行为发生，与本项目作者无关，作者保留依法追究相关责任人法律责任的权利。

- **Note 1**: the original architecture was designed and hand-coded by the author; after that source was damaged and lost, the current version was rebuilt with AI assistance and open-sourced. Per-module principles and measured metrics are documented in the module sections above.
- **Note 2**: this project is a general-purpose object detection and tracking framework; any illegal use is strictly prohibited, including game cheating and profiteering. No code for direct control of input devices (mouse/keyboard) is provided. Others must comply with the AGPL-3.0 license; any misuse is the sole responsibility of the user.

## 使用须知 / Usage Notice

欢迎您使用本仓库中的文件。在使用本仓库中的任何文件之前，请仔细阅读以下使用须知：

1. **版权声明**：本仓库中的原创性文件所有权利均归仓库所有者所有，未经许可不得用于商业用途。
2. **引用与借鉴**：如果您在自己的项目中借鉴、引用或使用了本仓库中的任何文件，请务必在您的项目文档（如README文件）中明确注明出处，并提供指向本仓库的链接。建议的引用格式如下：

   本项目中使用了 [DeadEyeAutoAimingSystem](https://github.com/DEVILENMO/DeadEye-Auto-Aiming-System) 仓库中的部分文件，特此表示感谢。

3. **再次分发**：如果您需要再次分发本仓库中的文件，请确保在分发时包含本使用须知，并明确注明原始仓库的链接。
4. **问题反馈**：欢迎通过 Issue 或 Pull Request 反馈问题。

Welcome to use the files in this repository. Before doing so, please read this notice carefully:

1. **Copyright**: original files are owned by the repository owner; commercial use is not allowed without permission.
2. **Citation**: if you reference files from this repository, clearly indicate the source in your documentation and link back to this repository.
3. **Redistribution**: include this notice and the original repository link.
4. **Feedback**: issues and pull requests are welcome.

## 参考项目 / Reference Projects

- [YOLOv7](https://github.com/WongKinYiu/yolov7): YOLOv7 的官方实现。
- [YOLOv8](https://github.com/ultralytics/ultralytics): YOLOv8 官方实现（亦为 v11 / v12 / 26 提供训练与导出）。
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) / [YOLO26](https://docs.ultralytics.com/models/yolo26/): 官方文档（v26 为端到端免 NMS 检测头）。
- [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series): TensorRT 部署 YOLO 系列的参考项目。
- [wang2024yolov10](https://github.com/THU-MIG/yolov10): YOLOv10 官方实现。
- [windows-capture](https://github.com/NiiightmareXD/windows-capture): Windows.Graphics.Capture 高速截屏库（本项目默认截屏后端）。
- [BetterCam](https://github.com/RootKit-Org/BetterCam): dxcam 社区维护分支（DXGI 截屏备选后端）。

## 开源许可 / Open Source License

本项目采用 GNU Affero General Public License v3.0 (AGPLv3) 开源许可证。请查看 [LICENSE](LICENSE) 文件以获取完整的许可证文本。

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3). See [LICENSE](LICENSE) for the full text.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVILENMO/DeadEye-Auto-Aiming-System&type=Date)](https://www.star-history.com/#DEVILENMO/DeadEye-Auto-Aiming-System&Date)
