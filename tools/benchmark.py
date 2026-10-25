# coding: utf-8
# cython: language_level=3
"""
检测流水线基准测试：对比不同后端（ORT CPU / ORT CUDA / TensorRT）在真实帧上的
预处理、推理、后处理耗时与端到端 FPS。

用法示例：
    python tools/benchmark.py                                      # 扫描 weights/*.onnx
    python tools/benchmark.py weights/test_v11s_320.onnx --frames 200
    python tools/benchmark.py weights/test_v26s_320.onnx --video res/videos/aimlab_test.mp4
    python tools/benchmark.py weights/test_v11s_320.onnx --engine   # 额外测 TRT 引擎（自动构建）
"""
import argparse
import glob
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deadeye import paths  # noqa: E402
from deadeye.detection_module.backends.onnx_backend import OnnxRuntimeBackend  # noqa: E402
from deadeye.detection_module.backends.tensorrt_backend import ensure_engine  # noqa: E402
from deadeye.detection_module.postprocess import decode_detections  # noqa: E402
from deadeye.detection_module.preprocess import Letterboxer  # noqa: E402

DEFAULT_VIDEO = paths.res_path('videos', 'aimlab_test.mp4')
DEFAULT_IMAGE = paths.res_path('images', 'apex_test1.png')


def load_frames(video_path: str, count: int, target_size: int) -> list[np.ndarray]:
    """从测试视频取帧；视频不可用时用测试图 + 合成运动帧兜底。"""
    frames: list[np.ndarray] = []
    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 流水线约定 RGB
        cap.release()
    if not frames and os.path.isfile(DEFAULT_IMAGE):
        base = cv2.cvtColor(cv2.imread(DEFAULT_IMAGE), cv2.COLOR_BGR2RGB)
        base = cv2.resize(base, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        for i in range(count):
            shift = np.uint8(np.abs(np.int32(i) * 7 % 40 - 20))
            frames.append(np.roll(base, shift, axis=1))
    if not frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (target_size, target_size, 3), dtype=np.uint8) for _ in range(count)]
    return frames


def bench_backend(backend, frames_rgb, *, conf, iou, max_dets, warmup=20):
    h, w = backend.meta.input_size
    letterboxer = Letterboxer(h, w)

    for img in frames_rgb[:warmup]:
        blob, info = letterboxer(img)
        outputs = backend.infer(blob)

    pre_ms, infer_ms, post_ms = [], [], []
    n_dets = 0
    for img in frames_rgb:
        t0 = time.perf_counter()
        blob, info = letterboxer(img)
        t1 = time.perf_counter()
        outputs = backend.infer(blob)
        t2 = time.perf_counter()
        dets = decode_detections(
            outputs[0], info, conf_thres=conf, iou_thres=iou, max_dets=max_dets,
            layout='auto', end2end_meta=backend.meta.end2end,
        )
        t3 = time.perf_counter()
        pre_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        post_ms.append((t3 - t2) * 1000)
        n_dets += dets.shape[0]

    total_ms = [a + b + c for a, b, c in zip(pre_ms, infer_ms, post_ms)]
    return {
        'pre': statistics.mean(pre_ms),
        'infer': statistics.mean(infer_ms),
        'post': statistics.mean(post_ms),
        'total': statistics.mean(total_ms),
        'p95': statistics.quantiles(total_ms, n=20)[-1] if len(total_ms) >= 20 else max(total_ms),
        'fps': 1000.0 / statistics.mean(total_ms),
        'dets_per_frame': n_dets / len(frames_rgb),
    }


def main():
    parser = argparse.ArgumentParser(description='YOLO 检测后端基准测试')
    parser.add_argument('models', nargs='*', help='ONNX / engine 路径（缺省扫描 weights/*.onnx）')
    parser.add_argument('--frames', type=int, default=200, help='测试帧数')
    parser.add_argument('--video', default=DEFAULT_VIDEO, help='测试视频')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--engine', action='store_true', help='额外构建并测试 TensorRT 引擎')
    parser.add_argument('--cpu', action='store_true', help='额外测试 ONNX Runtime CPU')
    parser.add_argument('--trt-ep', action='store_true', help='额外测试 ORT TensorrtExecutionProvider')
    args = parser.parse_args()

    models = []
    for item in args.models:
        if os.path.isdir(item):
            models.extend(sorted(glob.glob(os.path.join(item, '*.onnx'))))
        elif any(c in item for c in '*?'):
            models.extend(sorted(glob.glob(item)))
        else:
            models.append(item)
    if not models:
        models = sorted(glob.glob(os.path.join('weights', '*.onnx')))
    if not models:
        raise SystemExit('未找到模型')

    rows = []
    for model_path in models:
        frames = load_frames(args.video, args.frames, 320)
        variants = [('onnx-cuda', lambda: OnnxRuntimeBackend(model_path, device='cuda'))]
        if args.cpu:
            variants.append(('onnx-cpu', lambda: OnnxRuntimeBackend(model_path, device='cpu')))
        if args.trt_ep:
            variants.append(('onnx-trtep', lambda: OnnxRuntimeBackend(model_path, use_tensorrt_ep=True)))
        if args.engine:
            variants.append(('tensorrt', lambda: _make_trt(model_path)))

        for name, factory in variants:
            try:
                backend = factory()
            except Exception as exc:
                print(f'[跳过] {os.path.basename(model_path)} / {name}: {exc}')
                continue
            stats = bench_backend(
                backend, frames, conf=args.conf, iou=args.iou, max_dets=300
            )
            rows.append((os.path.basename(model_path), backend.backend_name, stats))
            backend.on_exit()
            print(f'{os.path.basename(model_path):28s} {backend.backend_name:32s} '
                  f'infer={stats["infer"]:6.2f}ms total={stats["total"]:6.2f}ms '
                  f'fps={stats["fps"]:7.1f} dets={stats["dets_per_frame"]:.1f}')

    print('=' * 96)
    print(f'{"model":28s} {"backend":32s} {"pre":>7s} {"infer":>8s} {"post":>7s} '
          f'{"e2e":>7s} {"p95":>7s} {"fps":>8s} {"det/f":>6s}')
    for model, name, s in rows:
        print(f'{model:28s} {name:32s} {s["pre"]:7.2f} {s["infer"]:8.2f} {s["post"]:7.2f} '
              f'{s["total"]:7.2f} {s["p95"]:7.2f} {s["fps"]:8.1f} {s["dets_per_frame"]:6.1f}')
    print('=' * 96)


def _make_trt(model_path: str):
    from deadeye.detection_module.backends.tensorrt_backend import TensorRTBackend

    if model_path.lower().endswith('.onnx'):
        engine_path = ensure_engine(model_path)
        return TensorRTBackend(engine_path)
    return TensorRTBackend(model_path)


if __name__ == '__main__':
    main()
