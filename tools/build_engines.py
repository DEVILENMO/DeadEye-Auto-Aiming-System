# coding: utf-8
# cython: language_level=3
"""
TensorRT 引擎批量构建工具（ONNX -> .engine，timing cache + 元数据 sidecar）。

TRT 11 起精度跟随模型：FP16 引擎自动优先使用旁侧的 ``<stem>.fp16.onnx``
（由 tools/convert_fp16.py 生成）作为构建源。

用法示例：
    python tools/convert_fp16.py weights/                # 先转 FP16 ONNX（一次性）
    python tools/build_engines.py weights/               # 全部构建（FP16 引擎）
    python tools/build_engines.py weights/apex_v11s_320.onnx --fp32
    python tools/build_engines.py weights/ --force       # 忽略缓存重建
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deadeye.detection_module.backends.tensorrt_backend import (  # noqa: E402
    build_engine_from_onnx,
    default_engine_path,
    engine_is_current,
    ensure_engine,
    resolve_fp16_source,
)


def main():
    parser = argparse.ArgumentParser(description='ONNX -> TensorRT 引擎构建工具')
    parser.add_argument('inputs', nargs='+', help='ONNX 文件或包含 ONNX 的目录')
    parser.add_argument('--fp32', action='store_true', help='构建 FP32 引擎（默认 FP16）')
    parser.add_argument('--workspace', type=float, default=4.0, help='workspace GB')
    parser.add_argument('--force', action='store_true', help='忽略缓存强制重建')
    args = parser.parse_args()

    fp16 = not args.fp32
    onnx_files: list[str] = []
    for item in args.inputs:
        if os.path.isdir(item):
            onnx_files.extend(sorted(glob.glob(os.path.join(item, '*.onnx'))))
        elif os.path.isfile(item):
            onnx_files.append(item)
        else:
            print(f'跳过不存在的路径: {item}')
    onnx_files = [f for f in onnx_files if not f.endswith('.fp16.onnx')]
    if not onnx_files:
        raise SystemExit('没有找到 ONNX 文件')

    built, reused, failed = [], [], []
    for onnx_path in onnx_files:
        engine_path = default_engine_path(onnx_path, fp16)
        source = resolve_fp16_source(onnx_path)
        was_current = os.path.isfile(engine_path) and engine_is_current(engine_path, source)
        try:
            if args.force:
                out = build_engine_from_onnx(
                    source, engine_path, fp16=fp16, workspace_gb=args.workspace
                )
            else:
                out = ensure_engine(
                    onnx_path, engine_path, fp16=fp16, workspace_gb=args.workspace
                )
            tag = '复用' if was_current and not args.force else '完成'
            print(f'[{tag}] {out} (源: {os.path.basename(source)})')
            (reused if tag == '复用' else built).append(out)
        except Exception as exc:
            print(f'[失败] {onnx_path}: {exc}')
            failed.append(onnx_path)

    print('=' * 60)
    print(f'构建 {len(built)} | 复用 {len(reused)} | 失败 {len(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
