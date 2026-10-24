# coding: utf-8
# cython: language_level=3
"""
YOLO 模型导出工具（Ultralytics 权重 -> ONNX / TensorRT engine）。

支持 YOLOv8 / YOLOv11 / YOLOv12 / YOLO26 的 .pt 权重。

用法示例：
    python tools/export_yolo.py weights/my_v11.pt --format onnx --imgsz 320 --simplify
    python tools/export_yolo.py weights/my_v26.pt --format onnx          # v26 默认端到端免 NMS
    python tools/export_yolo.py weights/my_v8.pt --format engine --half  # 直接导出 TRT engine

说明：
- ONNX 导出保持 nms=False（TRT/ORT 部署由本项目统一后处理解码）；
  YOLO26 例外——其检测头本身即为端到端设计，导出结果自带 top-k 免 NMS 输出。
- YOLOv5/v7 的老权重建议用原仓库 / TensorRT-For-YOLO-Series 导出 ONNX，
  本项目推理后端可正常加载其 [anchors, 5+nc] 输出布局。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO 导出工具')
    parser.add_argument('weights', help='.pt 权重路径')
    parser.add_argument('--format', default='onnx', choices=['onnx', 'engine'], help='导出格式')
    parser.add_argument('--imgsz', type=int, default=320, help='输入边长（正方形）')
    parser.add_argument('--half', action='store_true', help='FP16（engine / onnx 数值精度）')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset（TRT 兼容性优先取 12/13/17）')
    parser.add_argument('--simplify', action='store_true', default=True, help='onnx-simplifier')
    parser.add_argument('--batch', type=int, default=1, help='静态 batch')
    parser.add_argument('--dynamic', action='store_true', help='动态 batch（默认关闭，部署更稳）')
    parser.add_argument('--workspace', type=int, default=4, help='engine 构建 workspace (GB)')
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)

    # 延迟导入：导出环境才需要 ultralytics + torch
    from ultralytics import YOLO

    model = YOLO(args.weights, task='detect')
    export_kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        batch=args.batch,
        device=0,
    )
    if args.format == 'engine':
        export_kwargs['workspace'] = args.workspace

    print(f'导出: {args.weights} -> {args.format} | {export_kwargs}')
    path = model.export(**export_kwargs)
    print(f'完成: {path}')
    return path


if __name__ == '__main__':
    main()
