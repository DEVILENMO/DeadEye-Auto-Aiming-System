# coding: utf-8
# cython: language_level=3
"""
ONNX 混合精度转换工具：FP32 ONNX -> FP16 ONNX（内部权重/算子 FP16，IO 保持 FP32）。

TensorRT 11 起移除了 BuilderFlag.FP16（精度严格跟随模型类型），
因此「FP16 引擎」必须由 FP16 ONNX 构建才能获得真正的半精度加速；
ONNX Runtime CUDA EP 同样受益（cuDNN/cuBLAS FP16 kernel）。

keep_io_types=True 保持模型输入输出为 FP32：
- 预处理 blob、后处理解码、IOBinding/引擎绑定的代码无需任何改动；
- 数值敏感算子（Sigmoid/Softmax/ReduceMax 等）由转换器的默认 block list 保留 FP32。

用法：
    python tools/convert_fp16.py weights/apex_v11s_320.onnx
    python tools/convert_fp16.py weights/            # 目录下全部 onnx
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _scrub_stale_type_annotations(model) -> None:
    """
    删除被 block 的算子周边插入 Cast 后残留的过期 value_info 类型标注。

    转换器把 Cast 输出重命名为原张量名，但图里保留的 FP32 时代标注会与
    新的 FP16 实际类型冲突（ONNX Runtime 加载报 Type Error）。标注本身是
    可选信息，删除后由加载器重新推断即可。
    """
    import onnx

    def _cast_to(node) -> int | None:
        for attr in node.attribute:
            if attr.name == 'to':
                return onnx.helper.get_attribute_value(attr)
        return None

    cast16_out = {n.output[0] for n in model.graph.node if n.op_type == 'Cast' and _cast_to(n) == onnx.TensorProto.FLOAT16}
    cast32_out = {n.output[0] for n in model.graph.node if n.op_type == 'Cast' and _cast_to(n) == onnx.TensorProto.FLOAT}

    stale = []
    for vi in model.graph.value_info:
        is_f32 = vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        is_f16 = vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        if (vi.name in cast16_out and is_f32) or (vi.name in cast32_out and is_f16):
            stale.append(vi.name)
    if stale:
        keep = [vi for vi in model.graph.value_info if vi.name not in set(stale)]
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep)
        print(f'  清洗了 {len(stale)} 条过期类型标注: {stale[:3]}{"..." if len(stale) > 3 else ""}')


def _e2e_head_block_nodes(model) -> list[str]:
    """
    定位端到端检测头（含 TopK 的模块），返回其中「非卷积分支」的节点名列表。

    这些节点（Concat/TopK/Gather/Cast 等，混合 int64/float 张量）保持 FP32，
    避免转换后 Concat 出现 fp16/fp32 混入；one2one_cv*/cv* 卷积分支不阻断。
    非 end2end 模型（无 TopK）返回空列表。
    """
    import re

    head_prefixes = set()
    for node in model.graph.node:
        if node.op_type == 'TopK':
            match = re.match(r'(/model\.\d+/)', node.name or '')
            if match:
                head_prefixes.add(match.group(1))
    if not head_prefixes:
        return []

    block: list[str] = []
    for node in model.graph.node:
        name = node.name or ''
        for prefix in head_prefixes:
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if rest.startswith(('one2one', 'cv')):
                continue  # 头内卷积分支持 FP16
            block.append(name)
            break
    return block


def convert_model(onnx_path: str, out_path: str | None = None, force: bool = False) -> str:
    import onnx
    from onnxconverter_common import float16

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)
    if out_path is None:
        stem = onnx_path[:-5] if onnx_path.lower().endswith('.onnx') else onnx_path
        out_path = f'{stem}.fp16.onnx'
    if os.path.isfile(out_path) and not force:
        print(f'[跳过] 已存在 {out_path}（--force 重建）')
        return out_path

    model = onnx.load(onnx_path, load_external_data=True)
    metadata_props = {(p.key, p.value) for p in model.metadata_props}

    # 数值敏感算子（Sigmoid/Softmax/ReduceMax 等）由默认 block list 保留 FP32。
    # 端到端检测头（YOLOv10/26 的 TopK/top-k 机器，混合 int64/float）整体保持 FP32，
    # 其中的卷积分支（one2one_cv*）仍转 FP16。
    op_block_list = list(float16.DEFAULT_OP_BLOCK_LIST)
    if 'Resize' not in op_block_list:
        op_block_list.append('Resize')
    node_block_list = _e2e_head_block_nodes(model)
    try:
        converted = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=op_block_list,
            node_block_list=node_block_list,
        )
        io_dtype = 'fp32'
    except Exception as exc:
        # 端到端头输出路径复杂时（输出 cast 后仍有节点），keep_io_types 不适用，
        # 回退为 IO=FP16（推理后端均已按输入/输出 dtype 自适应，无需改动）
        print(f'  keep_io_types 不可用（{type(exc).__name__}），回退 IO=FP16')
        converted = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            op_block_list=op_block_list,
            node_block_list=node_block_list,
        )
        io_dtype = 'fp16'
    _scrub_stale_type_annotations(converted)

    # onnxconverter-common 会丢弃导出时写入的自定义元数据（类别名/imgsz/end2end），补回
    existing = {p.key for p in converted.metadata_props}
    for key, value in metadata_props - existing:
        prop = converted.metadata_props.add()
        prop.key, prop.value = key, value

    onnx.save(converted, out_path)

    # 转换后自检：能加载且能跑一次推理，坏图当场失败而不是留到部署时
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    dtype = np.float16 if 'float16' in str(inp.type) else np.float32
    session.run(None, {inp.name: np.random.rand(*shape).astype(dtype)})

    src_mb = os.path.getsize(onnx_path) / 1e6
    dst_mb = os.path.getsize(out_path) / 1e6
    print(f'[完成] {os.path.basename(onnx_path)} -> {os.path.basename(out_path)} '
          f'({src_mb:.1f}MB -> {dst_mb:.1f}MB, IO={io_dtype})')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='ONNX FP32 -> FP16 混合精度转换')
    parser.add_argument('inputs', nargs='+', help='ONNX 文件或目录')
    parser.add_argument('--force', action='store_true', help='覆盖已存在的 fp16.onnx')
    args = parser.parse_args()

    files: list[str] = []
    for item in args.inputs:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, '*.onnx'))))
        else:
            files.append(item)
    files = [f for f in files if not f.endswith('.fp16.onnx')]

    failed = []
    for path in files:
        try:
            convert_model(path, force=args.force)
        except Exception as exc:
            print(f'[失败] {path}: {exc}')
            failed.append(path)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
