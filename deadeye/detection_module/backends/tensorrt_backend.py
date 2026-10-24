# coding: utf-8
# cython: language_level=3
"""
原生 TensorRT 10+ 推理后端（cuda-python + 预分配显存，不经 onnxruntime）。

能力：
- 加载 / 推理：TRT 10 tensor-name API（set_tensor_address + execute_async_v3），
  pinned host 缓冲 + 预分配 device 缓冲，稳态零分配；
- 引擎构建：ONNX -> serialized engine（FP16 可选 + timing cache），
  产物写入 ``<onnx>.fp16.engine`` 并附带元数据 sidecar（.json）；
- 引擎缓存：sidecar 记录 TRT 版本 / ONNX mtime / 精度，过期自动判定。
"""
from __future__ import annotations

import ctypes
import json
import os
import time

import numpy as np

from deadeye.detection_module.backends.base import BaseInferBackend, ModelMeta, parse_names_metadata

ENGINE_EXTS = ('.engine', '.plan', '.trt')

_CTYPE_OF_DTYPE = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float16): ctypes.c_uint16,
    np.dtype(np.int32): ctypes.c_int32,
    np.dtype(np.int64): ctypes.c_int64,
    np.dtype(np.uint8): ctypes.c_uint8,
}


# --------------------------------------------------------------------------
# cuda-python 驱动层封装（优先新式 cuda.bindings，回退旧式 from cuda import cuda）
# --------------------------------------------------------------------------
class _CudaDriver:
    """统一返回值封装：cuda-python 中无返回值的函数只返回 CUresult，
    有返回值的函数返回 (CUresult, value)；这里统一为「出错抛异常，成功返回 value」。"""

    def __init__(self):
        try:
            from cuda.bindings import driver as cu
        except ImportError:
            try:
                from cuda import cuda as cu
            except ImportError as exc:
                raise RuntimeError('未安装 cuda-python，请执行: pip install cuda-python') from exc
        self._cu = cu

    def _call(self, fn, *args):
        result = fn(*args)
        # cuda-python：有返回值的函数 -> (err, value)；void 函数 -> (err,) 或裸 err
        if isinstance(result, tuple):
            if len(result) == 2:
                err, value = result
            else:
                err, value = result[0], None
        else:
            err, value = result, None
        if int(err) != 0:
            raise RuntimeError(f'CUDA 驱动调用失败 {getattr(fn, "__name__", fn)}: 0x{int(err):X}')
        return value

    def init(self):
        return self._call(self._cu.cuInit, 0)

    def primary_ctx_retain(self, device_id: int):
        return self._call(self._cu.cuDevicePrimaryCtxRetain, device_id)

    def ctx_push(self, ctx):
        return self._call(self._cu.cuCtxPushCurrent, ctx)

    def ctx_pop(self):
        return self._call(self._cu.cuCtxPopCurrent)

    def ctx_synchronize(self):
        return self._call(self._cu.cuCtxSynchronize)

    def mem_alloc(self, nbytes: int):
        return self._call(self._cu.cuMemAlloc, int(nbytes))

    def mem_free(self, ptr):
        return self._call(self._cu.cuMemFree, ptr)

    def host_alloc(self, nbytes: int):
        return self._call(self._cu.cuMemHostAlloc, int(nbytes), 0)

    def host_free(self, ptr):
        return self._call(self._cu.cuMemFreeHost, ptr)

    def memcpy_htod_async(self, dst_ptr: int, src_addr: int, nbytes: int, stream=0):
        return self._call(self._cu.cuMemcpyHtoDAsync, dst_ptr, src_addr, int(nbytes), stream)

    def memcpy_dtoh_async(self, dst_addr: int, src_ptr: int, nbytes: int, stream=0):
        return self._call(self._cu.cuMemcpyDtoHAsync, dst_addr, src_ptr, int(nbytes), stream)

    def stream_create(self):
        stream = self._call(self._cu.cuStreamCreate, 0)
        return int(stream)

    def stream_synchronize(self, stream):
        return self._call(self._cu.cuStreamSynchronize, stream)

    def stream_destroy(self, stream):
        return self._call(self._cu.cuStreamDestroy, stream)

    # ---------------- CUDA Graph ----------------

    def stream_begin_capture(self, stream, mode=None):
        if mode is None:
            mode = getattr(self._cu, 'CU_STREAM_CAPTURE_MODE_THREAD_LOCAL', 0)
        return self._call(self._cu.cuStreamBeginCapture, stream, mode)

    def stream_end_capture(self, stream):
        return self._call(self._cu.cuStreamEndCapture, stream)

    def graph_instantiate(self, graph):
        try:
            # CUDA 12/13: cuGraphInstantiate(graph, flags)
            return self._call(self._cu.cuGraphInstantiate, graph, 0)
        except (TypeError, AttributeError):
            # 旧签名: cuGraphInstantiate(graph)
            return self._call(self._cu.cuGraphInstantiate, graph)

    def graph_launch(self, graph_exec, stream):
        return self._call(self._cu.cuGraphLaunch, graph_exec, stream)

    def graph_exec_destroy(self, graph_exec):
        return self._call(self._cu.cuGraphExecDestroy, graph_exec)

    @staticmethod
    def pinned_view(ptr: int, dtype, count: int) -> np.ndarray:
        base = _CTYPE_OF_DTYPE.get(np.dtype(dtype))
        if base is None:
            raise TypeError(f'不支持的引擎张量类型: {dtype}')
        buffer = (base * max(count, 1)).from_address(ptr)
        return np.ctypeslib.as_array(buffer).view(np.dtype(dtype))


# --------------------------------------------------------------------------
# 元数据 sidecar
# --------------------------------------------------------------------------
def engine_meta_path(engine_path: str) -> str:
    return engine_path + '.json'


def write_engine_meta(engine_path: str, meta: dict) -> None:
    with open(engine_meta_path(engine_path), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_engine_meta(engine_path: str) -> dict | None:
    path = engine_meta_path(engine_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_onnx_metadata(model_path: str) -> dict:
    """用 onnxruntime 读取 Ultralytics 写入 ONNX 的元数据（构建引擎时写入 sidecar）。"""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    custom = session.get_modelmeta().custom_metadata_map or {}
    meta: dict = {'description': custom.get('description', '')}
    if 'names' in custom:
        names = parse_names_metadata(custom['names'])
        if names:
            meta['names'] = names
    if 'imgsz' in custom:
        try:
            meta['imgsz'] = json.loads(custom['imgsz'])
        except (ValueError, TypeError):
            pass
    if 'end2end' in custom:
        meta['end2end'] = str(custom['end2end']).lower() == 'true'
    return meta


def engine_is_current(engine_path: str, onnx_path: str | None = None) -> bool:
    """sidecar 中的 TRT 版本与当前一致、且 ONNX 未更新时，认为引擎可直接复用。"""
    import tensorrt as trt

    meta = load_engine_meta(engine_path)
    if meta is None:
        return False
    if meta.get('trt_version') != trt.__version__:
        return False
    if onnx_path is not None:
        if not os.path.isfile(onnx_path):
            return False
        if meta.get('onnx_mtime') != os.path.getmtime(onnx_path):
            return False
    return True


# --------------------------------------------------------------------------
# 引擎构建
# --------------------------------------------------------------------------
def default_engine_path(onnx_path: str, fp16: bool) -> str:
    stem = onnx_path[:-5] if onnx_path.lower().endswith('.onnx') else onnx_path
    if stem.endswith('.fp16') or stem.endswith('.fp32'):
        stem = stem[:-5]
    return f'{stem}.fp16.engine' if fp16 else f'{stem}.fp32.engine'


def resolve_fp16_source(onnx_path: str) -> str:
    """
    TRT 11 起精度严格跟随模型：FP16 引擎应从 FP16 ONNX 构建。
    若旁侧存在 ``<stem>.fp16.onnx``（tools/convert_fp16.py 产物）则优先使用。
    """
    if onnx_path.lower().endswith('.fp16.onnx'):
        return onnx_path
    stem = onnx_path[:-5] if onnx_path.lower().endswith('.onnx') else onnx_path
    fp16_onnx = f'{stem}.fp16.onnx'
    return fp16_onnx if os.path.isfile(fp16_onnx) else onnx_path


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str | None = None,
    *,
    fp16: bool = True,
    workspace_gb: float = 4.0,
    verbose: bool = False,
) -> str:
    """ONNX -> TensorRT serialized engine，写入 sidecar 元数据，返回引擎路径。"""
    import tensorrt as trt

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f'ONNX 模型不存在: {onnx_path}')
    if engine_path is None:
        engine_path = default_engine_path(onnx_path, fp16)

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10 默认 explicit batch
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError('ONNX 解析失败:\n' + '\n'.join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    # TRT 11 起移除 BuilderFlag.FP16（精度跟随模型类型，构建器自动选择最快实现）；
    # TRT 10 及以下仍通过 flag 开启 FP16。
    if fp16 and hasattr(trt.BuilderFlag, 'FP16'):
        config.set_flag(trt.BuilderFlag.FP16)

    # timing cache：相同网络重复构建时显著加速
    cache_path = engine_path + '.timecache'
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                timing_cache = config.create_timing_cache(f.read())
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
        except Exception:
            pass

    print(f'构建 TensorRT 引擎: {os.path.basename(onnx_path)} -> {os.path.basename(engine_path)} '
          f'(fp16={fp16}, workspace={workspace_gb}GB)，首次构建需要数分钟…')
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT 引擎构建失败（查看上方日志）')

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    try:
        new_cache = config.get_timing_cache()
        if new_cache is not None:
            with open(cache_path, 'wb') as f:
                f.write(new_cache.serialize())
    except Exception:
        pass

    meta = read_onnx_metadata(onnx_path)
    meta.update({
        'onnx_path': os.path.abspath(onnx_path),
        'onnx_mtime': os.path.getmtime(onnx_path),
        # TRT 11 精度跟随模型：记录真实来源（fp16.onnx -> 引擎即 FP16）
        'precision': 'fp16' if (fp16 or onnx_path.lower().endswith('.fp16.onnx')) else 'fp32',
        'trt_version': trt.__version__,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })
    write_engine_meta(engine_path, meta)
    print(f'引擎构建完成，耗时 {time.time() - t0:.1f}s -> {engine_path}')
    return engine_path


def ensure_engine(
    onnx_path: str,
    engine_path: str | None = None,
    *,
    fp16: bool = True,
    workspace_gb: float = 4.0,
    rebuild_if_stale: bool = True,
    verbose: bool = False,
) -> str:
    """返回可用的引擎路径：命中缓存则直接复用，否则构建。"""
    if engine_path is None:
        engine_path = default_engine_path(onnx_path, fp16)
    if fp16:
        onnx_path = resolve_fp16_source(onnx_path)

    if os.path.isfile(engine_path) and engine_is_current(engine_path, onnx_path):
        return engine_path
    if os.path.isfile(engine_path) and not rebuild_if_stale:
        raise RuntimeError(f'引擎已过期: {engine_path}（删除它或允许重建）')
    return build_engine_from_onnx(
        onnx_path, engine_path, fp16=fp16, workspace_gb=workspace_gb, verbose=verbose
    )


# --------------------------------------------------------------------------
# 推理后端
# --------------------------------------------------------------------------
class TensorRTBackend(BaseInferBackend):
    def __init__(self, engine_path: str, *, device_id: int = 0, use_cuda_graph: bool = True):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f'TensorRT 引擎不存在: {engine_path}')
        if os.path.splitext(engine_path)[1].lower() not in ENGINE_EXTS:
            raise ValueError(f'TensorRTBackend 支持 {ENGINE_EXTS}，收到: {engine_path}')

        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('未安装 tensorrt，请执行: pip install tensorrt') from exc

        self._precision = 'unknown'
        self._use_cuda_graph = bool(use_cuda_graph)
        self._graph_exec = None
        self._cuda = _CudaDriver()
        self._cuda.init()
        self._ctx = self._cuda.primary_ctx_retain(device_id)

        self._load_engine(engine_path)
        self._meta_from_sidecar(engine_path)
        self._try_capture_graph()

    # -------------------------------------------------- setup

    def _load_engine(self, engine_path: str) -> None:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f'引擎反序列化失败: {engine_path}')
        self._trt = trt
        self._context = self._engine.create_execution_context()

        self._tensors: dict[str, dict] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            shape = tuple(self._engine.get_tensor_shape(name))
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self._tensors[name] = {
                'dtype': dtype,
                'shape': shape,
                'is_input': is_input,
                'dynamic': any(not isinstance(d, int) or d < 0 for d in shape),
                'host': None,
                'host_ptr': 0,
                'device': None,
                'allocated': 0,
            }

        self.input_name = next(n for n, t in self._tensors.items() if t['is_input'])
        self._output_names = [n for n, t in self._tensors.items() if not t['is_input']]
        self._input_shape = self._tensors[self.input_name]['shape']
        self._alloc_buffers()

    def _alloc_buffers(self) -> None:
        self._cuda.ctx_push(self._ctx)
        try:
            # 非默认流：TRT 在默认流上会额外做 cudaStreamSynchronize 影响性能
            self._stream = self._cuda.stream_create()
            for name, tensor in self._tensors.items():
                if tensor['dynamic']:
                    continue  # 动态形状张量在首次 infer 时按实际 shape 分配
                count = int(np.prod(tensor['shape']))
                nbytes = max(count, 1) * tensor['dtype'].itemsize
                host_ptr = self._cuda.host_alloc(nbytes)
                dev_ptr = self._cuda.mem_alloc(nbytes)
                tensor['host'] = self._cuda.pinned_view(host_ptr, tensor['dtype'], count)
                tensor['host_ptr'] = int(host_ptr)
                tensor['device'] = dev_ptr
                tensor['allocated'] = nbytes
                self._context.set_tensor_address(name, int(dev_ptr))
        finally:
            self._cuda.ctx_pop()

    def _meta_from_sidecar(self, engine_path: str) -> None:
        sidecar = load_engine_meta(engine_path) or {}
        self._precision = str(sidecar.get('precision', 'unknown'))

        input_h, input_w = 640, 640
        if len(self._input_shape) == 4 and isinstance(self._input_shape[2], int):
            input_h, input_w = self._input_shape[2], self._input_shape[3]
        imgsz = sidecar.get('imgsz')
        if imgsz and len(imgsz) >= 2:
            input_h, input_w = int(imgsz[-2]), int(imgsz[-1])

        self.meta = ModelMeta(
            input_size=(int(input_h), int(input_w)),
            names={int(k): str(v) for k, v in (sidecar.get('names') or {}).items()},
            end2end=sidecar.get('end2end'),
            description=sidecar.get('description', ''),
        )

    # -------------------------------------------------- infer

    def _enqueue(self, stream) -> list[np.ndarray]:
        """在指定 stream 上入队完整推理链：H2D -> execute_async_v3 -> D2H。"""
        cuda = self._cuda
        in_tensor = self._tensors[self.input_name]
        cuda.memcpy_htod_async(
            int(in_tensor['device']), in_tensor['host_ptr'], in_tensor['allocated'], stream)
        if not self._context.execute_async_v3(stream):
            raise RuntimeError('TensorRT execute_async_v3 执行失败')
        outputs = []
        for name in self._output_names:
            tensor = self._tensors[name]
            if tensor['dynamic']:
                shape = tuple(self._context.get_tensor_shape(name))
                self._realloc_dynamic(name, shape)
                out = tensor['host'].reshape(shape)
            else:
                out = tensor['host'].reshape(tensor['shape'])
            cuda.memcpy_dtoh_async(
                tensor['host_ptr'], int(tensor['device']), tensor['allocated'], stream)
            outputs.append(out)
        return outputs

    def _try_capture_graph(self) -> None:
        """
        CUDA Graph：把 H2D -> 推理 -> D2H 整链捕获为单张图，每帧一次 launch。
        batch=1 的小模型是 kernel launch 开销受限（数百个 kernel），图捕获收益显著；
        静态形状专用，捕获失败自动回退逐 kernel 路径。
        """
        if not self._use_cuda_graph:
            return
        if any(t['dynamic'] for t in self._tensors.values()):
            print('CUDA Graph 跳过：引擎含动态形状张量。')
            return
        cuda = self._cuda
        cuda.ctx_push(self._ctx)
        try:
            self._enqueue(self._stream)  # 预热：完成 TRT tactic 初始化
            cuda.stream_synchronize(self._stream)

            cuda.stream_begin_capture(self._stream)
            try:
                self._enqueue(self._stream)
                graph = cuda.stream_end_capture(self._stream)
            except Exception:
                try:
                    cuda.stream_end_capture(self._stream)
                except Exception:
                    pass
                raise
            self._graph_exec = cuda.graph_instantiate(graph)
            # 首次 launch 验证可用性
            cuda.graph_launch(self._graph_exec, self._stream)
            cuda.stream_synchronize(self._stream)
            print('CUDA Graph 捕获成功：每帧单次 graph launch。')
        except Exception as exc:
            self._graph_exec = None
            print(f'CUDA Graph 捕获失败，回退逐 kernel 执行: {exc}')
        finally:
            cuda.ctx_pop()

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        cuda = self._cuda
        stream = self._stream
        cuda.ctx_push(self._ctx)
        try:
            in_tensor = self._tensors[self.input_name]
            blob = np.ascontiguousarray(blob, dtype=in_tensor['dtype'])
            if in_tensor['dynamic'] and tuple(blob.shape) != tuple(in_tensor['shape']):
                self._realloc_dynamic(self.input_name, tuple(blob.shape))
            self._context.set_input_shape(self.input_name, tuple(blob.shape))
            np.copyto(in_tensor['host'].reshape(blob.shape), blob)

            if self._graph_exec is not None:
                cuda.graph_launch(self._graph_exec, stream)
                cuda.stream_synchronize(stream)
                outputs = [
                    self._tensors[name]['host'].reshape(self._tensors[name]['shape'])
                    for name in self._output_names
                ]
            else:
                outputs = self._enqueue(stream)
                cuda.stream_synchronize(stream)

            return [out.astype(np.float32, copy=False) for out in outputs]
        finally:
            cuda.ctx_pop()

    def _realloc_dynamic(self, name: str, shape) -> None:
        tensor = self._tensors[name]
        count = max(int(np.prod(shape)), 1)
        nbytes = count * tensor['dtype'].itemsize
        if nbytes <= tensor['allocated'] and tensor['host'] is not None and tensor['host'].size >= count:
            return
        if tensor.get('device'):
            self._cuda.mem_free(tensor['device'])
        if tensor.get('host_ptr'):
            self._cuda.host_free(tensor['host_ptr'])
        host_ptr = self._cuda.host_alloc(nbytes)
        dev_ptr = self._cuda.mem_alloc(nbytes)
        tensor['host'] = self._cuda.pinned_view(host_ptr, tensor['dtype'], count)
        tensor['host_ptr'] = int(host_ptr)
        tensor['device'] = dev_ptr
        tensor['allocated'] = nbytes
        self._context.set_tensor_address(name, int(dev_ptr))

    @property
    def backend_name(self) -> str:
        return f'TensorRTBackend({self._precision})'

    def on_exit(self) -> None:
        print('Destroying TensorRT backend...')
        self._cuda.ctx_push(self._ctx)
        try:
            if self._graph_exec is not None:
                try:
                    self._cuda.graph_exec_destroy(self._graph_exec)
                except Exception:
                    pass
                self._graph_exec = None
            for tensor in self._tensors.values():
                if tensor.get('device'):
                    self._cuda.mem_free(tensor['device'])
                if tensor.get('host_ptr'):
                    self._cuda.host_free(tensor['host_ptr'])
                tensor['device'] = None
                tensor['host_ptr'] = 0
            if getattr(self, '_stream', None):
                self._cuda.stream_destroy(self._stream)
                self._stream = None
        finally:
            self._cuda.ctx_pop()
        self._context = None
        self._engine = None
