# coding: utf-8
# cython: language_level=3
"""
ONNX Runtime 推理后端（推荐入口）。

- 设备自适应：CUDAExecutionProvider 可用时走 GPU，否则回退 CPU；
  也可通过 TensorrtExecutionProvider 直接使用 TRT（需单独安装 tensorrt）。
- IOBinding 稳态零分配：输入 CPU 缓冲直接绑定（一次 H2D 拷贝），
  输出预分配到固定 host 缓冲（一次 D2H 拷贝），推理循环中不产生新的内存分配。
- 自动读取 Ultralytics 导出时写入 ONNX 的元数据（类别名 / imgsz / end2end）。
"""
from __future__ import annotations

import os

import numpy as np

from deadeye.detection_module.backends.base import BaseInferBackend, ModelMeta, parse_names_metadata
from deadeye.detection_module.cuda_env import bootstrap_nvidia_dlls


class OnnxRuntimeBackend(BaseInferBackend):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = 'auto',
        use_tensorrt_ep: bool = False,
        enable_profiling: bool = False,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('未安装 onnxruntime，请执行: pip install onnxruntime-gpu') from exc

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'ONNX 模型不存在: {model_path}')
        if not model_path.lower().endswith('.onnx'):
            raise ValueError(f'OnnxRuntimeBackend 仅支持 .onnx，收到: {model_path}')

        self._ort = ort
        self.device = self._resolve_device(device, ort, use_tensorrt_ep)

        bootstrap_nvidia_dlls()

        providers: list[str] = []
        provider_options: list[dict] = []
        if self.device == 'cuda':
            providers.append('CUDAExecutionProvider')
            provider_options.append({'cudnn_conv_algo_search': 'HEURISTIC'})
        elif self.device == 'tensorrt':
            providers.append('TensorrtExecutionProvider')
            provider_options.append({'trt_fp16_enable': '1'})
        providers.append('CPUExecutionProvider')
        provider_options.append({})  # provider_options 必须是与 providers 对齐的列表

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3
        if enable_profiling:
            session_options.enable_profiling = True

        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options,
            )
        except Exception as exc:
            raise RuntimeError(f'创建 ONNX Runtime 会话失败（device={self.device}）: {exc}') from exc

        if self.device != 'cpu' and 'CUDAExecutionProvider' not in self.session.get_providers() \
                and 'TensorrtExecutionProvider' not in self.session.get_providers():
            print(f'警告：请求 device={self.device} 但 GPU EP 未激活，已回退 CPU '
                  f'（检查 CUDA/cuDNN DLL 是否可用，见 deadeye/detection_module/cuda_env.py）。')

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self._input_dtype = self._numpy_dtype(self.session.get_inputs()[0].type)

        self._output_names = [o.name for o in self.session.get_outputs()]
        self._pre_alloc_outputs: dict[str, np.ndarray] | None = None
        self._binding = None
        self._setup_iobinding()

        self.meta = self._read_metadata(model_path)

    # -------------------------------------------------- setup

    @staticmethod
    def _resolve_device(device: str, ort, use_tensorrt_ep: bool) -> str:
        available = ort.get_available_providers()
        if use_tensorrt_ep and 'TensorrtExecutionProvider' in available:
            return 'tensorrt'
        if device == 'cpu':
            return 'cpu'
        if 'CUDAExecutionProvider' in available:
            return 'cuda'
        return 'cpu'

    def _setup_iobinding(self) -> None:
        """静态形状时启用 IOBinding + 输出预分配；含动态维度则回退 session.run。"""
        def _is_static(shape) -> bool:
            return all(isinstance(dim, int) and dim > 0 for dim in shape)

        if not _is_static(self.input_shape):
            return
        for out in self.session.get_outputs():
            if not _is_static(out.shape):
                return

        binding = self.session.io_binding()
        self._pre_alloc_outputs: dict[str, np.ndarray] = {
            out.name: np.empty(tuple(out.shape), dtype=self._numpy_dtype(out.type))
            for out in self.session.get_outputs()
        }
        for name, arr in self._pre_alloc_outputs.items():
            binding.bind_output(
                name=name,
                device_type='cpu',
                device_id=0,
                element_type=arr.dtype.type,
                shape=tuple(arr.shape),
                buffer_ptr=arr.ctypes.data,
            )
        self._binding = binding

    @staticmethod
    def _numpy_dtype(onnx_type: str):
        return np.float16 if 'float16' in str(onnx_type) else np.float32

    def _read_metadata(self, model_path: str) -> ModelMeta:
        meta = self.session.get_modelmeta()
        custom = meta.custom_metadata_map or {}

        input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

        names: dict[int, str] = {}
        if 'names' in custom:
            names = parse_names_metadata(custom['names'])

        end2end: bool | None = None
        if 'end2end' in custom:
            end2end = str(custom['end2end']).lower() == 'true'

        return ModelMeta(
            input_size=(int(input_h), int(input_w)),
            names=names,
            end2end=end2end,
            description=custom.get('description', ''),
        )

    # -------------------------------------------------- infer

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        if blob.dtype != self._input_dtype:
            blob = blob.astype(self._input_dtype)
        if self._binding is not None:
            self._binding.bind_input(
                name=self.input_name,
                device_type='cpu',
                device_id=0,
                element_type=blob.dtype,
                shape=tuple(blob.shape),
                buffer_ptr=blob.ctypes.data,
            )
            self.session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()
            return [self._pre_alloc_outputs[name] for name in self._output_names]

        outputs = self.session.run(self._output_names, {self.input_name: blob})
        return [np.asarray(out, dtype=np.float32) for out in outputs]

    @property
    def backend_name(self) -> str:
        return f'OnnxRuntimeBackend({self.device})'

    def on_exit(self) -> None:
        self.session = None
        self._binding = None
        self._pre_alloc_outputs = None
