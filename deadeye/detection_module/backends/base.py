# coding: utf-8
# cython: language_level=3
"""推理后端抽象：统一输入 blob 与输出张量格式，供 YoloDetector 按模型格式分发。"""
from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def parse_names_metadata(raw: str) -> dict[int, str]:
    """
    解析 Ultralytics 写入模型的类别名元数据。
    兼容 JSON（新版导出）与 Python repr（旧版单引号）两种格式。
    """
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            return {int(k): str(v) for k, v in dict(parsed).items()}
        except (ValueError, TypeError, SyntaxError):
            continue
    return {}


@dataclass
class ModelMeta:
    """模型元数据（从 Ultralytics ONNX/导出引擎元信息或 sidecar JSON 读取）。"""

    input_size: tuple[int, int]  # (h, w)
    names: dict[int, str] = field(default_factory=dict)
    end2end: bool | None = None  # True: 端到端免 NMS（YOLOv10/26）；None: 未知，按形状判断
    description: str = ''

    def name_list(self) -> list[str]:
        if not self.names:
            return []
        count = max(self.names) + 1
        return [self.names.get(i, str(i)) for i in range(count)]


class BaseInferBackend(ABC):
    """
    推理后端基类。

    约定：
    - ``infer(blob)`` 输入 float32 CHW blob（见 preprocess.Letterboxer），
      返回网络原始输出列表（保留 batch 维）。
    - ``meta`` 在构造完成后可用。
    - 后端实现应在 __init__ 内惰性导入可选依赖，未安装时抛出带安装提示的 RuntimeError。
    """

    meta: ModelMeta

    @abstractmethod
    def infer(self, blob: np.ndarray) -> list[np.ndarray]: ...

    @property
    def backend_name(self) -> str:
        return type(self).__name__

    def on_exit(self) -> None:
        return

    def __repr__(self) -> str:
        return (f'<{self.backend_name}: input={self.meta.input_size[0]}x{self.meta.input_size[1]}, '
                f'classes={len(self.meta.names)}, end2end={self.meta.end2end}>')
