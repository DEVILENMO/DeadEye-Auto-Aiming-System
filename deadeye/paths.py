# coding: utf-8
# cython: language_level=3
"""
仓库内默认路径约定（与运行时 CWD 无关，统一锚定仓库根目录）：

- ``res/``     静态资源：封面图 / 演示动图 / 示例图片与视频
- ``weights/`` 模型权重（.pt / .onnx / .engine）
- ``output/``  运行生成物（带标注的视频等，不入库）
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    """仓库根下的绝对路径。"""
    return os.path.join(REPO_ROOT, *parts)


def res_path(*parts: str) -> str:
    """res/ 资源目录下的绝对路径。"""
    return repo_path('res', *parts)


def weights_path(*parts: str) -> str:
    """weights/ 权重目录下的绝对路径。"""
    return repo_path('weights', *parts)


def output_path(*parts: str) -> str:
    """output/ 生成物目录下的绝对路径。"""
    return repo_path('output', *parts)
