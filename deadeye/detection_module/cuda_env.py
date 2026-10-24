# coding: utf-8
# cython: language_level=3
"""
CUDA 运行库 DLL 引导。

onnxruntime-gpu / TensorRT 的 Windows 包不自带 CUDA 运行库，pip 安装的
NVIDIA 包（nvidia-*-cu13）与 torch（torch/lib）也不会自动进入 DLL 搜索路径。
本模块把这些目录显式注册进当前进程，使 CUDA EP 能找到 cublas / cudnn 等 DLL。

创建任何推理会话之前调用一次 ``bootstrap_nvidia_dlls()`` 即可。
"""
from __future__ import annotations

import glob
import os
import sysconfig

_registered: list[str] = []


def _candidate_dirs(site_dir: str) -> list[str]:
    candidates: list[str] = []
    # pip 安装的 nvidia-*-cu13 / cu12 包（Linux 布局 nvidia/<pkg>/bin）
    candidates.extend(glob.glob(os.path.join(site_dir, 'nvidia', '*', 'bin')))
    candidates.extend(glob.glob(os.path.join(site_dir, 'nvidia', '*', 'x64')))
    # torch 捆绑的完整 CUDA 运行库（Windows 下最可靠的来源）
    candidates.append(os.path.join(site_dir, 'torch', 'lib'))
    return candidates


def bootstrap_nvidia_dlls() -> list[str]:
    """把 CUDA DLL 目录注册进 DLL 搜索路径，返回成功注册的目录。"""
    if _registered:
        return _registered
    if os.name != 'nt':
        return _registered

    site_dir = sysconfig.get_paths()['purelib']
    for directory in _candidate_dirs(site_dir):
        if not os.path.isdir(directory):
            continue
        try:
            os.add_dll_directory(directory)
            os.environ['PATH'] = directory + os.pathsep + os.environ.get('PATH', '')
            _registered.append(directory)
        except OSError:
            continue
    return _registered
