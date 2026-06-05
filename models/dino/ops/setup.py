# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import os
import glob
import platform

import torch


def _apply_default_torch_cuda_arch_list() -> None:
    """
    若未设置 TORCH_CUDA_ARCH_LIST，则自动为当前 GPU 生成合适的架构列表。
    - 默认：常见 SM + 自动补上本机 GPU 的 compute capability（覆盖 Blackwell 12.0 等新卡）。
      注意：部分 CUDA Toolkit 的 nvcc 尚不支持 compute_110（TORCH_CUDA_ARCH_LIST 中的 11.0），
      在 Colab 等环境会报 “Unsupported gpu architecture 'compute_110'”，默认列表因此不包含 11.0；
      若 GPU 为 sm_110 且 nvcc 已支持，请手动设置 TORCH_CUDA_ARCH_LIST=11.0。
    - DINO_MSDA_SINGLE_ARCH=1：只编当前 GPU（Colab 换卡后重装最快；换 GPU 需再编译一次）。
    仍可直接设置 TORCH_CUDA_ARCH_LIST=8.0 等覆盖本逻辑。
    参考: https://pytorch.org/docs/stable/cpp_extension.html
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip():
        return
    single = os.environ.get("DINO_MSDA_SINGLE_ARCH", "").strip().lower() in ("1", "true", "yes", "y")
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cur = f"{p.major}.{p.minor}"
            if single:
                os.environ["TORCH_CUDA_ARCH_LIST"] = cur
                return
    except Exception:
        pass

    # 不含 "11.0"：若干 nvcc（含部分 Colab 镜像）不支持 compute_110，会 fatal；需要 sm_110 时自行设环境变量。
    base = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0"]
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cur = f"{p.major}.{p.minor}"
            if cur not in base:
                base.append(cur)
    except Exception:
        pass
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(base)


_apply_default_torch_cuda_arch_list()

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []



    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        # MSVC 2025/2026 may be newer than nvcc officially supports
        if platform.system() == "Windows":
            extra_compile_args["nvcc"].append("-allow-unsupported-compiler")
    else:
        raise NotImplementedError(
            'CUDA is not available (torch.cuda.is_available() is False or CUDA_HOME is unset). '
            'Install a CUDA-enabled PyTorch build and the CUDA Toolkit, then run from this folder: '
            'pip install -e .   or   python setup.py build install'
        )

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
