import os
from pathlib import Path
from setuptools import setup, Extension
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT_DIR = os.path.dirname(__file__)

def _is_cuda() -> bool:
      return (torch.version.cuda is not None)

ext_modules = []

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = ["-O2", "-std=c++17"]

def glob(pattern: str):
    root = Path(__name__).parent
    return [str(p) for p in root.glob(pattern)]


if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

CUTLASS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../3rdparty/cutlass"))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if _is_cuda():
    cutlass_include_dirs = [
        f"{CUTLASS_DIR}/include",
        f"{CUTLASS_DIR}/examples/common",
        f"{CUTLASS_DIR}/tools/util/include",
        f"{CUTLASS_DIR}/tools/util/include",
        f"{PARENT_DIR}"

    ]
    ext_modules.append(
        CUDAExtension(
            name="my_torch_ext_v2",
            sources=glob("tma_double_buffer_add.cu") + glob("pybind.cpp"),
            include_dirs=cutlass_include_dirs,
            extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
            },
        )
      )
      
setup(name='my_torch_ext_v2',
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension}
)
