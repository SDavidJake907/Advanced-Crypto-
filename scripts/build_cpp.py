from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pybind11


ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "cpp" / "build"


def main() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cuda_root = os.environ.get("CUDA_PATH")
    cmake_cmd = ["cmake", "..", f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"]
    if cuda_root:
        nvcc = Path(cuda_root) / "bin" / "nvcc.exe"
        cmake_cmd.extend(
            [
                f"-DCUDAToolkit_ROOT={cuda_root}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}",
                "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler",
                f"-Tcuda={cuda_root}",
            ]
        )
    subprocess.check_call(
        cmake_cmd,
        cwd=BUILD_DIR,
    )
    subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=BUILD_DIR)
    subprocess.check_call(["cmake", "--install", "."], cwd=BUILD_DIR)


if __name__ == "__main__":
    main()
