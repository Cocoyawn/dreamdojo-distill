#!/bin/bash
set -euo pipefail

PROJECT=/project/peilab/ysunem/ziwei_liu
REPO=$PROJECT/DreamDojo
cd "$REPO"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamdojo

NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
for lib_dir in "$NVIDIA_DIR"/*/lib; do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done

CODEX_CUDA_LIBDIR="$PROJECT/codex_libs/cuda"
mkdir -p "$CODEX_CUDA_LIBDIR"
ln -sfn "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CODEX_CUDA_LIBDIR/libcudart.so"
export LD_LIBRARY_PATH="$CODEX_CUDA_LIBDIR:${LD_LIBRARY_PATH:-}"

python -c 'import ctypes; ctypes.CDLL("libcudart.so"); print("cudart_ok")'
