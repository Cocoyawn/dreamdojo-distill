#!/bin/bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamdojo
NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
for lib_dir in "$NVIDIA_DIR"/*/lib; do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done
export CUDA_HOME="$NVIDIA_DIR/cuda_nvrtc"
python -c 'import transformer_engine; print("te_ok")'
