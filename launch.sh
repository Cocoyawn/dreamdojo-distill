#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
SEED=${SEED:-42}

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# CUDA 12 fix: Force PyTorch to use its bundled CUDA libraries
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""  # Clear any preloaded libraries

echo "Running on $NNODES nodes with $NPROC processes per node. This node rank is $NODE_RANK."

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-"$SCRIPT_DIR/.cache/huggingface"}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-"$SCRIPT_DIR/dreamdojo_logs"}
# export WANDB_API_KEY=  # Set your key before removing job.wandb_mode=disabled

for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  proxy_value=${!proxy_var:-}
  if [[ "$proxy_value" == http://127.0.0.1:* || "$proxy_value" == http://localhost:* ]]; then
    unset "$proxy_var"
  fi
done

source "$SCRIPT_DIR/.venv/bin/activate"

PYTHON_SITE_PACKAGES=$("$VIRTUAL_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
IMAGEIO_FFMPEG_EXE=$("$VIRTUAL_ENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
if [[ -x "$IMAGEIO_FFMPEG_EXE" ]]; then
  export IMAGEIO_FFMPEG_EXE
  FFMPEG_COMPAT_BIN_DIR="$SCRIPT_DIR/.cache/ffmpeg/bin"
  mkdir -p "$FFMPEG_COMPAT_BIN_DIR"
  ln -sfn "$IMAGEIO_FFMPEG_EXE" "$FFMPEG_COMPAT_BIN_DIR/ffmpeg"
  export PATH="$FFMPEG_COMPAT_BIN_DIR:$PATH"
fi
VENV_NVIDIA_DIR="$PYTHON_SITE_PACKAGES/nvidia"
if [[ -d "$VENV_NVIDIA_DIR/cuda_nvrtc" && ! -e "${CUDA_HOME:-/usr/local/cuda}/lib64/libnvrtc.so" ]]; then
  export CUDA_HOME="$VENV_NVIDIA_DIR/cuda_nvrtc"
fi
for lib_dir in "$VENV_NVIDIA_DIR"/*/lib; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done
CUDA_COMPAT_LIB_DIR="$SCRIPT_DIR/.cache/cuda-compat/lib"
mkdir -p "$CUDA_COMPAT_LIB_DIR"
if [[ ! -e "$CUDA_COMPAT_LIB_DIR/libcudart.so" && -e "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]]; then
  ln -s "$VENV_NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CUDA_COMPAT_LIB_DIR/libcudart.so"
fi
export LD_LIBRARY_PATH="$CUDA_COMPAT_LIB_DIR:${LD_LIBRARY_PATH:-}"

config_name=${1:?Usage: $0 <experiment_name> [extra hydra overrides...]}
shift

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=$config_name \
  job.wandb_mode=online \
  ~dataloader_train.dataloaders \
  "$@"
