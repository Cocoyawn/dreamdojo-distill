#!/bin/bash
# Shared env for all DreamDojo distill jobs on this box (no SLURM, no conda).
# Source from each stage script: `source "$(dirname "$0")/_common.sh"`.

set -euo pipefail

# REPO = the DreamDojo-distill root (parent of local_jobs/).
# PROJECT = the parent of REPO — used only as a "home" area for auxiliary
# caches (hf_cache, ckpt symlink targets, secrets). Override with
# DD_PROJECT=/your/path if you want to relocate those caches elsewhere.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT=${DD_PROJECT:-$(dirname "$REPO")}
VENV=$REPO/.venv
LOG_DIR=$REPO/logs
HF_CACHE=${DD_HF_CACHE:-$PROJECT/data/hf_cache}
WANDB_ENV=$PROJECT/.secrets/wandb.env
PROXY_SH=$PROJECT/workplace/proxy.sh

TEACHER_LOCAL=$PROJECT/data/DreamDojo-ckpt/dreamdojo-piper-iter30000
TEACHER_LINK=$REPO/checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000
PIPER_DATA_SRC=$PROJECT/data/piper_full_lerobot
PIPER_DATA_LINK=$REPO/datasets/piper_insert_mouse_battery_lerobot

mkdir -p "$LOG_DIR" "$REPO/datasets" "$REPO/checkpoints" "$HF_CACHE"

# proxy + HF cache: avoid HF HEAD checks blocking on no-network; fall through proxy on cache miss.
if [ -f "$PROXY_SH" ]; then
  # shellcheck disable=SC1090
  source "$PROXY_SH"
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

if [ -f "$WANDB_ENV" ]; then
  # shellcheck disable=SC1091
  source "$WANDB_ENV"
fi

NVIDIA_DIR="$VENV/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_DIR" ]; then
  for lib_dir in "$NVIDIA_DIR"/*/lib; do
    [ -d "$lib_dir" ] && export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  done
  [ -d "$NVIDIA_DIR/cuda_nvrtc" ] && export CUDA_HOME="$NVIDIA_DIR/cuda_nvrtc"
fi

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HF_CACHE}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$REPO/dreamdojo_logs}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""
