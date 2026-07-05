#!/bin/bash
# Stage 2 (720x320 variant): warmup training on the 720x320 ODE dataset.
# 8 GPUs default.

cd "$(dirname "$0")/.."
source "$(dirname "$0")/_common.sh"

CODEX_CUDA_LIBDIR="$PROJECT/codex_libs/cuda"
mkdir -p "$CODEX_CUDA_LIBDIR"
if [ -f "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CODEX_CUDA_LIBDIR/libcudart.so"
  export LD_LIBRARY_PATH="$CODEX_CUDA_LIBDIR:${LD_LIBRARY_PATH:-}"
fi

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

LOG_FILE="$LOG_DIR/warmup_720x320_$(date +%Y%m%d_%H%M%S).log"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
  -- experiment=cosmos_predict2p5_2B_action_piper_720_320_warmup_no_s3 \
  job.wandb_mode=${WANDB_MODE:-online} \
  trainer.max_iter="${MAX_ITER:-20000}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"
