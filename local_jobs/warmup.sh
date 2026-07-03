#!/bin/bash
# Stage 2: warmup distillation. 8 GPUs default (override with NPROC=).
# Local equivalent of codex_jobs/codex_warmup_piper.sbatch.

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

# warmup script needs libcudart.so visible; mirror the codex shim.
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

LOG_FILE="$LOG_DIR/warmup_$(date +%Y%m%d_%H%M%S).log"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
  -- experiment=cosmos_predict2p5_2B_action_piper_warmup_no_s3 \
  job.wandb_mode=${WANDB_MODE:-online} \
  trainer.max_iter="${MAX_ITER:-20000}" \
  "$@" \
  2>&1 | tee "$LOG_FILE"
