#!/bin/bash
# Stage 3 (720x320 variant): self-forcing distillation.
# 8 GPUs FSDP. Loads student from a 720x320 warmup ckpt.

set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/_common.sh"

CODEX_CUDA_LIBDIR="$PROJECT/codex_libs/cuda"
mkdir -p "$CODEX_CUDA_LIBDIR"
if [ -f "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CODEX_CUDA_LIBDIR/libcudart.so"
  export LD_LIBRARY_PATH="$CODEX_CUDA_LIBDIR:${LD_LIBRARY_PATH:-}"
fi

export WANDB_HTTP_TIMEOUT=300
export WANDB_RETRY_MAX=20
export WANDB_STATS_SAMPLE_RATE_SECONDS=10
export WANDB_STATS_SAMPLES_PER_CORE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

WARMUP_CKPT=${WARMUP_CKPT:-}
SKIP_WARMUP_LOAD=${SKIP_WARMUP_LOAD:-0}

if [ -z "$WARMUP_CKPT" ] && [ "$SKIP_WARMUP_LOAD" != "1" ]; then
  echo "ERROR: WARMUP_CKPT not set. Pass:"
  echo "  WARMUP_CKPT=dreamdojo_logs/.../checkpoints/iter_000010000 bash $0"
  echo "Or set SKIP_WARMUP_LOAD=1 to start from teacher (wastes warmup)."
  exit 1
fi
if [ "$SKIP_WARMUP_LOAD" != "1" ] && [ ! -d "$WARMUP_CKPT" ]; then
  echo "ERROR: warmup ckpt not found: $WARMUP_CKPT"; exit 1
fi

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12351}
NODE_RANK=${NODE_RANK:-0}
MAX_ITER=${MAX_ITER:-5000}
WANDB_MODE=${WANDB_MODE:-online}

LOG_FILE="$LOG_DIR/self_forcing_720x320_$(date +%Y%m%d_%H%M%S).log"

OVERRIDES=(
  "experiment=cosmos_predict2p5_2B_action_piper_720_320_self_forcing_no_s3"
  "job.wandb_mode=$WANDB_MODE"
  "trainer.max_iter=$MAX_ITER"
)
if [ "$SKIP_WARMUP_LOAD" != "1" ]; then
  OVERRIDES+=(
    "checkpoint.load_path=$WARMUP_CKPT"
    "checkpoint.strict_resume=False"
    "checkpoint.load_training_state=False"
  )
fi

echo "=== Self-forcing 720x320 launch ==="
echo "experiment    : cosmos_predict2p5_2B_action_piper_720_320_self_forcing_no_s3"
echo "student init  : ${WARMUP_CKPT:-<from teacher>}"
echo "teacher (DMD) : checkpoints/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k/model"
echo "NPROC         : $NPROC"
echo "MAX_ITER      : $MAX_ITER"
echo "log           : $LOG_FILE"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
  -- \
  "${OVERRIDES[@]}" \
  2>&1 | tee "$LOG_FILE"
