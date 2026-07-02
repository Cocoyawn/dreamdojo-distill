#!/bin/bash
# Stage 3: self-forcing distillation. 8 GPUs FSDP.
# Local equivalent of launch_self_forcing.sh (fork's version), with piper wiring:
#   - experiment: cosmos_predict2p5_2B_action_piper_self_forcing_no_s3
#   - student net initialised from our own warmup ckpt (via checkpoint.load_path override)
#   - teacher score / fake_score net loaded from iter30000 bidirectional teacher
#     (unchanged; this path is bidirectional arch and must NOT be replaced by warmup ckpt)
#
# Usage:
#   # After warmup finishes (iter_000020000 exists):
#   bash local_jobs/self_forcing.sh
#
#   # Override warmup ckpt (default = latest iter_000020000 of the current warmup run):
#   WARMUP_CKPT=dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000020000 \
#     bash local_jobs/self_forcing.sh
#
#   # Change iters / gpus / wandb:
#   NPROC=8 MAX_ITER=10000 WANDB_MODE=online bash local_jobs/self_forcing.sh
#
# Notes:
#   - Self-forcing loads 3 nets (student, teacher, fake_score) → memory heavier than warmup.
#     8-card FSDP recommended; 4-card likely OOM.
#   - torch_compile off by default (training path).

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

# libcudart shim (same as warmup)
CODEX_CUDA_LIBDIR="$PROJECT/codex_libs/cuda"
mkdir -p "$CODEX_CUDA_LIBDIR"
if [ -f "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" ]; then
  ln -sfn "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CODEX_CUDA_LIBDIR/libcudart.so"
  export LD_LIBRARY_PATH="$CODEX_CUDA_LIBDIR:${LD_LIBRARY_PATH:-}"
fi

# Extra env from fork's launch_self_forcing.sh
export WANDB_HTTP_TIMEOUT=300
export WANDB_RETRY_MAX=20
export WANDB_STATS_SAMPLE_RATE_SECONDS=10
export WANDB_STATS_SAMPLES_PER_CORE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # longer, since self-forcing steps are heavier
export FI_EFA_USE_DEVICE_RDMA=1
export RDMAV_FORK_SAFE=1
export TORCH_DIST_INIT_BARRIER=1

# Default WARMUP_CKPT points to the current warmup run's iter_20000.
# When warmup is still going, the file may not exist yet — script will refuse to launch
# unless SKIP_WARMUP_LOAD=1 is set (student would then init from bidirectional teacher).
WARMUP_RUN=${WARMUP_RUN:-dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04}
WARMUP_CKPT=${WARMUP_CKPT:-$WARMUP_RUN/checkpoints/iter_000017000}
SKIP_WARMUP_LOAD=${SKIP_WARMUP_LOAD:-0}

if [ "$SKIP_WARMUP_LOAD" != "1" ] && [ ! -d "$WARMUP_CKPT" ]; then
  echo "ERROR: warmup ckpt not found: $WARMUP_CKPT"
  echo "Options:"
  echo "  1) point WARMUP_CKPT= at another iter (iter_000016000 / iter_000018000 etc)"
  echo "  2) SKIP_WARMUP_LOAD=1 to start from bidirectional teacher (wastes our warmup training)"
  exit 1
fi

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12351}   # different from warmup default 12341
NODE_RANK=${NODE_RANK:-0}
MAX_ITER=${MAX_ITER:-5000}
WANDB_MODE=${WANDB_MODE:-online}

LOG_FILE="$LOG_DIR/self_forcing_$(date +%Y%m%d_%H%M%S).log"

# Build the override list. If WARMUP_CKPT is set, tell Checkpointer to load it
# on top of the init_student_with_teacher initialisation (which will populate
# net_teacher and net_fake_score correctly).
OVERRIDES=(
  "experiment=cosmos_predict2p5_2B_action_piper_self_forcing_no_s3"
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

echo "=== Self-forcing launch ==="
echo "experiment    : cosmos_predict2p5_2B_action_piper_self_forcing_no_s3"
echo "student init  : ${WARMUP_CKPT:-<from bidirectional teacher>}"
echo "teacher (DMD) : checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model (unchanged)"
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
