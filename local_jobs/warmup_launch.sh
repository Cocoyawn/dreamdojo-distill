#!/bin/bash
# Wrapper around warmup.sh with named modes and safer defaults.
#
# Modes:
#   dryrun   —  8 GPU × batch=1 × 200 iter, wandb offline. Use to catch OOM.
#   small    —  8 GPU × batch=2 × 2000 iter, wandb online. Use to check convergence.
#   full     —  8 GPU × batch=2 × 10000 iter, wandb online. Recommended full run.
#
# Usage:
#   bash local_jobs/warmup_launch.sh dryrun
#   bash local_jobs/warmup_launch.sh full
#
# Overrides (env vars):
#   NPROC=8        — GPU count (default 8)
#   BATCH=2        — physical batch size / GPU
#   MAX_ITER=10000 — training iterations
#   WANDB_MODE=online|offline
#   RUN_NAME=piper_warmup_2026xxxx  — wandb / ckpt directory tag

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
MODE=${1:-dryrun}

# ------------------------------------------------ pre-flight
echo "======================================================================"
echo "  DreamDojo warmup launch  [mode: $MODE]"
echo "======================================================================"

if [ ! -d "datasets/piper_warmup_regenerated_4step/latents" ]; then
  echo "❌ dataset missing at datasets/piper_warmup_regenerated_4step/"
  echo "   run: bash local_jobs/setup_remote.sh"; exit 1
fi
if [ ! -f "datasets/cr1_empty_string_text_embeddings.pt" ] && [ ! -L "datasets/cr1_empty_string_text_embeddings.pt" ]; then
  echo "❌ datasets/cr1_empty_string_text_embeddings.pt missing"
  echo "   run: bash local_jobs/setup_remote.sh"; exit 1
fi

# ------------------------------------------------ mode presets
case "$MODE" in
  dryrun)
    NPROC=${NPROC:-8}
    BATCH=${BATCH:-1}
    MAX_ITER=${MAX_ITER:-200}
    WANDB_MODE=${WANDB_MODE:-offline}
    SAVE_ITER=${SAVE_ITER:-200}
    LOGGING_ITER=${LOGGING_ITER:-5}
    RUN_NAME=${RUN_NAME:-piper_warmup_dryrun_$(date +%Y%m%d_%H%M%S)}
    ;;
  small)
    NPROC=${NPROC:-8}
    BATCH=${BATCH:-2}
    MAX_ITER=${MAX_ITER:-2000}
    WANDB_MODE=${WANDB_MODE:-online}
    SAVE_ITER=${SAVE_ITER:-500}
    LOGGING_ITER=${LOGGING_ITER:-20}
    RUN_NAME=${RUN_NAME:-piper_warmup_small_$(date +%Y%m%d_%H%M%S)}
    ;;
  full)
    NPROC=${NPROC:-8}
    BATCH=${BATCH:-2}
    MAX_ITER=${MAX_ITER:-10000}
    WANDB_MODE=${WANDB_MODE:-online}
    SAVE_ITER=${SAVE_ITER:-1000}
    LOGGING_ITER=${LOGGING_ITER:-20}
    RUN_NAME=${RUN_NAME:-piper_warmup_full_$(date +%Y%m%d_%H%M%S)}
    ;;
  *)
    echo "unknown mode: $MODE"
    echo "usage: bash local_jobs/warmup_launch.sh {dryrun|small|full}"
    exit 1
    ;;
esac

echo "  NPROC       = $NPROC"
echo "  BATCH/GPU   = $BATCH  (effective batch = $((NPROC * BATCH)))"
echo "  MAX_ITER    = $MAX_ITER"
echo "  WANDB_MODE  = $WANDB_MODE"
echo "  SAVE_ITER   = $SAVE_ITER"
echo "  LOGGING_ITER= $LOGGING_ITER"
echo "  RUN_NAME    = $RUN_NAME"
echo

# ------------------------------------------------ hand-off to warmup.sh
NPROC="$NPROC" \
MAX_ITER="$MAX_ITER" \
WANDB_MODE="$WANDB_MODE" \
  bash local_jobs/warmup.sh \
  dataloader_train.batch_size=$BATCH \
  trainer.save_iter=$SAVE_ITER \
  trainer.logging_iter=$LOGGING_ITER \
  job.name="$RUN_NAME"
