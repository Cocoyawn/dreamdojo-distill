#!/bin/bash
# Stage 1 (720x320 variant): teacher generation using the new teacher ckpt
# Shirk6/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k.
# 8 GPUs single node.

set -u
cd "$(dirname "$0")/.."
source local_jobs/_common.sh

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

CKPT=${CKPT:-checkpoints/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k/model}
SAVE_ROOT=${SAVE_ROOT:-datasets/piper_720_320_warmup_regenerated_4step}
DATASET_PATH=${DATASET_PATH:-}
START=${START:-0}
END=${END:-10000}

LOG_FILE="$LOG_DIR/teacher_gen_720x320_$(date +%Y%m%d_%H%M%S).log"

if [ ! -d "$CKPT" ]; then
  echo "ERROR: teacher ckpt not found: $CKPT"
  echo "Run: hf download Shirk6/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k --local-dir checkpoints/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k"
  exit 1
fi

echo "=== teacher_gen 720x320 ==="
echo "  ckpt        : $CKPT"
echo "  save_root   : $SAVE_ROOT"
echo "  dataset_path: ${DATASET_PATH:-<default (get_data_path(piper))>}"
echo "  range       : [$START, $END)"
echo "  NPROC       : $NPROC"
echo "  log         : $LOG_FILE"
echo

DATASET_ARG=()
if [ -n "$DATASET_PATH" ]; then
    DATASET_ARG=(--dataset_path "$DATASET_PATH")
fi

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup \
  -- \
  --experiment=dreamdojo_2b_720_320_piper \
  --ckpt_path "$CKPT" \
  --save_root "$SAVE_ROOT" \
  "${DATASET_ARG[@]}" \
  --resolution 720,320 --video_key video.cam_vertical --fps 10 \
  --guidance 0 --chunk_size 12 --start "$START" --end "$END" \
  --query_steps 0,9,18,27,34 --context_parallel_size 1 \
  2>&1 | tee "$LOG_FILE"
