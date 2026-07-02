#!/bin/bash
# Stage 1: teacher generation. 8 GPUs, single node.
# Local equivalent of codex_jobs/codex_teacher_gen_piper.sbatch.

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

LOG_FILE="$LOG_DIR/teacher_gen_$(date +%Y%m%d_%H%M%S).log"

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup \
  -- \
  --experiment=dreamdojo_2b_1440_640_piper \
  --ckpt_path checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model \
  --save_root datasets/piper_warmup_regenerated_4step \
  --resolution 1440,640 --video_key video.cam_vertical --fps 10 \
  --guidance 0 --chunk_size 12 --start "${START:-0}" --end "${END:-10000}" \
  --query_steps 0,9,18,27,34 --context_parallel_size 1 \
  2>&1 | tee "$LOG_FILE"
