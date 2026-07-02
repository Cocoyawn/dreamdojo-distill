#!/bin/bash
# Stage 4: teacher baseline PSNR over N samples (instead of single index).
# Wraps eval_teacher.sh with a loop. 1 GPU.
#
# Usage:
#   bash local_jobs/eval_teacher_baseline.sh                   # 5 samples, 49 frames
#   SAMPLES=10 NUM_FRAMES=49 bash local_jobs/eval_teacher_baseline.sh
#
# Output:
#   results/piper_eval_teacher_baseline/
#     ├── idx_0000_frames_0049_*.mp4 (gt/pred/merged)
#     ├── idx_0000_frames_0049_*.npz
#     ├── ...
#     └── psnr_curve.{json,csv}    ← aggregated across all indices

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

SAMPLES=${SAMPLES:-5}
NUM_FRAMES=${NUM_FRAMES:-49}
START_INDEX=${START_INDEX:-0}
SAVE_DIR=${SAVE_DIR:-$REPO/results/piper_eval_teacher_baseline}
CKPT=${CKPT:-checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model_ema_bf16.pt}

mkdir -p "$SAVE_DIR"
LOG_FILE="$LOG_DIR/eval_teacher_baseline_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== Teacher baseline eval: $SAMPLES samples × $NUM_FRAMES frames ==="
  echo "ckpt    : $CKPT"
  echo "save_dir: $SAVE_DIR"
  for i in $(seq "$START_INDEX" $((START_INDEX + SAMPLES - 1))); do
    echo
    echo "===== sample $i / $((START_INDEX + SAMPLES - 1)) ====="
    python scripts/run_piper_autoreg_compare.py \
      --ckpt-path "$CKPT" \
      --save-dir "$SAVE_DIR" \
      --dataset-path datasets/piper_insert_mouse_battery_lerobot \
      --index "$i" \
      --num-frames "$NUM_FRAMES" \
      --height 1440 --width 640 --video-key video.cam_vertical --save-fps 10
  done

  echo
  echo "=== aggregate PSNR curve ==="
  python scripts/codex_psnr_curve.py "$SAVE_DIR" \
    --out-json "$SAVE_DIR/psnr_curve.json" \
    --out-csv  "$SAVE_DIR/psnr_curve.csv"
  echo
  echo "=== summary ==="
  cat "$SAVE_DIR/psnr_curve.json" 2>/dev/null | head -50
} 2>&1 | tee "$LOG_FILE"
