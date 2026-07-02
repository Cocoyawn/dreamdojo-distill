#!/bin/bash
# Stage 3 (eval): teacher autoregressive PSNR. 1 GPU.
# Local equivalent of codex_jobs/codex_eval_teacher_piper.sbatch.

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

SAVE_DIR=${SAVE_DIR:-$REPO/results/piper_eval_teacher}
mkdir -p "$SAVE_DIR"

LOG_FILE="$LOG_DIR/eval_teacher_$(date +%Y%m%d_%H%M%S).log"

{
  python scripts/run_piper_autoreg_compare.py \
    --ckpt-path checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model_ema_bf16.pt \
    --save-dir "$SAVE_DIR" \
    --dataset-path datasets/piper_insert_mouse_battery_lerobot \
    --index "${INDEX:-0}" \
    --num-frames "${NUM_FRAMES:-49}" \
    --height 1440 --width 640 --video-key video.cam_vertical --save-fps 10

  python scripts/codex_psnr_curve.py "$SAVE_DIR" \
    --out-json "$SAVE_DIR/psnr_curve.json" \
    --out-csv "$SAVE_DIR/psnr_curve.csv"
} 2>&1 | tee "$LOG_FILE"
