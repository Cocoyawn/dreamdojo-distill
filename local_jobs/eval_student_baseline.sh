#!/bin/bash
# Stage 4: student autoregressive PSNR — same output format as eval_teacher_baseline,
# so codex_psnr_curve.py aggregates side-by-side with teacher results.
#
# Usage:
#   # Test current warmup mid-checkpoint (5 samples, 49 frames):
#   CUDA_VISIBLE_DEVICES=0 \
#     CKPT=dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000004000 \
#     bash local_jobs/eval_student_baseline.sh
#
#   # Override sample count or save path:
#   SAMPLES=10 SAVE_DIR=results/piper_eval_student_iter10000 \
#     CKPT=.../iter_000010000 bash local_jobs/eval_student_baseline.sh

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

CKPT=${CKPT:?Set CKPT to a warmup/self_forcing iter_NNNNNN DCP dir}
SAMPLES=${SAMPLES:-5}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_STEPS=${NUM_STEPS:-4}
START_INDEX=${START_INDEX:-0}
SAVE_DIR=${SAVE_DIR:-$REPO/results/piper_eval_student_$(basename "$CKPT")}
EXPERIMENT=${EXPERIMENT:-cosmos_predict2p5_2B_action_piper_self_forcing_no_s3}
SEED=${SEED:-1}

mkdir -p "$SAVE_DIR"
LOG_FILE="$LOG_DIR/eval_student_baseline_$(date +%Y%m%d_%H%M%S).log"

# Same CUDA/TE shim used by launch_student_inference_teleop.sh
NVIDIA_LIB_BASE=$(.venv/bin/python -c "import nvidia; print(nvidia.__path__[0])")
TORCH_LIB=$(.venv/bin/python -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$NVIDIA_LIB_BASE/cuda_nvrtc/lib:$NVIDIA_LIB_BASE/cublas/lib:$NVIDIA_LIB_BASE/cufft/lib:$NVIDIA_LIB_BASE/cufile/lib:$NVIDIA_LIB_BASE/cusolver/lib:$NVIDIA_LIB_BASE/nccl/lib:$NVIDIA_LIB_BASE/cuda_runtime/lib:$NVIDIA_LIB_BASE/curand/lib:$NVIDIA_LIB_BASE/cudnn/lib:$NVIDIA_LIB_BASE/nvtx/lib:$NVIDIA_LIB_BASE/cuda_cupti/lib:$NVIDIA_LIB_BASE/nvjitlink/lib:$NVIDIA_LIB_BASE/cusparse/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=$(.venv/bin/python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")

{
  echo "=== Student baseline eval: $SAMPLES samples × $NUM_FRAMES frames @ $NUM_STEPS steps ==="
  echo "ckpt       : $CKPT"
  echo "experiment : $EXPERIMENT"
  echo "save_dir   : $SAVE_DIR"
  for i in $(seq "$START_INDEX" $((START_INDEX + SAMPLES - 1))); do
    echo
    echo "===== sample $i / $((START_INDEX + SAMPLES - 1)) ====="
    python scripts/run_piper_autoreg_compare_student.py \
      --ckpt-path "$CKPT" \
      --save-dir "$SAVE_DIR" \
      --dataset-path datasets/piper_insert_mouse_battery_lerobot \
      --index "$i" \
      --num-frames "$NUM_FRAMES" \
      --num-steps "$NUM_STEPS" \
      --height 1440 --width 640 --video-key video.cam_vertical --save-fps 10 \
      --experiment "$EXPERIMENT" \
      --seed "$SEED"
  done

  echo
  echo "=== aggregate PSNR curve ==="
  python scripts/codex_psnr_curve.py "$SAVE_DIR" \
    --out-json "$SAVE_DIR/psnr_curve.json" \
    --out-csv  "$SAVE_DIR/psnr_curve.csv"
  echo
  echo "=== summary ==="
  cat "$SAVE_DIR/psnr_curve.json" 2>/dev/null | python -m json.tool 2>/dev/null | head -20
} 2>&1 | tee "$LOG_FILE"
