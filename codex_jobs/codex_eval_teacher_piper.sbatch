#!/bin/bash
#SBATCH --job-name=dd_eval_teacher
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/ysunem/ziwei_liu/logs/dd_eval_teacher-%j.out

set -euo pipefail

PROJECT=/project/peilab/ysunem/ziwei_liu
REPO=$PROJECT/DreamDojo
SAVE_DIR=$REPO/results/piper_eval_teacher
cd "$REPO"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamdojo

NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
for lib_dir in "$NVIDIA_DIR"/*/lib; do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done
export CUDA_HOME="$NVIDIA_DIR/cuda_nvrtc"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$PROJECT/hf_home}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$REPO/dreamdojo_logs}

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
