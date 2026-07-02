#!/bin/bash
# Stage 4 / 5: student inference + PSNR. 1 GPU.
# Takes any student checkpoint (warmup-mid, warmup-final, or self-forcing) and
# generates videos from datasets/eval/info.json, then computes PSNR vs ground truth.
#
# Usage:
#   # 1. one-time: build eval entries (if datasets/eval/info.json missing)
#   .venv/bin/python scripts/build_eval_info_json.py --n 5
#
#   # 2. test current warmup mid ckpt
#   CKPT=dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000004000 \
#     bash local_jobs/eval_student.sh
#
#   # 3. once warmup done, test final
#   CKPT=dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000020000 \
#     bash local_jobs/eval_student.sh
#
# Notes:
#   - EXPERIMENT defaults to piper_self_forcing_no_s3 (student causal arch). Warmup ckpt
#     has the same student architecture so it can be loaded by this experiment.
#   - NUM_STEPS=4 matches student few-step distillation target.
#   - This needs ~25 GB VRAM. If 8 cards are in use by warmup FSDP, this may OOM.

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"

CKPT=${CKPT:?Set CKPT to a warmup/self_forcing iter_NNNNNN dir, e.g. dreamdojo_logs/.../iter_000004000}
EXPERIMENT=${EXPERIMENT:-cosmos_predict2p5_2B_action_piper_self_forcing_no_s3}
INPUT_JSON=${INPUT_JSON:-datasets/eval/info.json}
RESULTS_DIR=${RESULTS_DIR:-$REPO/results/piper_eval_student}
NUM_STEPS=${NUM_STEPS:-4}
MAX_FRAMES=${MAX_FRAMES:-13}
SEED=${SEED:-1}
MASTER_PORT=${MASTER_PORT:-29520}
NPROC=${NPROC:-1}

mkdir -p "$RESULTS_DIR"
LOG_FILE="$LOG_DIR/eval_student_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "$INPUT_JSON" ]; then
  echo "ERROR: $INPUT_JSON missing. Run:"
  echo "  .venv/bin/python scripts/build_eval_info_json.py --n 5"
  exit 1
fi

# Same TE/CUDA shim used by launch_student_inference_teleop.sh
NVIDIA_LIB_BASE=$(.venv/bin/python -c "import nvidia; print(nvidia.__path__[0])")
TORCH_LIB=$(.venv/bin/python -c "import torch; print(torch.__path__[0])")/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$NVIDIA_LIB_BASE/cuda_nvrtc/lib:$NVIDIA_LIB_BASE/cublas/lib:$NVIDIA_LIB_BASE/cufft/lib:$NVIDIA_LIB_BASE/cufile/lib:$NVIDIA_LIB_BASE/cusolver/lib:$NVIDIA_LIB_BASE/nccl/lib:$NVIDIA_LIB_BASE/cuda_runtime/lib:$NVIDIA_LIB_BASE/curand/lib:$NVIDIA_LIB_BASE/cudnn/lib:$NVIDIA_LIB_BASE/nvtx/lib:$NVIDIA_LIB_BASE/cuda_cupti/lib:$NVIDIA_LIB_BASE/nvjitlink/lib:$NVIDIA_LIB_BASE/cusparse/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=$(.venv/bin/python -c "import nvidia.cuda_nvrtc; print(nvidia.cuda_nvrtc.__path__[0])")

{
  echo "=== Student inference eval ==="
  echo "ckpt       : $CKPT"
  echo "experiment : $EXPERIMENT"
  echo "input_json : $INPUT_JSON"
  echo "results    : $RESULTS_DIR"
  echo "num_steps  : $NUM_STEPS  max_frames: $MAX_FRAMES"

  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    -m cosmos_predict2._src.predict2.interactive.inference.action_video2world \
    --config=cosmos_predict2/_src/predict2/interactive/configs/config_distill.py \
    --experiment="$EXPERIMENT" \
    --ckpt_path "$CKPT" \
    --input_json "$INPUT_JSON" \
    --num_steps "$NUM_STEPS" \
    --max_frames "$MAX_FRAMES" \
    --seed "$SEED" \
    --fps 10
} 2>&1 | tee "$LOG_FILE"
