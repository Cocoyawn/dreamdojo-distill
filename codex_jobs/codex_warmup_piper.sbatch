#!/bin/bash
#SBATCH --job-name=dd_warmup_piper
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=/project/peilab/ysunem/ziwei_liu/logs/dd_warmup_piper-%j.out

set -euo pipefail

PROJECT=/project/peilab/ysunem/ziwei_liu
REPO=$PROJECT/DreamDojo
cd "$REPO"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamdojo
source "$PROJECT/.secrets/wandb.env"

NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
for lib_dir in "$NVIDIA_DIR"/*/lib; do
  export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
done
CODEX_CUDA_LIBDIR="$PROJECT/codex_libs/cuda"
mkdir -p "$CODEX_CUDA_LIBDIR"
ln -sfn "$NVIDIA_DIR/cuda_runtime/lib/libcudart.so.12" "$CODEX_CUDA_LIBDIR/libcudart.so"
export LD_LIBRARY_PATH="$CODEX_CUDA_LIBDIR:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="$NVIDIA_DIR/cuda_nvrtc"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$PROJECT/hf_home}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$REPO/dreamdojo_logs}
export OMP_NUM_THREADS=8
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr "$MASTER_ADDR" \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py \
  -- experiment=cosmos_predict2p5_2B_action_piper_warmup_no_s3 \
  job.wandb_mode=online \
  trainer.max_iter="${MAX_ITER:-20000}" \
  2>&1 | tee "$PROJECT/logs/dd_warmup_piper_train.log"
