#!/bin/bash
#SBATCH --job-name=dd_teacher_piper
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/ysunem/ziwei_liu/logs/dd_teacher_piper-%j.out

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
  --node_rank=$NODE_RANK -m cosmos_predict2._src.predict2.action.inference.inference_gr00t_warmup \
  -- \
  --experiment=dreamdojo_2b_1440_640_piper \
  --ckpt_path checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model \
  --save_root datasets/piper_warmup_regenerated_4step \
  --resolution 1440,640 --video_key video.cam_vertical --fps 10 \
  --guidance 0 --chunk_size 12 --start "${START:-0}" --end "${END:-10000}" \
  --query_steps 0,9,18,27,34 --context_parallel_size 1
