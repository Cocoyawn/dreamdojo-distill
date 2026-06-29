#!/bin/bash
#SBATCH --job-name=dd_setup
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/ysunem/ziwei_liu/logs/dd_setup-%j.out

set -euo pipefail

PROJECT=/project/peilab/ysunem/ziwei_liu
REPO=$PROJECT/DreamDojo
ENV_NAME=dreamdojo
TEACHER_REPO=Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000
TEACHER_DIR=$REPO/checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000
PIPER_DATA=$REPO/datasets/piper_insert_mouse_battery_lerobot
SOURCE_DATA=$PROJECT/datasets/battery_assemble_lerobot/mixed_success_seed42_train_lerobot

cd "$REPO"
mkdir -p "$REPO/datasets" "$REPO/checkpoints" "$PROJECT/logs"
ln -sfn "$SOURCE_DATA" "$PIPER_DATA"

sed "s#/project/peilab/srk/wmpo_workspace/piper_insert_mouse_battery_lerobot#$PIPER_DATA#g" \
  configs/2b_1440_640_piper.yaml > configs/2b_480_640_piper.yaml

source "$HOME/miniconda3/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel uv huggingface_hub

uv pip install --python "$CONDA_PREFIX/bin/python" -e '.[cu128]'
python -m pip install openai tyro numpydantic albumentations tianshou pyarrow fastparquet piq lightning moviepy==2.2.1 'jsonargparse[signatures]>=4.27.7'
python -m pip install torchcodec==0.5 --index-url=https://download.pytorch.org/whl/cu128
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git' --no-build-isolation || true

huggingface-cli download "$TEACHER_REPO" --local-dir "$TEACHER_DIR"

python - <<'PY'
from pathlib import Path
import torch
import piq
import huggingface_hub

root = Path("/project/peilab/ysunem/ziwei_liu/DreamDojo")
assert (root / "datasets/piper_insert_mouse_battery_lerobot/meta/info.json").exists()
assert (root / "configs/2b_480_640_piper.yaml").exists()
assert (root / "checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model_ema_bf16.pt").exists()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("setup_smoke_ok")
PY
