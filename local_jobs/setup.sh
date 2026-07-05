#!/bin/bash
# One-time setup: uv venv + pip install + dataset/teacher symlinks + smoke check.
# Local equivalent of codex_jobs/codex_setup_dreamdojo.sbatch (no conda, no SLURM).
# Skips: HF teacher download (already present) and `sed` config gen (already present).
#
# This script is kept for the ORIGINAL local box (/mnt/afs-h200/yuyangcheng).
# For a fresh remote box use `local_jobs/setup_remote.sh` instead — that one
# has no hardcoded paths.

set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT=${DD_PROJECT:-$(dirname "$REPO")}
VENV=$REPO/.venv
UV_BIN=${UV_BIN:-uv}
PROXY_SH=$PROJECT/workplace/proxy.sh

TEACHER_LOCAL=$PROJECT/data/DreamDojo-ckpt/dreamdojo-piper-iter30000
TEACHER_LINK=$REPO/checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000
PIPER_DATA_SRC=$PROJECT/data/piper_full_lerobot
PIPER_DATA_LINK=$REPO/datasets/piper_insert_mouse_battery_lerobot

cd "$REPO"
mkdir -p "$REPO/datasets" "$REPO/checkpoints" "$REPO/logs"

# 1) Symlink dataset + teacher into the layout the repo expects (only if sources exist).
[ -e "$PIPER_DATA_SRC" ] && ln -sfn "$PIPER_DATA_SRC" "$PIPER_DATA_LINK"
[ -e "$TEACHER_LOCAL" ]  && ln -sfn "$TEACHER_LOCAL"  "$TEACHER_LINK"

# 2) Generate the 480x640 piper config from the 1440x640 one if missing.
if [ ! -f configs/2b_480_640_piper.yaml ]; then
  sed "s#/project/peilab/srk/wmpo_workspace/piper_insert_mouse_battery_lerobot#$PIPER_DATA_LINK#g" \
    configs/2b_1440_640_piper.yaml > configs/2b_480_640_piper.yaml
fi

# 3) Build / refresh the venv with uv (proxy needed for any PyPI fetches).
# shellcheck disable=SC1090
[ -f "$PROXY_SH" ] && source "$PROXY_SH"

if [ ! -f "$VENV/bin/activate" ]; then
  "$UV_BIN" venv --seed --python 3.10 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
"$UV_BIN" pip install --python "$VENV/bin/python" -U pip setuptools wheel huggingface_hub

"$UV_BIN" pip install --python "$VENV/bin/python" -e '.[cu128]'
"$UV_BIN" pip install --python "$VENV/bin/python" openai tyro numpydantic albumentations tianshou pyarrow fastparquet piq lightning moviepy==2.2.1 'jsonargparse[signatures]>=4.27.7'
"$UV_BIN" pip install --python "$VENV/bin/python" torchcodec==0.5 --index-url=https://download.pytorch.org/whl/cu128
"$UV_BIN" pip install --python "$VENV/bin/python" 'git+https://github.com/facebookresearch/pytorch3d.git' --no-build-isolation || true

# 4) Smoke check.
REPO_ENV="$REPO" python - <<'PY'
from pathlib import Path
import os, torch, piq, huggingface_hub  # noqa: F401

root = Path(os.environ["REPO_ENV"])
if (root / "datasets/piper_insert_mouse_battery_lerobot/meta/info.json").exists():
    print("dataset symlink ok")
if (root / "configs/2b_480_640_piper.yaml").exists():
    print("config ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("setup_smoke_ok")
PY
