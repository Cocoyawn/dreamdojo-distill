#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"
: "${HF_TOKEN:?Set HF_TOKEN before running this script}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

srun --jobid=456040 --overlap /project/peilab/srk/wmpo_workspace/DreamDojo/launch.sh dreamdojo_2b_1440_640_piper
