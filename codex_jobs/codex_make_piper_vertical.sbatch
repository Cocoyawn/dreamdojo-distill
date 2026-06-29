#!/bin/bash
#SBATCH --job-name=dd_make_piper_vertical
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/ysunem/ziwei_liu/logs/dd_make_piper_vertical-%j.out

set -euo pipefail

REPO=/project/peilab/ysunem/ziwei_liu/DreamDojo
cd "$REPO"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamdojo_cu121
python scripts/codex_make_piper_vertical.py
