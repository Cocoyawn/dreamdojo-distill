#!/bin/bash
# Bootstrap a fresh remote box to run DreamDojo piper warmup / self-forcing.
#
# What this does:
#   1. Sanity-check env (GPU count, disk, network)
#   2. Install python deps via install.sh (uv + torch + xformers + ...)
#   3. Prompt for HF token, huggingface-cli login
#   4. Download dataset (210 GB tar) + extract
#   5. Download teacher ckpt (17 GB, needed for SF; benign to have during warmup)
#   6. Download cr1 embedding (small file inside nvidia/Cosmos-Predict2.5-2B)
#   7. Verify everything is in place
#
# Usage (from the repo root):
#   bash local_jobs/setup_remote.sh              # interactive; will ask for HF token
#   HF_TOKEN=hf_xxx bash local_jobs/setup_remote.sh   # non-interactive
#   SKIP_DEPS=1 SKIP_TEACHER=1 bash local_jobs/setup_remote.sh   # skip stages
#
# Skip flags (any truthy value):
#   SKIP_DEPS=1      — don't reinstall .venv
#   SKIP_DATASET=1   — don't re-download dataset
#   SKIP_TEACHER=1   — don't download teacher ckpt (safe if only doing warmup)
#   SKIP_CR1=1       — don't download cr1 embedding (dangerous — warmup needs it)
#   SKIP_VERIFY=1    — skip final verify step

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD

echo "======================================================================"
echo "  DreamDojo remote setup"
echo "  repo: $REPO"
echo "  date: $(date)"
echo "======================================================================"
echo

# ---------------------------------------------------------------- pre-flight
echo "=== Pre-flight ==="
echo "  hostname: $(hostname)"
echo "  cwd     : $PWD"
echo -n "  git head: "; git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo "(not a git repo — did you clone?)"

if ! command -v nvidia-smi >/dev/null; then
  echo "  ❌ no nvidia-smi found. This box needs GPUs."; exit 1
fi
N_GPU=$(nvidia-smi --list-gpus | wc -l)
FREE_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -d 'G ')
echo "  GPUs    : $N_GPU"
echo "  free disk: ${FREE_GB} GB (need ~450 GB: 210 dataset + 17 teacher + 30 venv + buffer)"

if [ "$FREE_GB" -lt 450 ]; then
  echo "  ⚠️  disk under 450 GB — training data + teacher ckpt may not fit"
fi
if [ "$N_GPU" -lt 4 ]; then
  echo "  ⚠️  fewer than 4 GPUs — warmup can still run but slower"
fi
echo

# ---------------------------------------------------------------- deps
if [ -z "${SKIP_DEPS:-}" ]; then
  echo "=== [1/5] Installing python deps (uv + torch 2.7.0+cu128 + ...) ==="
  if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    echo "  .venv/ already exists — skipping (set SKIP_DEPS=1 to be explicit)"
  else
    bash install.sh
  fi
  echo "  ✅ deps ready"
  echo
else
  echo "=== [1/5] SKIP_DEPS=1 — skipping install.sh ==="
  echo
fi

# ---------------------------------------------------------------- HF auth
echo "=== [2/5] HF authentication ==="
if [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
  echo "  found cached HF token at ~/.cache/huggingface/token"
elif [ -n "${HF_TOKEN:-}" ]; then
  echo "  using HF_TOKEN from environment"
else
  echo "  no HF token found."
  echo "  paste HF token (input hidden), or Ctrl-C and re-run with HF_TOKEN=hf_xxx:"
  read -r -s HF_TOKEN
  echo
fi

export HF_TOKEN
# Persist for subsequent commands in this session
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=0

# Verify token
mkdir -p "$HOME/.cache/huggingface"
echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
WHO=$(.venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>&1)
if [ "${WHO:0:5}" = "Error" ] || [ -z "$WHO" ]; then
  echo "  ❌ HF token rejected: $WHO"; exit 1
fi
echo "  ✅ logged in as: $WHO"
echo

# ---------------------------------------------------------------- dataset
DATASET_DIR="$REPO/datasets/piper_warmup_regenerated_4step"
if [ -z "${SKIP_DATASET:-}" ]; then
  echo "=== [3/5] Downloading dataset (~210 GB) ==="
  mkdir -p "$REPO/datasets"

  if [ -d "$DATASET_DIR" ] && [ "$(ls "$DATASET_DIR/actions" 2>/dev/null | wc -l)" = "10000" ]; then
    echo "  dataset already extracted at $DATASET_DIR (10000 samples). Skipping."
  else
    echo "  downloading Cocoyawn32/dreamdojo-piper-warmup-4step ..."
    .venv/bin/hf download \
      Cocoyawn32/dreamdojo-piper-warmup-4step \
      --repo-type=dataset \
      --local-dir "$DATASET_DIR"

    echo "  extracting tars ..."
    cd "$DATASET_DIR"
    for f in latents.tar images.tar videos.tar actions.tar; do
      if [ -f "$f" ]; then
        echo "    unpacking $f ..."
        tar -xf "$f" && rm "$f"
      fi
    done
    cd "$REPO"
    echo "  ✅ dataset extracted at $DATASET_DIR"
  fi
  echo
else
  echo "=== [3/5] SKIP_DATASET=1 — skipping dataset download ==="
  echo
fi

# ---------------------------------------------------------------- teacher ckpt
TEACHER_DIR="$REPO/checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000"
if [ -z "${SKIP_TEACHER:-}" ]; then
  echo "=== [4/5] Downloading teacher checkpoint (~17 GB, for self-forcing stage) ==="
  mkdir -p "$REPO/checkpoints"
  if [ -f "$TEACHER_DIR/model_ema_bf16.pt" ] && [ -f "$TEACHER_DIR/model/.metadata" ]; then
    echo "  teacher ckpt already at $TEACHER_DIR. Skipping."
  else
    .venv/bin/hf download \
      Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000 \
      --local-dir "$TEACHER_DIR"
    echo "  ✅ teacher ckpt downloaded"
  fi
  echo
else
  echo "=== [4/5] SKIP_TEACHER=1 — teacher ckpt skipped (SF stage will fail without it) ==="
  echo
fi

# ---------------------------------------------------------------- cr1 embedding
CR1_LINK="$REPO/datasets/cr1_empty_string_text_embeddings.pt"
if [ -z "${SKIP_CR1:-}" ]; then
  echo "=== [5/5] Downloading cr1 empty-string text embedding ==="
  if [ -f "$CR1_LINK" ] || [ -L "$CR1_LINK" ]; then
    echo "  cr1 embedding already present at $CR1_LINK. Skipping."
  else
    # Try Cosmos-Predict2.5-2B first (has robot/action-cond/... subfolder)
    echo "  downloading nvidia/Cosmos-Predict2.5-2B (only the cr1 embedding file, ~200 MB) ..."
    if .venv/bin/hf download \
        nvidia/Cosmos-Predict2.5-2B \
        --include "robot/action-cond/cr1_empty_string_text_embeddings.pt" \
        --local-dir /tmp/cosmos25_partial 2>&1 | tail -5; then
      SRC=/tmp/cosmos25_partial/robot/action-cond/cr1_empty_string_text_embeddings.pt
      if [ -f "$SRC" ]; then
        ln -sf "$SRC" "$CR1_LINK"
        echo "  ✅ cr1 symlinked → $SRC"
      else
        echo "  ❌ downloaded but file not at $SRC"; exit 1
      fi
    else
      echo "  ❌ download failed. Repo may be gated — accept license at:"
      echo "     https://huggingface.co/nvidia/Cosmos-Predict2.5-2B"
      echo "     Then re-run this script."
      exit 1
    fi
  fi
  echo
else
  echo "=== [5/5] SKIP_CR1=1 — cr1 embedding skipped (warmup training will fail without it) ==="
  echo
fi

# ---------------------------------------------------------------- verify
if [ -z "${SKIP_VERIFY:-}" ]; then
  echo "=== Verifying installation ==="
  .venv/bin/python local_jobs/verify_dataset.py "$DATASET_DIR" || {
    echo "  ⚠️  verify_dataset.py reported issues; inspect above"
    exit 1
  }
fi

echo
echo "======================================================================"
echo "  ✅ Setup complete. Next step:"
echo "     bash local_jobs/warmup_launch.sh dryrun     # 200-iter sanity check"
echo "     bash local_jobs/warmup_launch.sh full       # full 10k-iter run"
echo "======================================================================"
