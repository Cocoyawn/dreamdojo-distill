#!/bin/bash
# Bootstrap a fresh remote box to run DreamDojo piper warmup / self-forcing.
#
# What this does:
#   1. Sanity-check env (GPU count, disk, network)
#   2. Install python deps via install.sh (uv + torch + xformers + ...)
#   3. Prompt for HF token, huggingface-cli login
#   4. Download teacher ckpt (default: new 720x320 teacher)
#   5. Download cr1 embedding
#   6. (Optional) Download pre-generated warmup dataset if PIPELINE=1440x640
#   7. Verify everything is in place
#
# NOTE: For the 720x320 pipeline, no pre-generated dataset is downloaded — you
# run `bash local_jobs/teacher_gen_720x320.sh` yourself to regenerate ODE data
# with the new teacher. Only the RAW lerobot data + teacher + cr1 embedding are
# needed to bootstrap.
#
# Usage:
#   bash local_jobs/setup_remote.sh                       # default: 720x320 pipeline
#   PIPELINE=1440x640 bash local_jobs/setup_remote.sh     # old 1440x640 pipeline
#   HF_TOKEN=hf_xxx bash local_jobs/setup_remote.sh       # non-interactive
#
# Skip flags:
#   SKIP_DEPS=1        — don't reinstall .venv
#   SKIP_TEACHER=1     — don't download teacher ckpt
#   SKIP_CR1=1         — don't download cr1 embedding (dangerous — warmup needs it)
#   SKIP_DATASET=1     — don't download pre-generated warmup dataset (720x320: always skipped)
#   SKIP_LEROBOT=1     — don't download raw lerobot data (assumes present at datasets/piper_insert_mouse_battery_lerobot/)

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
PIPELINE=${PIPELINE:-720x320}

case "$PIPELINE" in
  720x320)
    TEACHER_REPO="Shirk6/dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k"
    TEACHER_DIR_NAME="dreamdojo-piper-insert-mouse-battery-720-320-10fps-40k"
    PREGEN_DATASET_REPO=""              # regenerated locally by teacher_gen_720x320.sh
    ;;
  1440x640)
    TEACHER_REPO="Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000"
    TEACHER_DIR_NAME="dreamdojo-piper-vertical-1440-640-fps10-iter30000"
    PREGEN_DATASET_REPO="Cocoyawn32/dreamdojo-piper-warmup-4step"
    ;;
  *)
    echo "Unknown PIPELINE=$PIPELINE. Valid: 720x320, 1440x640"; exit 1;;
esac

echo "======================================================================"
echo "  DreamDojo remote setup"
echo "  pipeline: $PIPELINE"
echo "  teacher : $TEACHER_REPO"
echo "  repo    : $REPO"
echo "  date    : $(date)"
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

# ---------------------------------------------------------------- dataset (pre-generated, only 1440x640)
DATASET_DIR="$REPO/datasets/piper_warmup_regenerated_4step"
if [ "$PIPELINE" = "1440x640" ] && [ -z "${SKIP_DATASET:-}" ]; then
  echo "=== [3/6] Downloading pre-generated dataset (~210 GB) [1440x640 only] ==="
  mkdir -p "$REPO/datasets"

  if [ -d "$DATASET_DIR" ] && [ "$(ls "$DATASET_DIR/actions" 2>/dev/null | wc -l)" = "10000" ]; then
    echo "  dataset already extracted at $DATASET_DIR (10000 samples). Skipping."
  else
    echo "  downloading $PREGEN_DATASET_REPO ..."
    .venv/bin/hf download \
      "$PREGEN_DATASET_REPO" \
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
elif [ "$PIPELINE" = "720x320" ]; then
  echo "=== [3/6] Skipping pre-generated dataset (720x320 pipeline regenerates it locally via teacher_gen_720x320.sh) ==="
  echo
else
  echo "=== [3/6] SKIP_DATASET=1 — skipping dataset download ==="
  echo
fi

# ---------------------------------------------------------------- raw lerobot data (required for teacher_gen)
LEROBOT_DIR="$REPO/datasets/piper_insert_mouse_battery_lerobot"
if [ -z "${SKIP_LEROBOT:-}" ] && [ ! -d "$LEROBOT_DIR" ]; then
  echo "=== [4/6] Raw lerobot data ==="
  echo "  ⚠️  $LEROBOT_DIR not found."
  echo "     Download it manually to that path, or symlink from a shared mount, then re-run."
  echo "     Skipping for now (will not block setup)."
  echo
else
  echo "=== [4/6] Raw lerobot data present at $LEROBOT_DIR ==="
  echo
fi

# ---------------------------------------------------------------- teacher ckpt
TEACHER_DIR="$REPO/checkpoints/$TEACHER_DIR_NAME"
if [ -z "${SKIP_TEACHER:-}" ]; then
  echo "=== [5/6] Downloading teacher checkpoint ==="
  echo "  repo    : $TEACHER_REPO"
  echo "  local   : $TEACHER_DIR"
  mkdir -p "$REPO/checkpoints"
  if [ -f "$TEACHER_DIR/model/.metadata" ]; then
    echo "  teacher ckpt already at $TEACHER_DIR. Skipping."
  else
    .venv/bin/hf download \
      "$TEACHER_REPO" \
      --local-dir "$TEACHER_DIR"
    echo "  ✅ teacher ckpt downloaded"
  fi
  echo
else
  echo "=== [5/6] SKIP_TEACHER=1 — teacher ckpt skipped ==="
  echo
fi

# ---------------------------------------------------------------- cr1 embedding
CR1_LINK="$REPO/datasets/cr1_empty_string_text_embeddings.pt"
if [ -z "${SKIP_CR1:-}" ]; then
  echo "=== [6/6] Downloading cr1 empty-string text embedding ==="
  if [ -f "$CR1_LINK" ] || [ -L "$CR1_LINK" ]; then
    echo "  cr1 embedding already present at $CR1_LINK. Skipping."
  else
    echo "  downloading nvidia/Cosmos-Predict2.5-2B (cr1 embedding file only, ~200 MB) ..."
    if .venv/bin/hf download \
        nvidia/Cosmos-Predict2.5-2B \
        --include "robot/action-cond/cr1_empty_string_text_embeddings.pt" \
        --local-dir /tmp/cosmos25_partial 2>&1 | tail -5; then
      SRC=/tmp/cosmos25_partial/robot/action-cond/cr1_empty_string_text_embeddings.pt
      if [ -f "$SRC" ]; then
        mkdir -p "$REPO/datasets"
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
  echo "=== [6/6] SKIP_CR1=1 — cr1 embedding skipped (warmup training will fail without it) ==="
  echo
fi

# ---------------------------------------------------------------- verify (only for 1440x640 pipeline with pre-generated dataset)
if [ "$PIPELINE" = "1440x640" ] && [ -z "${SKIP_VERIFY:-}" ]; then
  echo "=== Verifying installation ==="
  .venv/bin/python local_jobs/verify_dataset.py "$DATASET_DIR" || {
    echo "  ⚠️  verify_dataset.py reported issues; inspect above"
    exit 1
  }
fi

echo
echo "======================================================================"
echo "  ✅ Setup complete. Pipeline: $PIPELINE"
echo
if [ "$PIPELINE" = "720x320" ]; then
  echo "  Next steps:"
  echo "    1. Ensure raw lerobot data is at datasets/piper_insert_mouse_battery_lerobot/"
  echo "    2. bash local_jobs/teacher_gen_720x320.sh       # regenerate ODE dataset"
  echo "    3. bash local_jobs/warmup_720x320.sh            # warmup training"
  echo "    4. WARMUP_CKPT=... bash local_jobs/self_forcing_720x320.sh   # SF distillation"
else
  echo "  Next steps:"
  echo "    1. bash local_jobs/warmup_launch.sh dryrun      # 200-iter sanity check"
  echo "    2. bash local_jobs/warmup_launch.sh full        # full 10k-iter run"
fi
echo "======================================================================"
