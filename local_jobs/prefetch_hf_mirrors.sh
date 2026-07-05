#!/bin/bash
# Prefetch all HF assets that teacher_gen / warmup / self-forcing need,
# using Cocoyawn32 mirrors instead of the gated NVIDIA repos.
#
# Populates the local HF cache under $HOME/.cache/huggingface/hub/ so that
# subsequent training runs with HF_HUB_OFFLINE=1 find everything locally.
#
# Usage:
#   bash local_jobs/prefetch_hf_mirrors.sh
#
# Overrides:
#   HF_TARGET_HUB=/path/to/hub   — pre-populate a different cache root
#                                   (default: $HOME/.cache/huggingface/hub)

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD

HF_TARGET_HUB=${HF_TARGET_HUB:-$HOME/.cache/huggingface/hub}
STAGE_DIR=${STAGE_DIR:-/tmp/dd_hf_stage}
mkdir -p "$HF_TARGET_HUB" "$STAGE_DIR"

echo "======================================================================"
echo "  Prefetch HF mirrors → $HF_TARGET_HUB"
echo "======================================================================"

# ---------------- 1. tokenizer.pth (WAN2.1 VAE) --------------------
echo
echo "=== [1/3] nvidia/Cosmos-Predict2.5-2B/tokenizer.pth (~508 MB) ==="
NVIDIA25=$HF_TARGET_HUB/models--nvidia--Cosmos-Predict2.5-2B
COSMOS25_REVISION=f176dc95b4a70f53ce01c4b302851595e7322b00

if [ -f "$NVIDIA25/snapshots/$COSMOS25_REVISION/tokenizer.pth" ]; then
    echo "  already cached, skipping."
else
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download \
      Cocoyawn32/cosmos-predict2p5-cr1-empty-embedding \
      --repo-type=dataset \
      --include "tokenizer.pth" \
      --local-dir "$STAGE_DIR/cr1_mirror"

    mkdir -p "$NVIDIA25/snapshots/$COSMOS25_REVISION" "$NVIDIA25/refs"
    cp "$STAGE_DIR/cr1_mirror/tokenizer.pth" \
       "$NVIDIA25/snapshots/$COSMOS25_REVISION/tokenizer.pth"
    echo -n "$COSMOS25_REVISION" > "$NVIDIA25/refs/main"
    echo "  ✅ tokenizer.pth placed"
fi

# ---------------- 2. Cosmos-Reason1-7B (16 GB) --------------------
echo
echo "=== [2/3] nvidia/Cosmos-Reason1-7B (~16 GB) ==="
REASON1=$HF_TARGET_HUB/models--nvidia--Cosmos-Reason1-7B
REASON_REVISION=3210bec0495fdc7a8d3dbb8d58da5711eab4b423

if [ -f "$REASON1/snapshots/$REASON_REVISION/model-00004-of-00004.safetensors" ]; then
    echo "  already cached, skipping."
else
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download \
      Cocoyawn32/cosmos-reason1-7b-mirror \
      --repo-type=dataset \
      --local-dir "$STAGE_DIR/reason1_mirror"

    mkdir -p "$REASON1/snapshots/$REASON_REVISION" "$REASON1/refs"
    for f in "$STAGE_DIR/reason1_mirror"/*; do
        name=$(basename "$f")
        case "$name" in
            .gitattributes|README.md.orig) continue ;;
        esac
        cp "$f" "$REASON1/snapshots/$REASON_REVISION/$name"
    done
    echo -n "$REASON_REVISION" > "$REASON1/refs/main"
    echo "  ✅ Reason1-7B placed"
fi

# ---------------- 3. CR1 empty-string embedding (datasets/cr1_...) --------------------
echo
echo "=== [3/3] cr1_empty_string_text_embeddings.pt (~100 MB, in datasets/) ==="
CR1_TARGET=$REPO/datasets/cr1_empty_string_text_embeddings.pt

if [ -f "$CR1_TARGET" ] || [ -L "$CR1_TARGET" ]; then
    echo "  already at datasets/cr1_empty_string_text_embeddings.pt, skipping."
else
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download \
      Cocoyawn32/cosmos-predict2p5-cr1-empty-embedding \
      --repo-type=dataset \
      --include "cr1_empty_string_text_embeddings.pt" \
      --local-dir "$STAGE_DIR/cr1_mirror"

    mkdir -p "$REPO/datasets"
    ln -sf "$STAGE_DIR/cr1_mirror/cr1_empty_string_text_embeddings.pt" "$CR1_TARGET"
    echo "  ✅ cr1 embedding symlinked → datasets/cr1_empty_string_text_embeddings.pt"
fi

# ---------------- verify --------------------
echo
echo "=== Verifying ==="
.venv/bin/python <<PY
import os
os.environ["HF_HOME"] = os.path.dirname(os.path.dirname("$HF_TARGET_HUB")) + "/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import hf_hub_download, snapshot_download

# 1. tokenizer.pth
p1 = hf_hub_download("nvidia/Cosmos-Predict2.5-2B", "tokenizer.pth", revision="main", local_files_only=True)
print(f"  ✅ tokenizer.pth        → {p1}  ({os.path.getsize(p1)/1e6:.1f} MB)")

# 2. Reason1-7B
p2 = snapshot_download("nvidia/Cosmos-Reason1-7B", revision="3210bec0495fdc7a8d3dbb8d58da5711eab4b423", local_files_only=True)
print(f"  ✅ Reason1-7B snapshot → {p2}")

# 3. cr1 embedding
import torch
e = torch.load("datasets/cr1_empty_string_text_embeddings.pt", map_location="cpu")
print(f"  ✅ cr1 embedding       → shape={tuple(e.shape)} norm={e.norm().item():.1f}")
PY

echo
echo "======================================================================"
echo "  ✅ All HF caches ready. To run training offline:"
echo
echo "     export DD_HF_CACHE=$(dirname "$HF_TARGET_HUB")"
echo "     export HF_HUB_OFFLINE=1"
echo "     bash local_jobs/teacher_gen_720x320.sh"
echo "======================================================================"
