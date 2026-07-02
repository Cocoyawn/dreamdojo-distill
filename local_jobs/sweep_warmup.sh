#!/bin/bash
# Loop over all warmup ckpts, run test_one_ckpt.py in a fresh Python process each time.
# CUDA memory fully released between iterations.

set -u
cd "$(dirname "$0")/.."
source local_jobs/_common.sh

CKPT_DIR=${CKPT_DIR:-dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints}
OUT_DIR=${OUT_DIR:-results/warmup_ckpt_sweep}
GPU=${GPU:-5}
N_SAMPLES=${N_SAMPLES:-3}

mkdir -p "$OUT_DIR"
LOG="$LOG_DIR/warmup_sweep_wrapper_$(date +%Y%m%d_%H%M%S).log"
echo "=== sweeping warmup ckpts on GPU $GPU, N_SAMPLES=$N_SAMPLES ===" | tee "$LOG"

for CKPT in "$CKPT_DIR"/iter_*/; do
    ITER=$(basename "$CKPT" | sed 's/iter_0*//')
    OUT_JSON="$OUT_DIR/iter_$(printf '%05d' $ITER).json"

    if [ -f "$OUT_JSON" ]; then
        echo ">>> iter $ITER already done, skipping" | tee -a "$LOG"
        continue
    fi

    echo | tee -a "$LOG"
    echo "======================================================================" | tee -a "$LOG"
    echo ">>> iter $ITER  ckpt=$CKPT" | tee -a "$LOG"
    echo "======================================================================" | tee -a "$LOG"

    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/test_one_ckpt.py \
        --ckpt "$CKPT" \
        --n-samples $N_SAMPLES \
        --out "$OUT_JSON" 2>&1 | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "=== all done ===" | tee -a "$LOG"
echo "aggregating ..." | tee -a "$LOG"
.venv/bin/python scripts/plot_warmup_sweep.py --dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
