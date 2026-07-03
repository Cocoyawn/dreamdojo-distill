# Remote Warmup Training

## Directory layout (after setup)

```
dreamdojo-distill/
├── .venv/                                          # python 3.10 + torch 2.7.0+cu128 + deps
├── cosmos_predict2/                                # source (in git)
├── local_jobs/                                     # launch scripts (in git)
├── scripts/                                        # helper scripts (in git)
├── docs/                                           # this document (in git)
├── datasets/
│   ├── cr1_empty_string_text_embeddings.pt         # symlink → HF cache, ~200 MB
│   └── piper_warmup_regenerated_4step/             # 210 GB, 10 000 samples
│       ├── actions/{0..9999}.json                  # (12, 384) action arrays
│       ├── images/{0..9999}.png                    # 1440×640 RGB first frames
│       ├── videos/{0..9999}.mp4                    # 13-frame 1440×640 clips
│       └── latents/{0..9999}.pt                    # dict[int → (16,4,180,80)]
└── checkpoints/
    └── dreamdojo-piper-vertical-1440-640-fps10-iter30000/    # 17 GB teacher ckpt
        ├── model/{__0..7_0.distcp, .metadata}
        └── model_ema_bf16.pt
```

Requirements: 8 × 80 GB GPU (A100/A800/H100), ≥ 450 GB free disk, CUDA 12.8 driver.

---

## Step 1 — clone

```bash
git clone https://github.com/Cocoyawn/dreamdojo-distill.git
cd dreamdojo-distill
```

---

## Step 2 — install python dependencies

```bash
bash install.sh
```

Installs `.venv/` via uv. Takes ~15 minutes. Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.7.0+cu128 True
```

---

## Step 3 — HuggingFace authentication

Accept the license at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B (required for the cr1 embedding file).

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p ~/.cache/huggingface
echo -n "$HF_TOKEN" > ~/.cache/huggingface/token

.venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
# expected: your HF username
```

---

## Step 4 — download dataset (210 GB)

```bash
.venv/bin/hf download \
  Cocoyawn32/dreamdojo-piper-warmup-4step \
  --repo-type=dataset \
  --local-dir datasets/piper_warmup_regenerated_4step

cd datasets/piper_warmup_regenerated_4step
for f in latents.tar images.tar videos.tar actions.tar; do
  tar -xf "$f" && rm "$f"
done
cd ../..
```

Verify:

```bash
ls datasets/piper_warmup_regenerated_4step/{actions,images,videos,latents} | wc -l
# expected: 40000 (10000 × 4)
```

---

## Step 5 — download teacher checkpoint (17 GB, required for self-forcing stage only)

```bash
.venv/bin/hf download \
  Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000 \
  --local-dir checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000
```

Verify:

```bash
ls checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/model/
# expected: .metadata __0_0.distcp __1_0.distcp ... __7_0.distcp
```

---

## Step 6 — download cr1 empty-string embedding

```bash
.venv/bin/hf download \
  nvidia/Cosmos-Predict2.5-2B \
  --include "robot/action-cond/cr1_empty_string_text_embeddings.pt" \
  --local-dir /tmp/cosmos25_partial

ln -sf /tmp/cosmos25_partial/robot/action-cond/cr1_empty_string_text_embeddings.pt \
       datasets/cr1_empty_string_text_embeddings.pt
```

Verify (norm should be ~7000, not 0):

```bash
.venv/bin/python -c "import torch; e=torch.load('datasets/cr1_empty_string_text_embeddings.pt'); print(e.shape, e.norm().item())"
# expected: torch.Size([1, 512, 100352]) 7151.77...
```

---

## Step 7 — verify dataset integrity

```bash
.venv/bin/python local_jobs/verify_dataset.py datasets/piper_warmup_regenerated_4step
```

Last line must be `✅ Dataset verified. You may launch warmup training.`

---

## Step 8 — dry-run (200 iterations, ~15 minutes)

```bash
bash local_jobs/warmup_launch.sh dryrun
```

Confirms training starts correctly and no OOM. Check log output for:

- `sec/iter`: expected 15–30 s (log line `[iter X/200] loss=...`)
- `loss`: should decrease from ~1.0 toward ~0.3 within 200 iterations

If both look correct, proceed.

---

## Step 9 — full training (~2–4 days)

```bash
tmux new -s dd-warmup
bash local_jobs/warmup_launch.sh full 2>&1 | tee logs/warmup_full.log
```

Detach: `Ctrl-b d`. Reattach: `tmux attach -t dd-warmup`.

Default parameters (`local_jobs/warmup_launch.sh full`):

| parameter    | value  |
|--------------|--------|
| NPROC        | 8      |
| BATCH/GPU    | 2      |
| MAX_ITER     | 10000  |
| SAVE_ITER    | 1000   |
| WANDB_MODE   | online |

Overrides via environment variables (example):

```bash
NPROC=4 BATCH=2 MAX_ITER=5000 SAVE_ITER=500 \
  bash local_jobs/warmup_launch.sh full
```

Checkpoints land at:

```
dreamdojo_logs/cosmos_interactive/interactive_warmup/<RUN_NAME>/checkpoints/iter_00000<n>000/
```

10 000 iterations at `save_iter=1000` produces 10 checkpoints (~150 GB total).

---

## Confirming training is running

While training is running:

```bash
# GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Log tail
tail -f logs/warmup_full.log

# Latest checkpoint
ls -t dreamdojo_logs/cosmos_interactive/interactive_warmup/*/checkpoints/ | head -3
```

Expected: all 8 GPUs at 90–100 % utilization; log advances every 15–30 s per iteration; new checkpoint every 1000 iterations.
