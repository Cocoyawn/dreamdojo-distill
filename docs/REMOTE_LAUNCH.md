# Remote Warmup Launch — Quickstart

End-to-end steps to bring up DreamDojo Piper warmup training on a fresh remote box, using GitHub for code + HuggingFace for data/ckpts.

## Prerequisites

- Linux box with CUDA 12.8-compatible NVIDIA driver
- **≥ 8 × 80 GB GPUs** (A100/A800/H100) for default configuration; 4 GPUs also works via `NPROC=4`. 40 GB cards likely OOM at 1440×640 latent grid.
- **≥ 450 GB free disk** (210 dataset + 17 teacher + 30 venv + 100 buffer)
- `git`, `curl`, `bash`, `python3.10+` on PATH
- HuggingFace token with **read access to gated `nvidia/Cosmos-Predict2.5-2B`** — accept license at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B first
- (Optional) wandb login for training curves

## Step 1 — clone

```bash
git clone https://github.com/Cocoyawn/dreamdojo-distill.git
cd dreamdojo-distill
```

## Step 2 — one-shot setup

```bash
HF_TOKEN=hf_xxxxxxxxxxxx bash local_jobs/setup_remote.sh
```

This installs deps (`.venv/`, ~15 min), downloads:
- `Cocoyawn32/dreamdojo-piper-warmup-4step` — 210 GB dataset, extracted in place
- `Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000` — 17 GB teacher ckpt (needed for self-forcing stage; downloaded now to avoid a second stall later)
- `cr1_empty_string_text_embeddings.pt` from `nvidia/Cosmos-Predict2.5-2B` — 200 MB, required for warmup

Then runs `verify_dataset.py`. Expected output ends with:
```
✅ Dataset verified. You may launch warmup training.
```

Skip flags (any truthy value): `SKIP_DEPS=1`, `SKIP_DATASET=1`, `SKIP_TEACHER=1`, `SKIP_CR1=1`.

## Step 3 — dry-run (200 iter, ~15 min)

Always run this first — catches OOM and misconfig **before** you commit to 2 days of training.

```bash
bash local_jobs/warmup_launch.sh dryrun
```

Watch for:
- **OOM**: reduce `BATCH=1` (it's already 1 in dryrun). If still OOM: something wrong with model config, not batch. See troubleshooting.
- **`sec/iter`**: aim for ~15–25 s. If ≫ 60 s, storage IO is the bottleneck (check `nvidia-smi dmon` for low util).
- **Loss trend**: should drop from ~1.0 to ~0.3 within 200 iter. If flat, `cr1_empty_string_text_embeddings.pt` may be missing or corrupt.

## Step 4 — full run

If dryrun looked healthy:

```bash
# Backgrounded via tmux so ssh disconnect doesn't kill it
tmux new -d -s dd-warmup 'bash local_jobs/warmup_launch.sh full 2>&1 | tee logs/warmup_full.log'
tmux attach -t dd-warmup       # Ctrl-b d to detach
```

Defaults: **8 GPU × batch 2 × 10 000 iter ≈ 50 h** at 18 s/iter (each iter consumes 16 samples). On 4 GPUs (`NPROC=4`): same wall-clock but half the throughput per iter — effective batch 8 instead of 16.

Checkpoints land in `dreamdojo_logs/cosmos_interactive/interactive_warmup/<RUN_NAME>/checkpoints/iter_00000<n>000/` every 1000 iter.

## Step 5 — publish trained ckpt

When done, upload the best iter (peak of PSNR sweep, or just last) to HF:

```bash
CKPT=dreamdojo_logs/cosmos_interactive/interactive_warmup/<RUN_NAME>/checkpoints/iter_000010000
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload \
  Cocoyawn32/dreamdojo-piper-warmup-v2 "$CKPT" . \
  --repo-type=model \
  --commit-message="warmup v2 (1440×640, cam_vertical, fps=10)"
```

---

## Troubleshooting matrix

| Symptom | Likely cause | Fix |
|---|---|---|
| `HTTPError: 401` during setup | HF token missing or invalid | `export HF_TOKEN=hf_xxx` and re-run |
| `nvidia/Cosmos-Predict2.5-2B` download fails with 403 | Gated model, license not accepted | Visit URL in step 2 error msg, accept, re-run |
| `datasets/cr1_...pt: No such file` at runtime | cr1 symlink target moved | `bash local_jobs/setup_remote.sh SKIP_DEPS=1 SKIP_DATASET=1 SKIP_TEACHER=1` |
| OOM immediately at iter 0 with `BATCH=1` | Activation checkpointing not kicking in | Check `mode="predict2_2b_720_aggressive"` in `exp_action_warmup.py`, must be set on `net.sac_config` |
| Loss flat at ~1.0 forever | cr1 emb load failed silently → all-zero prompt embed | `python -c "import torch; e=torch.load('datasets/cr1_empty_string_text_embeddings.pt'); print(e.shape, e.norm())"` — norm should be ~7000 not 0 |
| `sec/iter` >> 60 s, GPU util < 30 % | Storage IO bottleneck | Move dataset to local NVMe if possible; check `iotop` |
| NCCL timeout during first iter | Warmup usually fine; may indicate slow inter-GPU link | `export NCCL_ASYNC_ERROR_HANDLING=1` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` |
| `tokenizer.pth` download hangs | HF hub timeout | Set `HF_HUB_DOWNLOAD_TIMEOUT=60` and retry |

## Layout expected by training

After successful `setup_remote.sh`, this exact structure must exist:

```
dreamdojo-distill/
├── .venv/
├── cosmos_predict2/
├── local_jobs/
├── scripts/
├── datasets/
│   ├── cr1_empty_string_text_embeddings.pt      # symlink to Cosmos-Predict2.5-2B/robot/action-cond/...
│   └── piper_warmup_regenerated_4step/
│       ├── actions/{0..9999}.json
│       ├── images/{0..9999}.png
│       ├── videos/{0..9999}.mp4
│       └── latents/{0..9999}.pt
└── checkpoints/
    └── dreamdojo-piper-vertical-1440-640-fps10-iter30000/
        ├── model/{__0..7_0.distcp, .metadata}
        └── model_ema_bf16.pt
```

## What warmup does vs self-forcing

**Warmup** (this doc): Trains a from-scratch causal student to regress teacher ODE trajectories. **No teacher weights loaded** during warmup — only teacher-precomputed latents are consumed. Output: a student ckpt that can one-shot denoise from any noise level.

**Self-forcing** (later, not in this doc): Loads the warmup student + teacher ckpt, runs DMD distillation with student autoregressive rollout. See `local_jobs/self_forcing.sh`.

## Config knobs cheat-sheet

Overridable via CLI (`bash local_jobs/warmup_launch.sh full dataloader_train.batch_size=4`) or env:

| Knob | Default | Notes |
|---|---|---|
| `NPROC` | 8 | GPU count. Effective batch = NPROC × batch_size. Override with `NPROC=4 bash ...`. |
| `BATCH` | 2 (full mode) | Physical batch per GPU. Try 4 on 80 GB if headroom permits (memory scales roughly linearly). |
| `MAX_ITER` | 10000 | Paper uses 10k iter × batch 256 = 2.56M sample-iters. Ours is 5% of that. |
| `SAVE_ITER` | 1000 | Ckpt frequency. |
| `WANDB_MODE` | online | Set `offline` or `disabled` if no wandb login. |
| `dataloader_train.num_workers` | (config default) | Bump if seeing IO stalls. |

## Contact & code

- Repo: https://github.com/Cocoyawn/dreamdojo-distill
- Dataset: https://huggingface.co/datasets/Cocoyawn32/dreamdojo-piper-warmup-4step
- Old warmup ckpt (wrist-view, deprecated): https://huggingface.co/Cocoyawn32/dreamdojo-piper-warmup-iter17000
- Paper: [DreamDojo (arXiv:2602.06949)](https://arxiv.org/abs/2602.06949)
