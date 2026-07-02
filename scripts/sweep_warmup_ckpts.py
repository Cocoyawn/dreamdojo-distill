"""
Sweep warmup single-step PSNR across multiple checkpoints.

For each ckpt (iter_1000..iter_17000):
  - load model
  - run one-shot prediction on N sample from warmup dataset
  - compute PSNR against real GT video (VAE decode of pred_x0)
  - also compute teacher_clean baseline PSNR (upper bound)

Output:
  results/warmup_ckpt_sweep/
    ├── psnr_by_iter.json   {iter: {student_mean, teacher_mean, per_sample: [...]}}
    ├── psnr_by_iter.csv
    └── psnr_curve.png

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/sweep_warmup_ckpts.py \
      --ckpt-dir dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints \
      --n-samples 3
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from einops import rearrange


def load_warmup_model(ckpt_dir: str, prev_model=None):
    """Load warmup experiment + ckpt. Reuse prev_model if given (skip re-init) — currently just reloads.
    Note: DCP loader does model init from config each time; we can't easily hot-swap ckpt.
    Kept for future optimization.
    """
    from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint
    model, config = load_model_from_checkpoint(
        experiment_name="cosmos_predict2p5_2B_action_piper_warmup_no_s3",
        s3_checkpoint_dir=ckpt_dir,
        config_file="cosmos_predict2/_src/predict2/interactive/configs/config_warmup.py",
        load_ema_to_reg=False,
        experiment_opts=["ckpt_type=dcp"],
        skip_teacher_init=True,
    )
    model.eval()
    return model


def load_dataset():
    from cosmos_predict2._src.predict2.interactive.datasets.dataset_action_warmup import ActionDatasetSFWarmup
    return ActionDatasetSFWarmup(
        data_path="datasets/piper_warmup_regenerated_4step",
        cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
        fps=10,
    )


@torch.no_grad()
def decode_latent(model, latent: torch.Tensor) -> np.ndarray:
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(dtype=model.tensor_kwargs["dtype"], device=model.tensor_kwargs["device"])
    video = model.decode(latent)
    video = ((video + 1) / 2).clamp(0, 1)
    return (video[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


@torch.no_grad()
def run_warmup_forward(model, data_batch: dict):
    from cosmos_predict2._src.predict2.interactive.networks.utils import make_network_temporal_causal

    model_dtype = model.tensor_kwargs["dtype"]
    for k, v in list(data_batch.items()):
        if isinstance(v, torch.Tensor):
            v2 = v.unsqueeze(0).to(model.tensor_kwargs["device"])
            if torch.is_floating_point(v2) and k not in ("video", "input_image"):
                v2 = v2.to(dtype=model_dtype)
            data_batch[k] = v2
        elif isinstance(v, (int, float)):
            data_batch[k] = torch.tensor([v], device=model.tensor_kwargs["device"])

    original_gt_video = data_batch["video"][0].clone()

    condition = model.conditioner(data_batch)
    _, x0, condition = model.get_data_and_condition(data_batch)
    condition = condition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=model.config.min_num_conditional_frames,
        random_max_num_conditional_frames=model.config.max_num_conditional_frames,
    )

    input_image = data_batch["input_image"].to(**model.tensor_kwargs) / 127.5 - 1.0
    input_image = rearrange(input_image, "b c h w -> b c 1 h w")
    input_latent = model.encode(input_image).contiguous().float()

    ode_latents = data_batch["ode_latents"].to(**model.tensor_kwargs)
    B, K, C, T, H, W = ode_latents.shape
    noisy_input = ode_latents[:, 0]
    noisy_input[:, :, 0] = input_latent.squeeze(2)

    timesteps = torch.full((B, T), float(model.t_list[0]), device=model.tensor_kwargs["device"])

    h_tokens = int(H) // int(model.net.patch_spatial)
    w_tokens = int(W) // int(model.net.patch_spatial)
    make_network_temporal_causal(
        model.net, int(h_tokens), int(w_tokens),
        window_size=(model.config.cache_frame_size, -1, -1),
    )

    velocity_pred = model.net(
        x_B_C_T_H_W=noisy_input.to(**model.tensor_kwargs),
        timesteps_B_T=timesteps,
        **condition.to_dict(),
    ).float()

    ts_norm = timesteps.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) / 1000.0
    pred_x0 = noisy_input - ts_norm * velocity_pred
    return pred_x0[0], ode_latents[0, -1], original_gt_video


def compute_psnr(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    n = min(len(pred_uint8), len(gt_uint8))
    p = pred_uint8[:n].astype(np.float32)
    g = gt_uint8[:n].astype(np.float32)
    if p.shape != g.shape:
        return float("nan")
    mse = np.mean((p - g) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))


def sweep(ckpt_paths: list[str], sample_indices: list[int], out_dir: Path):
    from cosmos_predict2._src.imaginaire.utils import log
    print(f"Loading dataset ...")
    dataset = load_dataset()

    results = {}   # {iter_num: {'student': [...], 'teacher': [...]}}
    for ckpt_path in ckpt_paths:
        m = re.search(r"iter_(\d+)", ckpt_path)
        if not m:
            print(f"  skip (no iter number): {ckpt_path}")
            continue
        iter_num = int(m.group(1))
        print(f"\n{'='*70}\n[iter {iter_num}] loading model from {ckpt_path}\n{'='*70}")

        model = load_warmup_model(ckpt_path)

        stud_psnrs = []
        teach_psnrs = []
        for idx in sample_indices:
            item = dataset[idx]
            student_pred_latent, teacher_clean_latent, real_gt_video = run_warmup_forward(model, item)

            student_video = decode_latent(model, student_pred_latent)
            teacher_video = decode_latent(model, teacher_clean_latent)

            if real_gt_video.dim() == 4 and real_gt_video.shape[0] == 3:
                gt_np = real_gt_video.permute(1, 2, 3, 0).float().cpu().numpy()
            else:
                gt_np = real_gt_video.float().cpu().numpy()
            gt_np = np.clip(gt_np, 0, 255).astype(np.uint8)

            s_psnr = compute_psnr(student_video, gt_np)
            t_psnr = compute_psnr(teacher_video, gt_np)
            stud_psnrs.append(s_psnr)
            teach_psnrs.append(t_psnr)
            print(f"  sample {idx}:  student={s_psnr:6.2f} dB    teacher={t_psnr:6.2f} dB")

        results[iter_num] = {
            "student": stud_psnrs,
            "teacher": teach_psnrs,
            "student_mean": float(np.mean(stud_psnrs)),
            "teacher_mean": float(np.mean(teach_psnrs)),
        }
        print(f"  → mean student={results[iter_num]['student_mean']:.2f}  teacher={results[iter_num]['teacher_mean']:.2f}")

        # free model before next
        del model
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "psnr_by_iter.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "psnr_by_iter.csv").open("w") as f:
        f.write("iter,student_mean,teacher_mean\n")
        for k in sorted(results.keys()):
            f.write(f"{k},{results[k]['student_mean']:.4f},{results[k]['teacher_mean']:.4f}\n")
    print(f"\nsaved {out_dir}/psnr_by_iter.{{json,csv}}")

    plot_curve(results, out_dir / "psnr_curve.png")
    return results


def plot_curve(results: dict, out_path: Path):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.linewidth": 1.1,
    })
    PALETTE = {"student": "#C82423", "teacher": "#2878B5", "student_band": "#F8AC8C"}

    iters = sorted(results.keys())
    stud = np.array([np.mean(results[i]["student"]) for i in iters])
    teach = np.array([np.mean(results[i]["teacher"]) for i in iters])
    stud_std = np.array([np.std(results[i]["student"]) for i in iters])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.fill_between(iters, stud - stud_std, stud + stud_std,
                    color=PALETTE["student_band"], alpha=0.30, linewidth=0,
                    label=r"Student $\pm 1\sigma$")
    ax.plot(iters, stud, color=PALETTE["student"], lw=2.4, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.4,
            label="Student (single-step pred, warmup ckpt)")
    ax.plot(iters, teach, color=PALETTE["teacher"], lw=2.0, marker="s", markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, ls="--",
            label="Teacher clean (VAE decode)  — upper bound")

    ax.set_xlabel("Warmup iteration")
    ax.set_ylabel("Single-step prediction PSNR vs. real GT (dB)")
    ax.set_title("Warmup checkpoint sweep — single-step reconstruction quality")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", framealpha=0.95)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out_path} and {out_path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--iters", type=int, nargs="+", default=None,
                   help="specific iter numbers; else sweep all iter_* dirs")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--sample-indices", type=int, nargs="+", default=None)
    p.add_argument("--out", default="results/warmup_ckpt_sweep")
    args = p.parse_args()

    ckpt_pattern = os.path.join(args.ckpt_dir, "iter_*")
    all_ckpts = sorted(glob.glob(ckpt_pattern))
    if args.iters:
        wanted = set(f"iter_{i:09d}" for i in args.iters)
        ckpts = [c for c in all_ckpts if os.path.basename(c) in wanted]
    else:
        ckpts = all_ckpts

    print(f"will sweep {len(ckpts)} ckpts:")
    for c in ckpts:
        print(f"  {c}")

    sample_indices = args.sample_indices or list(range(args.n_samples))
    print(f"sample indices: {sample_indices}")

    results = sweep(ckpts, sample_indices, Path(args.out))

    print("\n" + "="*70)
    print("FINAL SUMMARY (mean PSNR per iter)")
    print("="*70)
    print(f"{'iter':>10}  {'student':>10}  {'teacher':>10}  {'gap':>8}")
    for i in sorted(results.keys()):
        r = results[i]
        gap = r['student_mean'] - r['teacher_mean']
        print(f"{i:>10}  {r['student_mean']:>10.2f}  {r['teacher_mean']:>10.2f}  {gap:>+8.2f}")


if __name__ == "__main__":
    main()
