"""
Warmup single-step prediction sanity test.

Goal: verify that the warmup checkpoint can, in ONE forward pass, denoise a
noisy latent back to something visually close to teacher's clean latent.

If this fails → problem is in stage 2 (warmup).
If this passes but multi-step rollout fails → problem is in stage 3 (self-forcing).

Output: side-by-side comparison figure per sample
    row 1: real GT (13 frames from dataset video)
    row 2: teacher clean (VAE decode of ode_latents[:, -1])
    row 3: student pred (VAE decode of warmup model's pred_x0)

Usage:
    CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/test_warmup_single_step.py \
        --ckpt dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000017000 \
        --n-samples 3
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_warmup_model(ckpt_dir: str):
    """Load warmup experiment + ckpt via imaginaire model loader."""
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
    return model, config


def load_dataset():
    from cosmos_predict2._src.predict2.interactive.datasets.dataset_action_warmup import ActionDatasetSFWarmup
    return ActionDatasetSFWarmup(
        data_path="datasets/piper_warmup_regenerated_4step",
        cr1_embeddings_path="datasets/cr1_empty_string_text_embeddings.pt",
        fps=10,
    )


@torch.no_grad()
def decode_latent(model, latent: torch.Tensor) -> np.ndarray:
    """VAE decode: [C, T, H, W] → [T, Ho, Wo, 3] uint8."""
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)  # add batch
    latent = latent.to(dtype=model.tensor_kwargs["dtype"], device=model.tensor_kwargs["device"])
    video = model.decode(latent)  # [B, 3, T, H, W] in [-1, 1]
    video = ((video + 1) / 2).clamp(0, 1)
    video_uint8 = (video[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, 3]
    return video_uint8


@torch.no_grad()
def run_warmup_forward(model, data_batch: dict) -> torch.Tensor:
    """Replicate the core of training_step but with fixed high-noise timestep.

    Returns:
        pred_x0: [B=1, 16, 4, lh, lw] — student's one-shot prediction from max noise
    """
    from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io  # noqa
    from einops import rearrange

    # move to device — tensor fields get unsqueeze(0), scalars→tensor
    # Also cast floating tensors to model dtype (bf16) to match model.
    model_dtype = model.tensor_kwargs["dtype"]
    for k, v in list(data_batch.items()):
        if isinstance(v, torch.Tensor):
            v2 = v.unsqueeze(0).to(model.tensor_kwargs["device"])
            if torch.is_floating_point(v2) and k not in ("video", "input_image"):
                # keep uint8-based video/image raw; cast float action, embeddings, etc.
                v2 = v2.to(dtype=model_dtype)
            data_batch[k] = v2
        elif isinstance(v, (int, float)):
            data_batch[k] = torch.tensor([v], device=model.tensor_kwargs["device"])

    # SAVE original GT video BEFORE conditioner mutates data_batch["video"] in-place
    # (conditioner._forward → _normalize_video_databatch_inplace turns uint8 → float [-1,1])
    original_gt_video = data_batch["video"][0].clone()   # (3, T, H, W) uint8

    condition = model.conditioner(data_batch)
    _, x0, condition = model.get_data_and_condition(data_batch)
    condition = condition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=model.config.min_num_conditional_frames,
        random_max_num_conditional_frames=model.config.max_num_conditional_frames,
    )

    # Build inputs mirroring _prepare_generator_input_output BUT with fixed max-noise timestep
    input_image = data_batch["input_image"].to(**model.tensor_kwargs) / 127.5 - 1.0
    input_image = rearrange(input_image, "b c h w -> b c 1 h w")
    input_latent = model.encode(input_image).contiguous().float()  # (1, 16, 1, lh, lw)

    ode_latents = data_batch["ode_latents"].to(**model.tensor_kwargs)  # (1, 5, 16, lt, lh, lw)
    B, K, C, T, H, W = ode_latents.shape

    # Use HIGH-NOISE latent as input (index 0 in denoising_step_list = t=1000 timestep)
    # This is the hardest case for warmup: student must one-shot denoise from full noise.
    noisy_input = ode_latents[:, 0]  # (1, 16, 4, lh, lw)
    noisy_input[:, :, 0] = input_latent.squeeze(2)  # replace first frame with clean image latent

    # Timestep: t_list[0] corresponds to denoising_step_list[0]=0 which is max noise
    timesteps = torch.full((B, T), float(model.t_list[0]),
                           device=model.tensor_kwargs["device"])

    # Set up net's temporal causal window (mirror training_step)
    h_tokens = int(H) // int(model.net.patch_spatial)
    w_tokens = int(W) // int(model.net.patch_spatial)
    from cosmos_predict2._src.predict2.interactive.networks.utils import make_network_temporal_causal
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
    pred_x0 = noisy_input - ts_norm * velocity_pred  # (1, 16, T, H, W)
    return pred_x0[0], ode_latents[0, -1], original_gt_video  # student pred, teacher clean, ORIGINAL uint8 GT


def make_comparison_figure(samples: list, out_path: Path, n_show_frames: int = 13):
    """samples: list of dicts with keys: gt_video, teacher_video, student_video (all [T,H,W,3] uint8)."""
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples * 3, n_show_frames, figsize=(n_show_frames * 1.6, n_samples * 3 * 2.2))
    if n_samples == 1:
        axes = axes.reshape(3, n_show_frames)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]

    row_labels = ["Real GT", "Teacher clean (VAE decode)", "Student pred (1-step)"]
    for i, sample in enumerate(samples):
        row_data = [sample["gt_video"], sample["teacher_video"], sample["student_video"]]
        for r, (data, label) in enumerate(zip(row_data, row_labels)):
            row = i * 3 + r
            T = min(len(data), n_show_frames)
            step = max(1, T // n_show_frames)
            for c in range(n_show_frames):
                t_idx = min(c * step, T - 1)
                ax = axes[row, c]
                if data.ndim == 4 and data.shape[-1] == 3:
                    ax.imshow(data[t_idx])
                else:
                    ax.imshow(data[t_idx], cmap="gray")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(f"sample {i}\n{label}", fontsize=9)
                if row == 0:
                    ax.set_title(f"t={t_idx}", fontsize=8)

    plt.suptitle("Warmup single-step prediction sanity check",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--sample-indices", type=int, nargs="+", default=None)
    p.add_argument("--out", default="results/warmup_single_step_check")
    args = p.parse_args()

    print(f"Loading warmup model from {args.ckpt} ...")
    model, config = load_warmup_model(args.ckpt)

    print("Loading dataset ...")
    dataset = load_dataset()
    print(f"  {len(dataset)} samples")

    indices = args.sample_indices or list(range(args.n_samples))
    samples_data = []
    for idx in indices:
        print(f"\n>>> sample {idx}")
        item = dataset[idx]
        student_pred_latent, teacher_clean_latent, real_gt_video = run_warmup_forward(model, item)
        print(f"  student pred latent shape: {tuple(student_pred_latent.shape)}")
        print(f"  teacher clean latent shape: {tuple(teacher_clean_latent.shape)}")
        print(f"  real GT video shape: {tuple(real_gt_video.shape)}")

        # decode
        print("  decoding student prediction ...")
        student_video = decode_latent(model, student_pred_latent)
        print("  decoding teacher clean latent ...")
        teacher_video = decode_latent(model, teacher_clean_latent)

        # real GT video was loaded as (3, T, H, W) tensor from dataset — reformat
        if real_gt_video.dim() == 4 and real_gt_video.shape[0] == 3:
            gt_np = real_gt_video.permute(1, 2, 3, 0).float().cpu().numpy()
        else:
            gt_np = real_gt_video.float().cpu().numpy()
        # Clip and cast to uint8 (data may be uint8 already, or float in [0,255], or bf16)
        gt_np = np.clip(gt_np, 0, 255).astype(np.uint8)

        samples_data.append({
            "gt_video": gt_np,
            "teacher_video": teacher_video,
            "student_video": student_video,
        })

        # per-sample PSNR
        min_T = min(len(gt_np), len(student_video), len(teacher_video))
        s = student_video[:min_T].astype(np.float32)
        g = gt_np[:min_T].astype(np.float32)
        t = teacher_video[:min_T].astype(np.float32)
        # resize teacher_video / student_video to gt if h,w mismatch
        if s.shape[1:3] != g.shape[1:3]:
            print(f"  ⚠ shape mismatch: student {s.shape} vs gt {g.shape} — PSNR will be relative")
        else:
            s_psnr = 20*np.log10(255) - 10*np.log10(np.mean((s-g)**2))
            t_psnr = 20*np.log10(255) - 10*np.log10(np.mean((t-g)**2))
            print(f"  student vs GT: PSNR {s_psnr:.2f} dB")
            print(f"  teacher vs GT: PSNR {t_psnr:.2f} dB")

    out_path = Path(args.out) / "single_step_comparison.png"
    make_comparison_figure(samples_data, out_path)

    # also save individual videos
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
    for i, s in enumerate(samples_data):
        outdir = Path(args.out) / f"sample_{i:02d}"
        outdir.mkdir(parents=True, exist_ok=True)
        for name, vid in [("gt", s["gt_video"]), ("teacher", s["teacher_video"]), ("student", s["student_video"])]:
            arr = torch.from_numpy(vid).float() / 127.5 - 1  # [-1, 1]
            arr = arr.permute(3, 0, 1, 2)  # C T H W
            save_img_or_video((arr + 1) / 2, str(outdir / name), fps=10)
    print(f"\nsaved videos under {args.out}")


if __name__ == "__main__":
    main()
