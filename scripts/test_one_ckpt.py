"""
Test a SINGLE warmup ckpt: load, run N sample single-step predictions, dump PSNR JSON, exit.
Designed to be called in a loop from bash — each run is a fresh Python process
so CUDA memory is fully released between checkpoints.

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/test_one_ckpt.py \
      --ckpt <path/to/iter_XXXXXXXXX> \
      --n-samples 3 \
      --out results/warmup_ckpt_sweep/iter_XXXXX.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


def load_warmup_model(ckpt_dir: str):
    from cosmos_predict2._src.predict2.distill.utils.model_loader import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(
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
def decode_latent(model, latent):
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    latent = latent.to(dtype=model.tensor_kwargs["dtype"], device=model.tensor_kwargs["device"])
    video = model.decode(latent)
    video = ((video + 1) / 2).clamp(0, 1)
    return (video[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()


@torch.no_grad()
def run_forward(model, data_batch):
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

    original_gt = data_batch["video"][0].clone()

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
    return pred_x0[0], ode_latents[0, -1], original_gt


def compute_psnr(pred, gt):
    n = min(len(pred), len(gt))
    p = pred[:n].astype(np.float32)
    g = gt[:n].astype(np.float32)
    if p.shape != g.shape:
        return float("nan")
    mse = float(np.mean((p - g) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--sample-indices", type=int, nargs="+", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[LOAD] {args.ckpt}", flush=True)
    model = load_warmup_model(args.ckpt)
    dataset = load_dataset()

    indices = args.sample_indices or list(range(args.n_samples))
    stud, teach = [], []
    for idx in indices:
        item = dataset[idx]
        pred_lat, teacher_lat, gt_v = run_forward(model, item)
        stud_v = decode_latent(model, pred_lat)
        teach_v = decode_latent(model, teacher_lat)
        if gt_v.dim() == 4 and gt_v.shape[0] == 3:
            gt_np = gt_v.permute(1, 2, 3, 0).float().cpu().numpy()
        else:
            gt_np = gt_v.float().cpu().numpy()
        gt_np = np.clip(gt_np, 0, 255).astype(np.uint8)
        s_psnr = compute_psnr(stud_v, gt_np)
        t_psnr = compute_psnr(teach_v, gt_np)
        stud.append(s_psnr)
        teach.append(t_psnr)
        print(f"  sample {idx}: student={s_psnr:6.2f}  teacher={t_psnr:6.2f}", flush=True)

    result = {
        "ckpt": args.ckpt,
        "sample_indices": indices,
        "student": stud,
        "teacher": teach,
        "student_mean": float(np.mean(stud)),
        "teacher_mean": float(np.mean(teach)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[DONE] mean student={result['student_mean']:.2f}  teacher={result['teacher_mean']:.2f}", flush=True)
    print(f"       saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
