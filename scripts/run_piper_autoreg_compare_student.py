"""
Student autoregressive PSNR — same output format as run_piper_autoreg_compare.py
so codex_psnr_curve.py can aggregate teacher + student baselines side by side.

Uses ActionStreamingInference (action_video2world entry) so warmup / self-forcing
DCP checkpoints work.

Example:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_piper_autoreg_compare_student.py \
    --ckpt-path dreamdojo_logs/cosmos_interactive/debug/piper_no_s3_2026-06-30_05-17-04/checkpoints/iter_000004000 \
    --save-dir results/piper_eval_student \
    --dataset-path datasets/piper_insert_mouse_battery_lerobot \
    --index 0 --num-frames 49 --num-steps 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from cosmos_predict2._src.predict2.interactive.inference.action_video2world import (
    ActionStreamingInference,
)
from groot_dreams.dataloader import MultiVideoActionDataset


def write_video(path: Path, video: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.ascontiguousarray(video))
    torchvision.io.write_video(str(path), tensor, fps=fps, video_codec="libx264")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True, help="warmup/self_forcing iter_NNNNNN DCP dir")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--save-fps", type=int, default=10)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--fps", type=int, default=10, help="fps conditioning passed to streaming inference (piper: 10; gr1/g1/agibot/yam: 4)")
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--video-key", default="video.cam_vertical")
    p.add_argument("--experiment", default="cosmos_predict2p5_2B_action_piper_self_forcing_no_s3")
    p.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/interactive/configs/config_distill.py",
    )
    p.add_argument(
        "--cr1-embeddings-path",
        default="datasets/cr1_empty_string_text_embeddings.pt",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-save-npz", action="store_true")
    args = p.parse_args()

    if (args.num_frames - 1) % args.chunk_size != 0:
        raise ValueError("--num-frames must be 1 + chunk_size * N")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Pull ground-truth video + actions from the lerobot dataset ---
    dataset = MultiVideoActionDataset(
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        data_split="train",
        restrict_len=args.index + 1,
        height=args.height,
        width=args.width,
        video_key=args.video_key,
        fps=args.save_fps,
    )
    data = dataset[args.index]
    # data["video"]: [C, T, H, W] in [0, 255] uint8/float
    gt_video = data["video"].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    gt_uint8 = np.clip(np.round(gt_video), 0, 255).astype(np.uint8)
    actions = data["action"][: args.num_frames - 1].numpy()

    # --- 2. Build student streaming inference ---
    infer = ActionStreamingInference(
        config_path=args.config_file,
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path="",
        cr1_embeddings_path=args.cr1_embeddings_path,
        context_parallel_size=1,
        enable_fsdp=False,
        torch_compile=False,
    )

    # Use the GT first frame as the conditioning frame (same as teacher path)
    video = infer.generate_action_streaming(
        video_path=gt_uint8,  # numpy [T,H,W,C] uint8
        actions_np=actions,
        resolution_hw=(args.height, args.width),
        num_steps=args.num_steps,
        seed=args.seed,
        start_frame_idx=0,
        max_frames=args.num_frames,
        fps=args.fps,
    )
    # video: [B, C, T, H, W] in [-1, 1]
    pred_norm = ((video[0] + 1) / 2).clamp(0, 1)
    pred_uint8 = (pred_norm * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

    n_common = min(len(pred_uint8), len(gt_uint8))
    pred_uint8 = pred_uint8[:n_common]
    gt_match = gt_uint8[:n_common]
    merged = np.concatenate([gt_match, pred_uint8], axis=2)

    stem = f"idx_{args.index:04d}_frames_{n_common:04d}"
    np.save(save_dir / f"{stem}_actions.npy", actions)
    frames_npz = ""
    if not args.no_save_npz:
        frames_npz = str(save_dir / f"{stem}_frames.npz")
        np.savez_compressed(frames_npz, pred=pred_uint8, gt=gt_match, merged=merged)
    write_video(save_dir / f"{stem}_pred.mp4", pred_uint8, args.save_fps)
    write_video(save_dir / f"{stem}_gt.mp4", gt_match, args.save_fps)
    write_video(save_dir / f"{stem}_merged.mp4", merged, args.save_fps)

    summary = {
        "index": args.index,
        "num_frames_requested": args.num_frames,
        "num_frames_saved": int(n_common),
        "pred": str(save_dir / f"{stem}_pred.mp4"),
        "gt": str(save_dir / f"{stem}_gt.mp4"),
        "merged": str(save_dir / f"{stem}_merged.mp4"),
        "actions": str(save_dir / f"{stem}_actions.npy"),
        "frames_npz": frames_npz,
        "experiment": args.experiment,
        "ckpt_path": args.ckpt_path,
        "num_steps": args.num_steps,
    }
    with open(save_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
