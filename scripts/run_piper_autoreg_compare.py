import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT
from groot_dreams.dataloader import MultiVideoActionDataset


def write_video(path: Path, video: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.ascontiguousarray(video))
    torchvision.io.write_video(str(path), tensor, fps=fps, video_codec="libx264")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--save-fps", type=int, default=10)
    parser.add_argument("--guidance", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--video-key", default="video.cam_vertical")
    parser.add_argument("--no-save-npz", action="store_true")
    args = parser.parse_args()

    if (args.num_frames - 1) % args.chunk_size != 0:
        raise ValueError("--num-frames must be 1 + chunk_size * N")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = Video2WorldInference(
        experiment_name="dreamdojo_2b_1440_640_piper",
        ckpt_path=args.ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )

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
    gt_video = data["video"].permute(1, 2, 3, 0).cpu().numpy()
    img_array = data["video"].transpose(0, 1)[:1]
    actions = data["action"][: args.num_frames - 1].numpy()
    lam_video = data["lam_video"]

    chunk_video = []
    first_round = True

    for i in range(0, len(actions), args.chunk_size):
        actions_chunk = actions[i : i + args.chunk_size]
        if actions_chunk.shape[0] != args.chunk_size:
            break

        current_lam_video = lam_video[i * 2 : (i + args.chunk_size) * 2]
        if not first_round:
            img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
        else:
            img_tensor = img_array
        first_round = False

        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
            dim=0,
        )
        vid_input = vid_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            video = model.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float(),
                guidance=args.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution="none",
                seed=i,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_steps=args.num_steps,
                lam_video=current_lam_video,
            )

        video_normalized = (video + 1) / 2
        video_clamped = (
            (torch.clamp(video_normalized[0], 0, 1) * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
        img_array = video_clamped[-1]
        chunk_video.append(video_clamped)
        del video, video_normalized
        torch.cuda.empty_cache()

    pred_video = np.concatenate(
        [chunk_video[0]] + [chunk_video[i][: args.chunk_size] for i in range(1, len(chunk_video))],
        axis=0,
    )
    gt_video = gt_video[: len(pred_video)]
    merged_video = np.concatenate([gt_video, pred_video], axis=2)

    stem = f"idx_{args.index:04d}_frames_{len(pred_video):04d}"
    np.save(save_dir / f"{stem}_actions.npy", actions)
    frames_npz = ""
    if not args.no_save_npz:
        frames_npz = str(save_dir / f"{stem}_frames.npz")
        np.savez_compressed(frames_npz, pred=pred_video, gt=gt_video, merged=merged_video)
    write_video(save_dir / f"{stem}_pred.mp4", pred_video, args.save_fps)
    write_video(save_dir / f"{stem}_gt.mp4", gt_video, args.save_fps)
    write_video(save_dir / f"{stem}_merged.mp4", merged_video, args.save_fps)

    summary = {
        "index": args.index,
        "num_frames_requested": args.num_frames,
        "num_frames_saved": int(len(pred_video)),
        "pred": str(save_dir / f"{stem}_pred.mp4"),
        "gt": str(save_dir / f"{stem}_gt.mp4"),
        "merged": str(save_dir / f"{stem}_merged.mp4"),
        "actions": str(save_dir / f"{stem}_actions.npy"),
        "frames_npz": frames_npz,
    }
    with open(save_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    model.cleanup()


if __name__ == "__main__":
    main()
