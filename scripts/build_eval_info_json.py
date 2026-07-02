#!/usr/bin/env python3
"""
Build datasets/eval/info.json for action_video2world inference.

Pulls N episodes from the piper lerobot dataset, extracts:
  - conditioning frame  → datasets/eval/cond_frame_{i}.png  (or use video path)
  - action sequence     → datasets/eval/action_{i}.npy

Then writes datasets/eval/info.json with entries of the shape expected by
action_video2world._process_entries:
  {"input_video": ..., "input_action": ..., "output_video": ..., "start_frame_idx": 0, "resolution": [1440, 640]}

Usage:
  python scripts/build_eval_info_json.py --n 5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="number of eval entries")
    p.add_argument("--dataset", default="datasets/piper_insert_mouse_battery_lerobot")
    p.add_argument("--out-dir", default="datasets/eval")
    p.add_argument("--video-key", default="video.cam_vertical")
    p.add_argument("--results-dir", default="results/piper_eval_student")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the lerobot dataset to find episode video paths + action sequences.
    from groot_dreams.dataloader import MultiVideoActionDataset

    ds = MultiVideoActionDataset(
        dataset_path=args.dataset,
        num_frames=49,
        data_split="train",
        restrict_len=args.n,
        height=1440,
        width=640,
        video_key=args.video_key,
        fps=10,
    )

    entries = []
    for i in range(args.n):
        sample = ds[i]
        # action array — store as .npy
        action_np = sample["action"].numpy() if hasattr(sample["action"], "numpy") else np.asarray(sample["action"])
        action_path = out_dir / f"action_{i:04d}.npy"
        np.save(action_path, action_np)

        # video path lookup from dataset internals
        info = ds.lerobot_dataset.meta.info
        video_template = info["video_path"]
        ep_idx = int(sample.get("episode_index", i))
        chunk = ep_idx // info.get("chunks_size", 1000)
        video_path = Path(args.dataset) / video_template.format(
            episode_chunk=chunk, video_key=args.video_key, episode_index=ep_idx
        )

        out_video = Path(args.results_dir) / f"idx_{i:04d}_pred.mp4"
        entries.append({
            "input_video": str(video_path),
            "input_action": str(action_path),
            "output_video": str(out_video),
            "start_frame_idx": 0,
            "resolution": [1440, 640],
        })

    json_path = out_dir / "info.json"
    with open(json_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {json_path} with {len(entries)} entries")
    print(f"first entry: {entries[0]}")


if __name__ == "__main__":
    main()
