"""Sanity-check a downloaded piper_warmup_regenerated_4step dataset.

Confirms:
  - 10 000 samples in each of actions/, images/, videos/, latents/
  - shape checks on sample 0 and sample 9999 (start + end of range)
  - 3-view stack layout (top ≈ cam_high, mid ≈ cam_left_wrist, bot ≈ cam_right_wrist)
    if raw lerobot is also available; otherwise just verify shapes
  - cr1 embedding file present

Usage:
    python local_jobs/verify_dataset.py [/path/to/piper_warmup_regenerated_4step]
"""
import json
import os
import sys
from pathlib import Path


def check_shape(name, got, want):
    ok = tuple(got) == tuple(want)
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name} shape = {tuple(got)}  (want {tuple(want)})")
    return ok


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/piper_warmup_regenerated_4step")
    if not root.exists():
        print(f"❌ dataset dir not found: {root}")
        return 1

    print(f"=== dataset root: {root} ===\n")

    # ------------------------------------------------ file counts
    all_ok = True
    for sub, expect in [("actions", 10000), ("images", 10000), ("videos", 10000), ("latents", 10000)]:
        n = len(list((root / sub).glob("*"))) if (root / sub).is_dir() else 0
        ok = n == expect
        all_ok &= ok
        print(f"  {'✅' if ok else '❌'} {sub}/  = {n} files  (want {expect})")

    if not all_ok:
        print("\n⚠️  file counts mismatch. Did tar extraction complete?")
        return 1

    # ------------------------------------------------ shape checks (idx 0 and 9999)
    import torch
    import numpy as np
    import decord

    for idx in (0, 9999):
        print(f"\n--- sample idx = {idx} ---")

        # video
        try:
            vr = decord.VideoReader(str(root / "videos" / f"{idx}.mp4"))
            frames = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, C)
            all_ok &= check_shape("video (T, H, W, C)", frames.shape, (13, 1440, 640, 3))
        except Exception as e:
            print(f"  ❌ videos/{idx}.mp4 unreadable: {e}"); all_ok = False

        # latent
        try:
            lat = torch.load(root / "latents" / f"{idx}.pt", map_location="cpu")
            keys_ok = sorted(lat.keys()) == [0, 9, 18, 27, 34, 35]
            print(f"  {'✅' if keys_ok else '❌'} latent keys = {sorted(lat.keys())}  (want [0, 9, 18, 27, 34, 35])")
            all_ok &= keys_ok
            first_key = sorted(lat.keys())[0]
            all_ok &= check_shape("latent[0] (C, T, H, W)", lat[first_key].shape, (16, 4, 180, 80))
        except Exception as e:
            print(f"  ❌ latents/{idx}.pt unreadable: {e}"); all_ok = False

        # action
        try:
            act = json.load(open(root / "actions" / f"{idx}.json"))
            act_np = np.array(act)
            all_ok &= check_shape("action", act_np.shape, (12, 384))
        except Exception as e:
            print(f"  ❌ actions/{idx}.json unreadable: {e}"); all_ok = False

    # ------------------------------------------------ cr1 embedding
    print("\n--- cr1 empty-string embedding ---")
    cr1 = Path("datasets/cr1_empty_string_text_embeddings.pt")
    if not cr1.exists():
        print(f"  ⚠️  {cr1} not found — warmup training will fail. Rerun setup_remote.sh (or re-verify path).")
        all_ok = False
    else:
        emb = torch.load(cr1, map_location="cpu")
        all_ok &= check_shape("cr1 embedding", emb.shape, (1, 512, 100352))

    # ------------------------------------------------ 3-view sanity (optional)
    lerobot = Path("datasets/piper_insert_mouse_battery_lerobot")
    if lerobot.exists():
        print("\n--- 3-view stack sanity (vs raw lerobot) ---")
        try:
            vr = decord.VideoReader(str(root / "videos" / "0.mp4"))
            f0 = vr.get_batch([0]).asnumpy()[0]  # (1440, 640, 3)
            base = lerobot / "videos" / "chunk-000"
            for name, sl, cam in [
                ("top", f0[:480],   "observation.images.cam_high"),
                ("mid", f0[480:960], "observation.images.cam_left_wrist"),
                ("bot", f0[960:1440], "observation.images.cam_right_wrist"),
            ]:
                ref = decord.VideoReader(str(base / cam / "episode_000000.mp4")).get_batch([0]).asnumpy()[0]
                d = np.abs(sl.astype(int) - ref.astype(int)).mean()
                ok = d < 5
                all_ok &= ok
                print(f"  {'✅' if ok else '❌'} {name} vs {cam}: mean |diff| = {d:.2f}  (want < 5, h264 noise floor)")
        except Exception as e:
            print(f"  ⚠️  could not run 3-view sanity: {e}")
    else:
        print("\n  (skipping 3-view sanity — datasets/piper_insert_mouse_battery_lerobot/ not present)")

    print()
    if all_ok:
        print("======================================================================")
        print("  ✅ Dataset verified. You may launch warmup training.")
        print("======================================================================")
        return 0
    else:
        print("======================================================================")
        print("  ❌ Verification failed. See errors above.")
        print("======================================================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
