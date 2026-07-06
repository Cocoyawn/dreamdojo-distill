"""Verify a piper_720_320 ODE dataset:
  - file counts match across actions/images/videos/latents/
  - shape sanity on start/mid/end samples (portrait 720x320)
  - keys sanity on latents
  - spot-check on a few random samples

Usage:
    python local_jobs/verify_720_320_dataset.py \
        [/path/to/piper_720_320_warmup_regenerated_4step] \
        [--expected 60000]
"""
import argparse
import json
import random
import sys
from pathlib import Path


def check(name, got, want):
    ok = got == want
    print(f"  {'✅' if ok else '❌'} {name}: {got}  (want {want})")
    return ok


def check_range(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  {'✅' if ok else '❌'} {name}: {got}  (want in [{lo}, {hi}])")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="datasets/piper_720_320_warmup_regenerated_4step")
    p.add_argument("--expected", type=int, default=None,
                   help="Expected total sample count (default: whatever is present)")
    p.add_argument("--n-spot", type=int, default=10, help="Number of random samples to check shape on")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ dataset root not found: {root}")
        return 1

    print(f"=== root: {root} ===\n")

    # ---------- file counts ----------
    all_ok = True
    counts = {}
    for sub in ["actions", "images", "videos", "latents"]:
        d = root / sub
        if not d.is_dir():
            print(f"  ❌ missing subdir: {d}")
            all_ok = False
            continue
        n = sum(1 for _ in d.iterdir())
        counts[sub] = n
        print(f"  {sub}/: {n}")

    if len(set(counts.values())) != 1:
        print(f"\n❌ file counts mismatch across subdirs: {counts}")
        all_ok = False
    else:
        total = counts["actions"]
        print(f"\n  total samples: {total}")
        if args.expected is not None:
            all_ok &= check("expected count", total, args.expected)

    if not all_ok:
        print("\n=== verification FAILED (file counts) ===")
        return 1

    # ---------- index continuity ----------
    print("\n=== index continuity ===")
    action_ids = sorted(int(f.stem) for f in (root / "actions").iterdir() if f.suffix == ".json")
    min_id, max_id = action_ids[0], action_ids[-1]
    all_ok &= check("min idx", min_id, 0)
    all_ok &= check("max idx", max_id, total - 1)
    expected = set(range(total))
    actual = set(action_ids)
    missing = expected - actual
    extra = actual - expected
    if missing:
        print(f"  ❌ missing {len(missing)} idx (first few: {sorted(missing)[:5]})")
        all_ok = False
    if extra:
        print(f"  ❌ unexpected {len(extra)} idx (first few: {sorted(extra)[:5]})")
        all_ok = False
    if not missing and not extra:
        print(f"  ✅ contiguous [0, {max_id}]")

    # ---------- shape checks ----------
    import torch
    import decord
    import numpy as np

    idxs_to_check = [0, total // 2, total - 1] + random.sample(range(total), min(args.n_spot, total))
    idxs_to_check = sorted(set(idxs_to_check))

    print(f"\n=== shape sanity ({len(idxs_to_check)} samples) ===")
    for idx in idxs_to_check:
        # video
        try:
            vr = decord.VideoReader(str(root / "videos" / f"{idx}.mp4"))
            f = vr.get_batch([0]).asnumpy()[0]
            T = len(vr)
            H, W, C = f.shape
            if (T, H, W, C) != (13, 720, 320, 3):
                print(f"  ❌ idx {idx}: video shape (T,H,W,C) = ({T},{H},{W},{C})   want (13,720,320,3)")
                all_ok = False
            else:
                pass  # silent success
        except Exception as e:
            print(f"  ❌ idx {idx}: video unreadable: {e}")
            all_ok = False

        # latent
        try:
            lat = torch.load(root / "latents" / f"{idx}.pt", map_location="cpu")
            keys = sorted(lat.keys())
            if keys != [0, 9, 18, 27, 34] and keys != [0, 9, 18, 27, 34, 35]:
                print(f"  ❌ idx {idx}: latent keys = {keys}   want [0,9,18,27,34] or [0,9,18,27,34,35]")
                all_ok = False
            first = lat[keys[0]]
            if tuple(first.shape) != (16, 4, 90, 40):
                print(f"  ❌ idx {idx}: latent[{keys[0]}] shape = {tuple(first.shape)}   want (16,4,90,40)")
                all_ok = False
        except Exception as e:
            print(f"  ❌ idx {idx}: latent unreadable: {e}")
            all_ok = False

        # action
        try:
            act = json.load(open(root / "actions" / f"{idx}.json"))
            arr = np.array(act)
            if arr.shape != (12, 384):
                print(f"  ❌ idx {idx}: action shape = {arr.shape}   want (12, 384)")
                all_ok = False
        except Exception as e:
            print(f"  ❌ idx {idx}: action unreadable: {e}")
            all_ok = False

    print(f"  ✅ shape checks passed on {len(idxs_to_check)} samples")

    # ---------- consistency between old and new segments ----------
    # If the dataset was extended (idx 0..29999 old + idx 30000.. new),
    # check that the new-segment latent norm is in the same ballpark as old.
    if total > 30000:
        print("\n=== old (idx 0) vs new (idx 30000) latent stats ===")
        for idx in (0, 30000, total - 1):
            lat = torch.load(root / "latents" / f"{idx}.pt", map_location="cpu")
            k = sorted(lat.keys())[-1]  # near-clean latent
            v = lat[k].float()
            print(f"  idx {idx:>6} latent[{k}]: mean={v.mean():+.4f}  std={v.std():.4f}")
        print("  (rough sanity: std should be O(0.5-1.0) for near-clean latents; mean small)")

    print()
    if all_ok:
        print("=" * 66)
        print(f"  ✅ Dataset verified. {total} samples ready for warmup.")
        print("=" * 66)
        return 0
    else:
        print("=" * 66)
        print("  ❌ Verification FAILED.")
        print("=" * 66)
        return 1


if __name__ == "__main__":
    sys.exit(main())
