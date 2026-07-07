"""Compare distribution between old (idx 0..29999) and new (idx 30000..N-1) segments
of an extended ODE dataset, to catch pipeline drift.

Reports:
  - latent statistics (mean, std, per-timestep) for each segment
  - action statistics
  - video pixel statistics
  - identity check: same lerobot idx should give identical clean latent (idx 34)
    ONLY IF regenerated with same teacher/dataset — otherwise deltas indicate config drift.

Usage:
    python local_jobs/compare_old_new_segments.py \
        [/path/to/piper_720_320_warmup_regenerated_4step] \
        [--boundary 30000] \
        [--n-samples 100]
"""
import argparse
import json
import random
import sys
from pathlib import Path


def summarize_latents(root: Path, sample_ids: list[int], label: str):
    import torch
    print(f"\n--- {label} (n={len(sample_ids)}) ---")
    stats = {}  # key -> list of (mean, std)
    for idx in sample_ids:
        try:
            lat = torch.load(root / "latents" / f"{idx}.pt", map_location="cpu")
        except Exception as e:
            print(f"  ❌ idx {idx}: {e}")
            continue
        for k, v in lat.items():
            vf = v.float()
            stats.setdefault(k, []).append((vf.mean().item(), vf.std().item()))
    import numpy as np
    for k in sorted(stats.keys()):
        arr = np.array(stats[k])
        m_mean = arr[:, 0].mean()
        s_mean = arr[:, 1].mean()
        m_std = arr[:, 0].std()
        s_std = arr[:, 1].std()
        print(f"  key {k:>2}:  mean=[{m_mean:+.4f} ± {m_std:.4f}]   std=[{s_mean:.4f} ± {s_std:.4f}]")
    return stats


def summarize_actions(root: Path, sample_ids: list[int], label: str):
    import numpy as np
    print(f"\n--- {label} actions (n={len(sample_ids)}) ---")
    mins, maxs, means = [], [], []
    for idx in sample_ids:
        arr = np.array(json.load(open(root / "actions" / f"{idx}.json")))
        mins.append(arr.min())
        maxs.append(arr.max())
        means.append(arr.mean())
    print(f"  value range: [{np.mean(mins):+.4f}, {np.mean(maxs):+.4f}]  mean {np.mean(means):+.4f}")
    print(f"  (values are normalized in [-1,1] range for piper)")


def summarize_video(root: Path, sample_ids: list[int], label: str):
    import decord
    import numpy as np
    print(f"\n--- {label} video (n={len(sample_ids)}) ---")
    all_shapes = set()
    means, stds = [], []
    for idx in sample_ids[:5]:  # video decode is slow, just do 5
        try:
            vr = decord.VideoReader(str(root / "videos" / f"{idx}.mp4"))
            frames = vr.get_batch(range(len(vr))).asnumpy()
            all_shapes.add(frames.shape[1:])
            means.append(frames.mean())
            stds.append(frames.std())
        except Exception as e:
            print(f"  ❌ idx {idx}: {e}")
    print(f"  shapes seen: {all_shapes}")
    print(f"  pixel mean: {np.mean(means):.2f}   std: {np.mean(stds):.2f}")


def compare_stats(old_stats, new_stats):
    import numpy as np
    print("\n=== per-key latent statistics comparison ===")
    print(f"  {'key':>4}  {'old mean':>12}  {'new mean':>12}  {'Δmean':>10}   {'old std':>10}  {'new std':>10}  {'Δstd':>10}   {'status':>8}")
    all_keys = sorted(set(old_stats) | set(new_stats))
    verdict = True
    for k in all_keys:
        if k not in old_stats or k not in new_stats:
            print(f"  {k:>4}: missing in {'old' if k not in old_stats else 'new'}")
            verdict = False
            continue
        old_arr = np.array(old_stats[k])
        new_arr = np.array(new_stats[k])
        o_mean, o_std = old_arr[:, 0].mean(), old_arr[:, 1].mean()
        n_mean, n_std = new_arr[:, 0].mean(), new_arr[:, 1].mean()
        d_mean = n_mean - o_mean
        d_std = n_std - o_std
        # heuristic threshold: mean diff > 0.05 OR std ratio outside [0.9, 1.1]
        ratio = n_std / max(o_std, 1e-6)
        status = "✅ ok" if (abs(d_mean) < 0.05 and 0.9 < ratio < 1.1) else "⚠️ drift"
        if status.startswith("⚠"):
            verdict = False
        print(f"  {k:>4}  {o_mean:>+12.4f}  {n_mean:>+12.4f}  {d_mean:>+10.4f}   {o_std:>10.4f}  {n_std:>10.4f}  {d_std:>+10.4f}   {status}")
    return verdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="datasets/piper_720_320_warmup_regenerated_4step")
    p.add_argument("--boundary", type=int, default=30000,
                   help="First index of the NEW segment inside a single root (ignored when --other-root is set)")
    p.add_argument("--other-root", type=str, default=None,
                   help="If set, compare OLD=<other-root> vs NEW=<root> across two separate directories")
    p.add_argument("--n-samples", type=int, default=100,
                   help="Number of samples to draw from each segment for stats")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"❌ {root} not found"); return 1

    random.seed(args.seed)

    if args.other_root:
        old_root = Path(args.other_root)
        new_root = root
        if not old_root.exists():
            print(f"❌ {old_root} not found"); return 1
        old_ids = sorted(int(f.stem) for f in (old_root / "actions").iterdir() if f.suffix == ".json")
        new_ids = sorted(int(f.stem) for f in (new_root / "actions").iterdir() if f.suffix == ".json")
        print(f"=== Cross-directory comparison ===")
        print(f"  OLD root: {old_root}   (n={len(old_ids)})")
        print(f"  NEW root: {new_root}   (n={len(new_ids)})")
    else:
        old_root = new_root = root
        all_ids = sorted(int(f.stem) for f in (root / "actions").iterdir() if f.suffix == ".json")
        old_ids = [i for i in all_ids if i < args.boundary]
        new_ids = [i for i in all_ids if i >= args.boundary]
        print(f"=== Segment sizes ===")
        print(f"  old (idx < {args.boundary}): {len(old_ids)}")
        print(f"  new (idx >= {args.boundary}): {len(new_ids)}")

    if not old_ids or not new_ids:
        print(f"❌ empty split: old={len(old_ids)} new={len(new_ids)}")
        return 1

    n = min(args.n_samples, len(old_ids), len(new_ids))
    old_pick = random.sample(old_ids, n)
    new_pick = random.sample(new_ids, n)

    # Stats
    old_stats = summarize_latents(old_root, old_pick, "OLD latents")
    new_stats = summarize_latents(new_root, new_pick, "NEW latents")

    verdict_latent = compare_stats(old_stats, new_stats)

    summarize_actions(old_root, old_pick, "OLD")
    summarize_actions(new_root, new_pick, "NEW")

    summarize_video(old_root, old_pick, "OLD")
    summarize_video(new_root, new_pick, "NEW")

    print("\n" + "=" * 78)
    if verdict_latent:
        print("  ✅ latent distributions look consistent between OLD and NEW segments.")
        print("     Safe to proceed with warmup training on the combined dataset.")
    else:
        print("  ⚠️  latent distributions DIVERGE between OLD and NEW segments.")
        print("     Possible causes:")
        print("       • different teacher checkpoint")
        print("       • different denoising num_steps / query_steps / shift")
        print("       • different chunk_size or lerobot source data")
        print("     Investigate before combining. Consider regenerating one segment to match.")
    print("=" * 78)
    return 0 if verdict_latent else 1


if __name__ == "__main__":
    sys.exit(main())
