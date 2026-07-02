"""Aggregate per-ckpt JSONs into a plot + CSV."""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {"student": "#C82423", "teacher": "#2878B5", "student_band": "#F8AC8C"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    args = p.parse_args()

    d = Path(args.dir)
    files = sorted(glob.glob(str(d / "iter_*.json")))
    if not files:
        print(f"no iter_*.json in {d}")
        return

    rows = []
    for f in files:
        r = json.load(open(f))
        m = re.search(r"iter_0*(\d+)", Path(f).stem)
        it = int(m.group(1))
        rows.append((it, r["student_mean"], r["teacher_mean"],
                     np.std(r["student"]), r["student"], r["teacher"]))
    rows.sort()

    iters = np.array([r[0] for r in rows])
    stud = np.array([r[1] for r in rows])
    teach = np.array([r[2] for r in rows])
    stud_std = np.array([r[3] for r in rows])

    # CSV
    with (d / "psnr_by_iter.csv").open("w") as f:
        f.write("iter,student_mean,teacher_mean,student_std\n")
        for it, s, t, sd, _, _ in rows:
            f.write(f"{it},{s:.4f},{t:.4f},{sd:.4f}\n")

    # JSON summary
    (d / "psnr_by_iter.json").write_text(json.dumps(
        {int(r[0]): {"student_mean": r[1], "teacher_mean": r[2],
                     "student_std": r[3], "student": r[4], "teacher": r[5]}
         for r in rows}, indent=2))

    # plot
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.linewidth": 1.1,
    })
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.fill_between(iters, stud - stud_std, stud + stud_std,
                    color=PALETTE["student_band"], alpha=0.30, linewidth=0,
                    label=r"Student $\pm 1\sigma$")
    ax.plot(iters, stud, color=PALETTE["student"], lw=2.4, marker="o", markersize=5.5,
            markerfacecolor="white", markeredgewidth=1.4,
            label="Student (single-step pred)")
    ax.plot(iters, teach, color=PALETTE["teacher"], lw=2.0, marker="s", markersize=5,
            markerfacecolor="white", markeredgewidth=1.2, ls="--",
            label="Teacher clean (VAE decode)  — upper bound")

    ax.set_xlabel("Warmup iteration")
    ax.set_ylabel("Single-step prediction PSNR vs. real GT (dB)")
    ax.set_title("Warmup checkpoint sweep — single-step reconstruction quality")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", framealpha=0.95)
    plt.tight_layout()
    out = d / "psnr_curve.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")

    # short summary
    print()
    print(f"{'iter':>8}  {'student':>9}  {'teacher':>9}  {'gap':>7}")
    for it, s, t, _, _, _ in rows:
        print(f"{it:>8}  {s:>9.2f}  {t:>9.2f}  {s-t:>+7.2f}")


if __name__ == "__main__":
    main()
