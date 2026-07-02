"""Pretty version of the warmup PSNR curve (paper palette, Times New Roman, square markers)."""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "teacher":       "#2878B5",  # deep blue  — teacher mean
    "teacher_light": "#9AC9DB",  # light blue — teacher marker fill
    "student_band":  "#F8AC8C",  # peach      — student ±1σ band
    "student":       "#C82423",  # deep red   — student mean
    "highlight":     "#FF8884",  # light red  — peak marker
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/warmup_ckpt_sweep/psnr_by_iter.csv")
    p.add_argument("--out", default="results/warmup_ckpt_sweep/psnr_curve_v2.png")
    args = p.parse_args()

    iters, stud, teach, stud_std = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            stud.append(float(row["student_mean"]))
            teach.append(float(row["teacher_mean"]))
            stud_std.append(float(row["student_std"]))
    iters = np.array(iters); stud = np.array(stud)
    teach = np.array(teach); stud_std = np.array(stud_std)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
    })

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=200)

    # ±1σ band under student line
    ax.fill_between(iters, stud - stud_std, stud + stud_std,
                    color=PALETTE["student_band"], alpha=0.45, linewidth=0,
                    label=r"Student $\pm 1\sigma$", zorder=1)

    # Teacher upper bound
    ax.plot(iters, teach,
            color=PALETTE["teacher"], lw=2.0, ls="--", zorder=2,
            marker="s", markersize=6, markerfacecolor=PALETTE["teacher_light"],
            markeredgecolor=PALETTE["teacher"], markeredgewidth=1.3,
            label="Teacher clean (VAE decode) — upper bound")

    # Student
    ax.plot(iters, stud,
            color=PALETTE["student"], lw=2.4, zorder=3,
            marker="s", markersize=6.5, markerfacecolor="white",
            markeredgecolor=PALETTE["student"], markeredgewidth=1.6,
            label="Student (single-step pred)")

    # Highlight peak
    peak_idx = int(np.argmax(stud))
    px, py = iters[peak_idx], stud[peak_idx]
    ax.plot(px, py, marker="s", markersize=11,
            markerfacecolor=PALETTE["highlight"],
            markeredgecolor=PALETTE["student"], markeredgewidth=1.6,
            zorder=4)
    ax.annotate(f"peak · iter {px}\n{py:.2f} dB",
                xy=(px, py), xytext=(px - 4200, py + 1.4),
                fontsize=11, color=PALETTE["student"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=PALETTE["student"],
                                lw=1.0, alpha=0.7,
                                connectionstyle="arc3,rad=0.15"),
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor=PALETTE["student"],
                          linewidth=1.0, alpha=0.9))

    ax.set_xlabel("Warmup iteration")
    ax.set_ylabel("Single-step prediction PSNR vs. real GT (dB)")
    ax.set_title("Warmup checkpoint sweep — single-step reconstruction quality",
                 pad=10)
    ax.grid(True, alpha=0.30, linewidth=0.6, linestyle=":")
    ax.set_axisbelow(True)

    # Nice tick spacing
    ax.set_xlim(iters.min() - 500, iters.max() + 500)
    y_lo = min(stud.min() - stud_std.max() - 0.6, 27.0)
    y_hi = max(teach.max() + 0.6, 38.5)
    ax.set_ylim(y_lo, y_hi)

    leg = ax.legend(loc="lower right", frameon=True, edgecolor="0.3",
                    framealpha=0.95, fancybox=False, borderpad=0.7)
    leg.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
