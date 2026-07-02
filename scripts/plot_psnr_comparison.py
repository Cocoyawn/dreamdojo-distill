"""
Per-frame PSNR comparison plot: teacher vs student (iter_1000, iter_4000).
Style: research paper aesthetic, custom palette, Times/Liberation Serif.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


PALETTE = {
    "teacher":   "#2878B5",   # deep blue
    "band_teacher": "#9AC9DB",
    "student_1k":"#F8AC8C",   # peach
    "student_4k":"#C82423",   # deep red
    "chunk_boundary": "#FF8884",
}

# Font: Times New Roman with Liberation Serif fallback (metric-identical)
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    14,
    "axes.titlesize":    15,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.linewidth":    1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
})


def load_curve(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array(json.load(f)["mean_psnr"], dtype=np.float32)


teacher   = load_curve("results/piper_eval_teacher_baseline/psnr_curve.json")
student_1 = load_curve("results/piper_eval_student_iter_000001000/psnr_curve.json")
student_4 = load_curve("results/piper_eval_student_iter_000004000/psnr_curve.json")

# Compute per-frame std across the 5 samples for teacher (visualisation band)
def per_frame_std(save_dir: str) -> np.ndarray:
    from glob import glob
    paths = sorted(glob(f"{save_dir}/*_frames.npz"))
    curves = []
    for p in paths:
        z = np.load(p)
        pred = z["pred"].astype(np.float32)
        gt = z["gt"].astype(np.float32)
        mse = ((pred - gt) ** 2).mean(axis=(1, 2, 3))
        with np.errstate(divide="ignore"):
            psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
        curves.append(psnr)
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c
    return np.nanstd(padded, axis=0)


teacher_std = per_frame_std("results/piper_eval_teacher_baseline")

n_frames = len(teacher)
x = np.arange(n_frames)

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

# Chunk boundaries (dashed vertical guides)
for cb in [12, 24, 36]:
    ax.axvline(cb, color=PALETTE["chunk_boundary"], lw=0.9, ls="--", alpha=0.55, zorder=1)

# Teacher: uncertainty band + main line
ax.fill_between(x, teacher - teacher_std, teacher + teacher_std,
                color=PALETTE["band_teacher"], alpha=0.35, linewidth=0, zorder=2,
                label=r"Teacher $\pm 1\sigma$")
ax.plot(x, teacher, color=PALETTE["teacher"], lw=2.4, marker="o", markersize=4,
        markerfacecolor="white", markeredgewidth=1.4, zorder=5,
        label=r"Teacher (35-step, bidirectional)")

# Student iter 1000
ax.plot(x, student_1, color=PALETTE["student_1k"], lw=2.0, marker="s", markersize=4,
        markerfacecolor="white", markeredgewidth=1.2, zorder=4,
        label=r"Student iter 1000 (4-step causal)")

# Student iter 4000
ax.plot(x, student_4, color=PALETTE["student_4k"], lw=2.0, marker="^", markersize=4.5,
        markerfacecolor="white", markeredgewidth=1.2, zorder=4,
        label=r"Student iter 4000 (4-step causal)")

# Chunk boundary text hint
ax.text(6, ax.get_ylim()[1] * 0.98 if False else 47,
        "chunk boundaries", color=PALETTE["chunk_boundary"],
        fontsize=9, alpha=0.9, ha="left", va="top", style="italic")

ax.set_xlabel("Frame index (autoregressive step)")
ax.set_ylabel("Per-frame PSNR (dB)")
ax.set_title("DreamDojo Piper: Teacher vs. Distilled Student — per-frame PSNR")
ax.set_xlim(-1, n_frames)
ax.set_ylim(15, 50)
ax.set_xticks(np.arange(0, n_frames + 1, 6))
ax.grid(True, alpha=0.25, linewidth=0.5)

leg = ax.legend(loc="upper right", frameon=True, fancybox=False,
                edgecolor="black", framealpha=0.95)
leg.get_frame().set_linewidth(0.8)

# Small stat annotation
def mean_of(v: np.ndarray) -> float:
    return float(np.nanmean(v))

stats_text = (
    f"overall mean PSNR:\n"
    f"  Teacher   : {mean_of(teacher):5.2f} dB\n"
    f"  iter 1000 : {mean_of(student_1):5.2f} dB\n"
    f"  iter 4000 : {mean_of(student_4):5.2f} dB\n"
    f"  1k → 4k    : {mean_of(student_4) - mean_of(student_1):+.2f} dB"
)
ax.text(0.017, 0.03, stats_text, transform=ax.transAxes,
        fontsize=9.5, family="monospace", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="0.4", linewidth=0.7, alpha=0.92))

plt.tight_layout()

out_dir = Path("results/comparison")
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    p = out_dir / f"psnr_frame_curves.{ext}"
    fig.savefig(p, bbox_inches="tight")
    print(f"saved {p}")

# also a compact version without stats box for clean paper use
fig2, ax2 = plt.subplots(figsize=(8, 4.2), dpi=200)
for cb in [12, 24, 36]:
    ax2.axvline(cb, color=PALETTE["chunk_boundary"], lw=0.9, ls="--", alpha=0.55)
ax2.fill_between(x, teacher - teacher_std, teacher + teacher_std,
                 color=PALETTE["band_teacher"], alpha=0.35, linewidth=0)
ax2.plot(x, teacher, color=PALETTE["teacher"], lw=2.4, marker="o", markersize=4,
         markerfacecolor="white", markeredgewidth=1.4, label="Teacher (35-step)")
ax2.plot(x, student_1, color=PALETTE["student_1k"], lw=2.0, marker="s", markersize=4,
         markerfacecolor="white", markeredgewidth=1.2, label="Student @ iter 1000")
ax2.plot(x, student_4, color=PALETTE["student_4k"], lw=2.0, marker="^", markersize=4.5,
         markerfacecolor="white", markeredgewidth=1.2, label="Student @ iter 4000")
ax2.set_xlabel("Frame index")
ax2.set_ylabel("PSNR (dB)")
ax2.set_xlim(-1, n_frames)
ax2.set_ylim(15, 50)
ax2.set_xticks(np.arange(0, n_frames + 1, 6))
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.95)
plt.tight_layout()
for ext in ("png", "pdf"):
    p = out_dir / f"psnr_frame_curves_compact.{ext}"
    fig2.savefig(p, bbox_inches="tight")
    print(f"saved {p}")
