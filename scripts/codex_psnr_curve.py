#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def frame_psnr(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = np.mean((pred - gt) ** 2, axis=(1, 2, 3))
    psnr = np.full_like(mse, np.inf, dtype=np.float32)
    nonzero = mse > 0
    psnr[nonzero] = 20 * np.log10(255.0) - 10 * np.log10(mse[nonzero])
    return psnr


def summarize(paths: list[Path]) -> dict:
    curves = []
    for path in paths:
        data = np.load(path)
        curves.append(frame_psnr(data["pred"], data["gt"]))
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=np.float32)
    for i, curve in enumerate(curves):
        padded[i, : len(curve)] = curve
    mean = np.nanmean(padded, axis=0)
    return {
        "files": [str(p) for p in paths],
        "num_samples": len(paths),
        "mean_psnr": [float(x) for x in mean],
        "overall_mean_psnr": float(np.nanmean(padded)),
        "last_frame_mean_psnr": float(mean[-1]),
    }


def self_test() -> None:
    pred = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    gt = np.zeros_like(pred)
    assert np.isinf(frame_psnr(pred, gt)[0])
    gt[1] = 255
    assert np.isclose(frame_psnr(pred, gt)[1], 0.0, atol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?")
    parser.add_argument("--out-json", default="psnr_curve.json")
    parser.add_argument("--out-csv", default="psnr_curve.csv")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("self_test_ok")
        return

    paths = sorted(Path(args.input_dir).glob("*_frames.npz"))
    if not paths:
        raise SystemExit(f"no *_frames.npz files in {args.input_dir}")

    summary = summarize(paths)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "mean_psnr"])
        writer.writerows(enumerate(summary["mean_psnr"]))
    print(json.dumps({k: summary[k] for k in ["num_samples", "overall_mean_psnr", "last_frame_mean_psnr"]}, indent=2))


if __name__ == "__main__":
    main()
