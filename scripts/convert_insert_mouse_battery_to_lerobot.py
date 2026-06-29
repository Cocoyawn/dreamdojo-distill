#!/usr/bin/env python3
"""Convert insert-mouse-battery HDF5 episodes to LeRobot format.

The raw files are ALOHA-style HDF5 episodes:

  /observations/images/cam_high
  /observations/images/cam_left_wrist
  /observations/images/cam_right_wrist
  /observations/qpos          -> observation.state
  /observations/qvel
  /observations/effort
  /observations/eef_pose_quat
  /observations/eef_pose_rpy
  /action
  /base_action

This script uses LeRobot's public dataset writer instead of hand-writing
parquet/video/meta files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
VERTICAL_IMAGE_KEY = "cam_vertical"
VERTICAL_IMAGE_SOURCE_KEYS = IMAGE_KEYS
STATE_KEYS = ("qpos", "qvel", "effort", "eef_pose_quat", "eef_pose_rpy")
STATE_FEATURE_NAMES = {
    "qpos": "observation.state",
    "qvel": "observation.qvel",
    "effort": "observation.effort",
    "eef_pose_quat": "observation.eef_pose_quat",
    "eef_pose_rpy": "observation.eef_pose_rpy",
}

PIPER_MODALITY = {
    "state": {
        "left_arm_joint_position": {
            "original_key": "observation.state",
            "start": 0,
            "end": 7,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        },
        "right_arm_joint_position": {
            "original_key": "observation.state",
            "start": 7,
            "end": 14,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        },
    },
    "action": {
        "left_arm_joint_position": {
            "original_key": "action",
            "start": 0,
            "end": 7,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        },
        "right_arm_joint_position": {
            "original_key": "action",
            "start": 7,
            "end": 14,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        },
    },
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
        "cam_vertical": {"original_key": "observation.images.cam_vertical"},
    },
    "annotation": {
        "language.action_text": {"original_key": "task_index"},
    },
}

h5py = None
LeRobotDataset = None


def load_hdf5_dependency() -> None:
    global h5py
    if h5py is None:
        import h5py as h5py_module

        h5py = h5py_module


def load_lerobot_dependency() -> None:
    global LeRobotDataset
    if LeRobotDataset is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset as dataset_cls
        except ImportError:
            try:
                from lerobot.datasets import LeRobotDataset as dataset_cls
            except ImportError:  # pragma: no cover - compatibility with older LeRobot layouts.
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as dataset_cls

        LeRobotDataset = dataset_cls


def configure_video_encoder(args: argparse.Namespace) -> None:
    """Override LeRobot's slow default AV1 encoder for this conversion job."""
    if args.image_writer:
        return

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
    import lerobot.common.datasets.video_utils as video_utils

    original_encode_video_frames = video_utils.encode_video_frames
    accepts_preset = "preset" in inspect.signature(original_encode_video_frames).parameters

    def encode_video_frames_fast(imgs_dir, video_path, fps, overwrite=False, **kwargs):
        kwargs.update(
            {
                "vcodec": args.video_codec,
                "crf": args.video_crf,
                "g": args.video_gop_size,
                "fast_decode": args.video_fast_decode,
            }
        )
        if args.video_preset and accepts_preset:
            kwargs["preset"] = args.video_preset
        return original_encode_video_frames(imgs_dir, video_path, fps, overwrite=overwrite, **kwargs)

    video_utils.encode_video_frames = encode_video_frames_fast
    lerobot_dataset_module.encode_video_frames = encode_video_frames_fast
    print(
        "Video encoder:",
        f"codec={args.video_codec}",
        f"crf={args.video_crf}",
        f"gop={args.video_gop_size}",
        f"fast_decode={args.video_fast_decode}",
        f"preset={args.video_preset}",
    )


def natural_episode_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def list_episode_files(raw_dir: Path, splits: tuple[str, ...]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = raw_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")
        for path in sorted(split_dir.glob("episode_*.hdf5"), key=natural_episode_id):
            files.append((split, path))
    if not files:
        raise FileNotFoundError(f"no episode_*.hdf5 files found under {raw_dir}")
    return files


def require_dataset(h5: h5py.File, key: str) -> h5py.Dataset:
    if key not in h5:
        raise KeyError(f"missing dataset: /{key}")
    return h5[key]


def infer_features(first_file: Path, video: bool, add_vertical_view: bool) -> tuple[dict, int]:
    with h5py.File(first_file, "r") as h5:
        action = require_dataset(h5, "action")
        fps = int(h5.attrs.get("fps", 30))
        features = {
            "action": {
                "dtype": "float32",
                "shape": tuple(action.shape[1:]),
                "names": [f"action_{i}" for i in range(int(np.prod(action.shape[1:])))],
            },
            "base_action": {
                "dtype": "float32",
                "shape": tuple(require_dataset(h5, "base_action").shape[1:]),
                "names": None,
            },
        }

        for key in STATE_KEYS:
            ds = require_dataset(h5, f"observations/{key}")
            features[STATE_FEATURE_NAMES[key]] = {
                "dtype": "float32",
                "shape": tuple(ds.shape[1:]),
                "names": None,
            }

        image_shapes: dict[str, tuple[int, int, int]] = {}
        for key in IMAGE_KEYS:
            ds = require_dataset(h5, f"observations/images/{key}")
            if len(ds.shape) != 4:
                raise ValueError(f"expected image dataset /observations/images/{key} to be THWC, got {ds.shape}")
            height, width, channels = ds.shape[1:]
            if channels != 3:
                raise ValueError(f"expected RGB images for {key}, got shape {ds.shape}")
            image_shapes[key] = (height, width, channels)
            features[f"observation.images.{key}"] = {
                "dtype": "video" if video else "image",
                "shape": (channels, height, width),
                "names": ["channels", "height", "width"],
            }

        if add_vertical_view:
            source_shapes = [image_shapes[key] for key in VERTICAL_IMAGE_SOURCE_KEYS]
            widths = {shape[1] for shape in source_shapes}
            channels_set = {shape[2] for shape in source_shapes}
            if len(widths) != 1 or len(channels_set) != 1:
                raise ValueError(f"cannot vertically stack views with different widths/channels: {source_shapes}")
            height = sum(shape[0] for shape in source_shapes)
            width = source_shapes[0][1]
            channels = source_shapes[0][2]
            features[f"observation.images.{VERTICAL_IMAGE_KEY}"] = {
                "dtype": "video" if video else "image",
                "shape": (channels, height, width),
                "names": ["channels", "height", "width"],
            }

    return features, fps


def add_vertical_view_feature(features: dict, video: bool) -> dict:
    """Return features with a video/image feature for the postprocessed vertical view."""
    features = dict(features)
    source_shapes = [features[f"observation.images.{key}"]["shape"] for key in VERTICAL_IMAGE_SOURCE_KEYS]
    channels_set = {shape[0] for shape in source_shapes}
    widths = {shape[2] for shape in source_shapes}
    if len(channels_set) != 1 or len(widths) != 1:
        raise ValueError(f"cannot vertically stack views with different channels/widths: {source_shapes}")

    channels = source_shapes[0][0]
    height = sum(shape[1] for shape in source_shapes)
    width = source_shapes[0][2]
    features[f"observation.images.{VERTICAL_IMAGE_KEY}"] = {
        "dtype": "video" if video else "image",
        "shape": (channels, height, width),
        "names": ["channels", "height", "width"],
    }
    return features


def print_hdf5_summary(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        print(f"\n{path}")
        print("root attrs:", {k: repr(v) for k, v in h5.attrs.items()})

        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"  /{name}: shape={obj.shape}, dtype={obj.dtype}")

        h5.visititems(visitor)


def check_episode_lengths(h5: h5py.File, path: Path) -> int:
    length = int(require_dataset(h5, "action").shape[0])
    keys = ["base_action", *(f"observations/{k}" for k in STATE_KEYS), *(f"observations/images/{k}" for k in IMAGE_KEYS)]
    for key in keys:
        ds = require_dataset(h5, key)
        if int(ds.shape[0]) != length:
            raise ValueError(f"{path}: /{key} length {ds.shape[0]} != action length {length}")
    return length


def frame_from_hdf5(h5: h5py.File, index: int, task: str, add_vertical_view: bool) -> dict:
    frame = {
        "task": task,
        "action": np.asarray(h5["action"][index], dtype=np.float32),
        "base_action": np.asarray(h5["base_action"][index], dtype=np.float32),
    }

    for key in STATE_KEYS:
        frame[STATE_FEATURE_NAMES[key]] = np.asarray(h5[f"observations/{key}"][index], dtype=np.float32)

    images: dict[str, np.ndarray] = {}
    for key in IMAGE_KEYS:
        image = np.asarray(h5[f"observations/images/{key}"][index])
        if image.dtype != np.uint8:
            image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
        images[key] = image
        frame[f"observation.images.{key}"] = image

    if add_vertical_view:
        frame[f"observation.images.{VERTICAL_IMAGE_KEY}"] = np.concatenate(
            [images[key] for key in VERTICAL_IMAGE_SOURCE_KEYS],
            axis=0,
        )

    return frame


def find_ffmpeg_exe(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg was not found in PATH and imageio_ffmpeg is not installed") from exc


def video_codec_metadata(codec: str) -> str:
    return {"libsvtav1": "av1", "h264": "h264", "hevc": "hevc"}.get(codec, codec)


def run_ffmpeg_vstack(
    ffmpeg: str,
    input_paths: list[Path],
    output_path: Path,
    fps: int,
    codec: str,
    crf: int,
    gop_size: int,
    preset: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    for path in input_paths:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            f"[0:v][1:v][2:v]vstack=inputs={len(input_paths)}[v]",
            "-map",
            "[v]",
            "-r",
            str(fps),
            "-an",
            "-c:v",
            "libx264" if codec == "h264" else codec,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-g",
            str(gop_size),
        ]
    )
    if preset:
        cmd.extend(["-preset", preset])
    cmd.extend(
        [
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True)


def synthesize_vertical_stats(stats: dict) -> dict:
    source_stats = [stats[f"observation.images.{key}"] for key in VERTICAL_IMAGE_SOURCE_KEYS]
    mins = np.stack([np.asarray(item["min"], dtype=np.float64) for item in source_stats])
    maxs = np.stack([np.asarray(item["max"], dtype=np.float64) for item in source_stats])
    means = np.stack([np.asarray(item["mean"], dtype=np.float64) for item in source_stats])
    variances = np.stack([np.asarray(item["std"], dtype=np.float64) ** 2 for item in source_stats])
    mean = means.mean(axis=0)
    variance = (variances + (means - mean) ** 2).mean(axis=0)
    return {
        "min": mins.min(axis=0).tolist(),
        "max": maxs.max(axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": source_stats[0]["count"],
    }


def update_vertical_metadata(output_dir: Path, fps: int, video_codec: str) -> None:
    info_path = output_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    features = add_vertical_view_feature(info["features"], video=True)
    vertical = features[f"observation.images.{VERTICAL_IMAGE_KEY}"]
    _, height, width = vertical["shape"]
    vertical["info"] = {
        "video.height": height,
        "video.width": width,
        "video.codec": video_codec_metadata(video_codec),
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    info["features"] = features
    info["total_videos"] = info["total_episodes"] * sum(
        1 for feature in features.values() if feature.get("dtype") == "video"
    )
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    stats_path = output_dir / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        updated_lines = []
        with open(stats_path) as f:
            for line in f:
                item = json.loads(line)
                item["stats"][f"observation.images.{VERTICAL_IMAGE_KEY}"] = synthesize_vertical_stats(
                    item["stats"]
                )
                updated_lines.append(json.dumps(item))
        with open(stats_path, "w") as f:
            f.write("\n".join(updated_lines) + "\n")


def add_vertical_view_videos(args: argparse.Namespace, output_dir: Path, fps: int) -> None:
    ffmpeg = find_ffmpeg_exe(args.ffmpeg_path)
    vertical_key = f"observation.images.{VERTICAL_IMAGE_KEY}"
    source_keys = [f"observation.images.{key}" for key in VERTICAL_IMAGE_SOURCE_KEYS]
    vertical_dir = output_dir / "videos" / "chunk-000" / vertical_key
    source_dirs = [output_dir / "videos" / "chunk-000" / key for key in source_keys]
    first_source = source_dirs[0]
    episodes = sorted(first_source.glob("episode_*.mp4"))
    if not episodes:
        raise FileNotFoundError(f"no source videos found under {first_source}")

    def convert_one(source_episode: Path) -> Path:
        input_paths = [source_dir / source_episode.name for source_dir in source_dirs]
        for path in input_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        output_path = vertical_dir / source_episode.name
        run_ffmpeg_vstack(
            ffmpeg=ffmpeg,
            input_paths=input_paths,
            output_path=output_path,
            fps=fps,
            codec=args.vertical_video_codec,
            crf=args.vertical_video_crf,
            gop_size=args.video_gop_size,
            preset=args.video_preset,
        )
        return output_path

    print(
        "Adding vertical view videos:",
        f"episodes={len(episodes)}",
        f"workers={args.vertical_video_workers}",
        f"codec={args.vertical_video_codec}",
        f"crf={args.vertical_video_crf}",
    )
    with ThreadPoolExecutor(max_workers=args.vertical_video_workers) as executor:
        futures = {executor.submit(convert_one, episode): episode for episode in episodes}
        for idx, future in enumerate(as_completed(futures), start=1):
            output_path = future.result()
            if idx == 1 or idx == len(futures) or idx % 10 == 0:
                print(f"  [{idx}/{len(futures)}] {output_path.name}")

    update_vertical_metadata(output_dir, fps=fps, video_codec=args.vertical_video_codec)


def finish_dataset(dataset) -> None:
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()


def write_dreamdojo_modality(output_dir: Path, add_vertical_view: bool) -> None:
    modality = json.loads(json.dumps(PIPER_MODALITY))
    if not add_vertical_view:
        modality["video"].pop(VERTICAL_IMAGE_KEY, None)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)


def convert(args: argparse.Namespace) -> None:
    load_hdf5_dependency()

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    splits = tuple(args.splits)
    episode_files = list_episode_files(raw_dir, splits)
    if args.limit_episodes is not None:
        episode_files = episode_files[: args.limit_episodes]

    features, fps = infer_features(
        episode_files[0][1],
        video=not args.image_writer,
        add_vertical_view=args.add_vertical_view and args.image_writer,
    )
    if args.fps is not None:
        fps = args.fps

    if args.dry_run:
        print(f"raw_dir: {raw_dir}")
        print(f"output_dir: {output_dir}")
        print(f"splits: {splits}")
        print(f"episodes: {len(episode_files)}")
        print(f"fps: {fps}")
        print("features:")
        print(json.dumps(features, indent=2))
        print_hdf5_summary(episode_files[0][1])
        return

    load_lerobot_dependency()
    configure_video_encoder(args)

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_dir,
        fps=fps,
        features=features,
        robot_type=args.robot_type,
        use_videos=not args.image_writer,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    for new_episode_index, (split, path) in enumerate(episode_files):
        with h5py.File(path, "r") as h5:
            length = check_episode_lengths(h5, path)
            print(f"[{new_episode_index + 1}/{len(episode_files)}] {split}/{path.name}: {length} frames")
            for i in range(length):
                dataset.add_frame(
                    frame_from_hdf5(
                        h5,
                        i,
                        task=args.task,
                        add_vertical_view=args.add_vertical_view and args.image_writer,
                    )
                )
            dataset.save_episode()

    finish_dataset(dataset)
    if args.add_vertical_view and not args.image_writer:
        add_vertical_view_videos(args, output_dir=output_dir, fps=fps)
    if args.dreamdojo_modality:
        write_dreamdojo_modality(output_dir, add_vertical_view=args.add_vertical_view)
    print(f"wrote LeRobot dataset to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("insert-mouse-battery-data-raw"),
        help="Directory containing perfect/ and retry/ HDF5 episode folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("piper_insert_mouse_battery_lerobot"),
        help="Output LeRobot dataset directory.",
    )
    parser.add_argument("--repo-id", default="local/piper_insert_mouse_battery", help="LeRobot repo_id metadata.")
    parser.add_argument("--robot-type", default="aloha", help="LeRobot robot_type metadata.")
    parser.add_argument("--task", default="insert mouse battery", help="Task text stored for each episode.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["perfect"],
        choices=["perfect", "retry"],
        help="Which raw subdirectories to convert, in order.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS. Defaults to HDF5 attr fps or 30.")
    parser.add_argument(
        "--add-vertical-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add observation.images.cam_vertical by stacking cam_high, cam_left_wrist, "
            "and cam_right_wrist vertically."
        ),
    )
    parser.add_argument("--limit-episodes", type=int, default=None, help="Convert only the first N episodes.")
    parser.add_argument("--dry-run", action="store_true", help="Print inferred features and first-file summary.")
    parser.add_argument("--force", action="store_true", help="Delete output directory before writing.")
    parser.add_argument(
        "--image-writer",
        action="store_true",
        help="Store image columns instead of encoding LeRobot video columns. Larger, but easier to debug.",
    )
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--video-codec",
        choices=["h264", "hevc", "libsvtav1"],
        default="libsvtav1",
        help="Codec used when writing LeRobot mp4 videos. h264 is much faster on CPU than the LeRobot default libsvtav1.",
    )
    parser.add_argument("--video-crf", type=int, default=30, help="Video CRF passed to the encoder.")
    parser.add_argument(
        "--video-preset",
        default="veryfast",
        help="Video encoder preset when supported, e.g. ultrafast, veryfast, faster, fast, medium.",
    )
    parser.add_argument(
        "--video-gop-size",
        type=int,
        default=2,
        help="Video GOP size. Keep 2 for LeRobot's fast random frame access behavior.",
    )
    parser.add_argument("--video-fast-decode", type=int, default=0, help="Fast-decode flag passed to supported codecs.")
    parser.add_argument(
        "--vertical-video-codec",
        choices=["h264", "hevc", "libsvtav1"],
        default="h264",
        help="Codec for postprocessed cam_vertical videos.",
    )
    parser.add_argument(
        "--vertical-video-crf",
        type=int,
        default=23,
        help="CRF for postprocessed cam_vertical videos.",
    )
    parser.add_argument(
        "--vertical-video-workers",
        type=int,
        default=4,
        help="Number of parallel ffmpeg processes used to create cam_vertical after conversion.",
    )
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit ffmpeg executable path.")
    parser.add_argument(
        "--dreamdojo-modality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write meta/modality.json for DreamDojo piper embodiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
