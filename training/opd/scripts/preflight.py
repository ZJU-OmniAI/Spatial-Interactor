#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "prompt",
    "privileged_prompt",
    "answer",
    "videos",
    "source_id",
    "task_type",
    "reward_family",
    "trace_id",
}

VSTI_TASKS = {
    "camera_movement_direction",
    "camera_movement_direction_interval",
    "camera_displacement",
    "camera_displacement_interval",
}
EXPECTED_TRACE_SEGMENTS = [
    ("1", "01", "08"),
    ("2", "09", "16"),
    ("3", "17", "24"),
    ("4", "25", "32"),
]
TRACE_SEGMENT_RE = re.compile(
    r"\[SEGMENT_(\d)\]\s+Frames\s+(\d{2})\s*[-\u2013]\s*(\d{2})\s*:"
)
SUPPORTED_MODELS = {
    ("qwen2_5_vl", 3584): "qwen25vl7b",
    ("qwen2_5_vl", 2048): "qwen25vl3b",
    ("qwen3_vl", 4096): "qwen3vl8b",
    ("qwen3_vl", 2560): "qwen3vl4b",
}


def video_values(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyr1-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--video-nframes", type=int, default=32)
    parser.add_argument("--video-check-workers", type=int, default=8)
    parser.add_argument("--expected-model-id", choices=sorted(SUPPORTED_MODELS.values()))
    return parser.parse_args()


def identify_model(config: dict) -> dict[str, str | int]:
    model_type = str(config.get("model_type") or "")
    text_config = config.get("text_config") or {}
    hidden_size = int(config.get("hidden_size") or text_config.get("hidden_size") or 0)
    model_id = SUPPORTED_MODELS.get((model_type, hidden_size), "")
    return {
        "model_id": model_id,
        "model_type": model_type,
        "hidden_size": hidden_size,
    }


def inspect_video(path: Path, nframes: int) -> dict[str, int]:
    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    total_frames = len(reader)
    if total_frames < nframes:
        raise ValueError(f"only {total_frames} native frames; requires {nframes}")
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(np.int64)
    decoded = reader.get_batch(indices.tolist())
    if len(decoded.shape) != 4 or int(decoded.shape[0]) != nframes:
        raise ValueError(f"decoded unexpected shape {tuple(decoded.shape)}")
    return {
        "native_frames": int(total_frames),
        "decoded_frames": int(decoded.shape[0]),
        "height": int(decoded.shape[1]),
        "width": int(decoded.shape[2]),
    }


def inspect_all_videos(paths: list[Path], nframes: int, workers: int) -> tuple[dict, list[str]]:
    stats = {
        "requested_frames_per_video": nframes,
        "attempted_videos": len(paths),
        "checked_videos": 0,
        "decode_failures": 0,
        "min_native_frames": None,
        "max_native_frames": None,
        "min_height": None,
        "min_width": None,
    }
    errors: list[str] = []
    if workers <= 0:
        return stats, ["video-check-workers must be positive"]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(inspect_video, path, nframes): path for path in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                result = future.result()
                stats["checked_videos"] += 1
                for key, value in (
                    ("min_native_frames", result["native_frames"]),
                    ("min_height", result["height"]),
                    ("min_width", result["width"]),
                ):
                    current = stats[key]
                    stats[key] = value if current is None else min(current, value)
                current_max = stats["max_native_frames"]
                stats["max_native_frames"] = (
                    result["native_frames"]
                    if current_max is None
                    else max(current_max, result["native_frames"])
                )
            except Exception as exc:
                stats["decode_failures"] += 1
                if len(errors) < 20:
                    errors.append(f"video cannot provide {nframes} uniform frames: {path}: {exc}")
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"video preflight: {completed}/{len(futures)} checked, "
                    f"{stats['decode_failures']} failures",
                    file=sys.stderr,
                    flush=True,
                )
    return stats, errors


def main():
    args = parse_args()
    errors = []
    versions = {}
    for name in (
        "torch",
        "transformers",
        "ray",
        "vllm",
        "datasets",
        "pandas",
        "pyarrow",
        "tensorboard",
        "decord",
    ):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            errors.append(f"cannot import {name}: {exc}")

    for path in (
        args.easyr1_dir / "verl" / "trainer" / "opd.py",
        args.easyr1_dir / "verl" / "workers" / "actor" / "dp_actor.py",
        args.easyr1_dir / "verl" / "trainer" / "rollout_audit.py",
        args.model_path / "config.json",
        args.data_dir / "train.parquet",
        args.data_dir / "val.parquet",
    ):
        if not path.exists():
            errors.append(f"missing {path}")

    stats = {}
    config_path = args.model_path / "config.json"
    if config_path.exists():
        try:
            model_config = json.loads(config_path.read_text())
            model_identity = identify_model(model_config)
            stats["model"] = model_identity
            if not model_identity["model_id"]:
                errors.append(
                    "MODEL_PATH is not one of the supported Qwen2.5-VL/Qwen3-VL checkpoints: "
                    f"model_type={model_identity['model_type']}, "
                    f"hidden_size={model_identity['hidden_size']}"
                )
            elif args.expected_model_id and model_identity["model_id"] != args.expected_model_id:
                errors.append(
                    f"MODEL_PATH is {model_identity['model_id']}, expected {args.expected_model_id}"
                )
        except Exception as exc:
            errors.append(f"cannot read model config: {exc}")

    split_videos = {}
    split_traces = {}
    resolved_media = set()
    media_root = args.media_root.resolve()
    video_to_traces: dict[str, set[str]] = defaultdict(set)
    trace_to_videos: dict[str, set[str]] = defaultdict(set)
    trace_format_rows_checked = 0
    for split in ("train", "val"):
        path = args.data_dir / f"{split}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            errors.append(f"{split}.parquet missing columns: {sorted(missing)}")
            continue
        stats[f"{split}_rows"] = len(frame)
        stats[f"{split}_tasks"] = frame["task_type"].value_counts().to_dict()
        if split == "train":
            stats["train_vsti_rows"] = int(frame["task_type"].isin(VSTI_TASKS).sum())
        videos_for_split = set()
        traces_for_split = set()
        for row_index, row in frame.iterrows():
            values = video_values(row["videos"])
            if len(values) != 1 or not values[0]:
                errors.append(f"{split} row {row_index} must contain exactly one video")
                continue
            video = Path(values[0])
            if video.is_absolute():
                errors.append(f"{split} row {row_index} has non-portable absolute video path: {video}")
            resolved = (video if video.is_absolute() else args.media_root / video).resolve()
            if not resolved.is_relative_to(media_root):
                errors.append(
                    f"{split} row {row_index} video escapes MEDIA_ROOT: {video} -> {resolved}"
                )
            videos_for_split.add(str(resolved))
            resolved_media.add(resolved)

            trace_id = str(row["trace_id"] or "").strip()
            if not trace_id:
                errors.append(f"{split} row {row_index} has empty trace_id")
            traces_for_split.add(trace_id)
            video_to_traces[str(resolved)].add(trace_id)
            trace_to_videos[trace_id].add(str(resolved))

            prompt = str(row["prompt"] or "")
            privileged = str(row["privileged_prompt"] or "")
            if "<PRIVILEGED_STATE_TRANSITIONS>" in prompt:
                errors.append(f"{split} row {row_index} leaks privileged context into rollout prompt")
            if privileged.count("<PRIVILEGED_STATE_TRANSITIONS>") != 1:
                errors.append(f"{split} row {row_index} has malformed privileged prompt")
            segments = TRACE_SEGMENT_RE.findall(privileged)
            trace_format_rows_checked += 1
            if segments != EXPECTED_TRACE_SEGMENTS:
                errors.append(
                    f"{split} row {row_index} trace does not contain the fixed four 32-frame intervals"
                )
            try:
                answer = json.loads(str(row["answer"]))
                if not isinstance(answer, dict):
                    raise TypeError("answer is not an object")
                if answer.get("reward_family") != row["reward_family"]:
                    errors.append(f"{split} row {row_index} reward family mismatch")
            except Exception as exc:
                errors.append(f"{split} row {row_index} has invalid answer JSON: {exc}")

        split_videos[split] = videos_for_split
        split_traces[split] = traces_for_split
        stats[f"{split}_unique_videos"] = len(videos_for_split)
        stats[f"{split}_unique_traces"] = len(traces_for_split)

    video_overlap = split_videos.get("train", set()) & split_videos.get("val", set())
    trace_overlap = split_traces.get("train", set()) & split_traces.get("val", set())
    if video_overlap:
        errors.append(f"train/val video leakage: {len(video_overlap)} videos")
    if trace_overlap:
        errors.append(f"train/val trace leakage: {len(trace_overlap)} traces")
    stats["all_unique_videos"] = len(set().union(*split_videos.values())) if split_videos else 0
    stats["all_unique_traces"] = len(set().union(*split_traces.values())) if split_traces else 0
    videos_with_multiple_traces = sum(len(values) != 1 for values in video_to_traces.values())
    traces_with_multiple_videos = sum(len(values) != 1 for values in trace_to_videos.values())
    stats["trace_alignment"] = {
        "trace_format_rows_checked": trace_format_rows_checked,
        "videos_with_multiple_traces": videos_with_multiple_traces,
        "traces_with_multiple_videos": traces_with_multiple_videos,
    }
    if videos_with_multiple_traces:
        errors.append(f"video-to-trace mapping is not one-to-one for {videos_with_multiple_traces} videos")
    if traces_with_multiple_videos:
        errors.append(f"trace-to-video mapping is not one-to-one for {traces_with_multiple_videos} traces")
    stats["missing_media"] = 0
    for path in sorted(resolved_media):
        if not path.is_file():
            stats["missing_media"] += 1
            if stats["missing_media"] <= 20:
                errors.append(f"video is missing: {path}")

    if args.video_nframes != 32:
        errors.append(
            f"OPD privileged traces require exactly 32 video frames, got {args.video_nframes}"
        )
    existing_media = sorted(path for path in resolved_media if path.is_file())
    video_decode_stats, video_decode_errors = inspect_all_videos(
        existing_media,
        nframes=args.video_nframes,
        workers=args.video_check_workers,
    )
    stats["video_decode"] = video_decode_stats
    errors.extend(video_decode_errors)

    result = {"ok": not errors, "versions": versions, "data": stats, "errors": errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
