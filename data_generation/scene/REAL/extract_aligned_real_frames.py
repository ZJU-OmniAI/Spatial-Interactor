#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import cv2


VSI_ROOT = Path("/path/to/workspace/DATA/VSI-590K")
REAL_OFFICIAL_ROOT = Path("/path/to/workspace/DATA/REAL_OFFICIAL")
SCANNET_PACK_ROOT = REAL_OFFICIAL_ROOT / "scannet_pose_intrinsic_packs"
SCANNET_CACHE_ROOT = REAL_OFFICIAL_ROOT / "scannet_pose_intrinsic_cache"
ARKIT_RAW_ROOT = REAL_OFFICIAL_ROOT / "arkitscenes" / "raw"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"invalid_video_metadata:{video_path}")
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": (frame_count - 1) / fps if frame_count > 1 else 0.0,
    }


def extract_frames_by_index(
    video_path: Path,
    frame_indices: List[int],
    output_paths: List[Path],
) -> None:
    if len(frame_indices) != len(output_paths):
        raise ValueError("frame_indices and output_paths must have the same length")
    if not frame_indices:
        return

    pairs = sorted(zip(frame_indices, output_paths), key=lambda item: item[0])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")

    try:
        next_ptr = 0
        current_index = 0
        target_index = pairs[next_ptr][0]

        while next_ptr < len(pairs):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"failed_to_read_frame:{video_path}:wanted={target_index}:current={current_index}"
                )

            while next_ptr < len(pairs) and current_index == pairs[next_ptr][0]:
                output_path = pairs[next_ptr][1]
                ensure_dir(output_path.parent)
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"failed_to_write_frame:{output_path}")
                next_ptr += 1
                if next_ptr < len(pairs):
                    target_index = pairs[next_ptr][0]

            current_index += 1
    finally:
        cap.release()


def parse_indices_arg(raw: str | None) -> List[int] | None:
    if not raw:
        return None
    indices = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            indices.append(int(chunk))
    return sorted(set(indices))


def downsample_indices(total: int, sample_every: int, max_items: int | None) -> List[int]:
    if total <= 0:
        return []
    step = max(1, sample_every)
    chosen = list(range(0, total, step))
    if chosen[-1] != total - 1:
        chosen.append(total - 1)
    if max_items is not None and len(chosen) > max_items:
        if max_items <= 1:
            return [chosen[0]]
        return [
            chosen[min(len(chosen) - 1, round(i * (len(chosen) - 1) / (max_items - 1)))]
            for i in range(max_items)
        ]
    return chosen


def parse_arkit_traj_line(line: str) -> dict:
    values = [float(x) for x in line.strip().split()]
    return {
        "timestamp": values[0],
        "raw_values": values[1:],
        "raw_line": line.strip(),
    }


@lru_cache(maxsize=None)
def locate_arkit_scene(video_id: str) -> Path:
    direct_candidates = [
        ARKIT_RAW_ROOT / "Training" / video_id,
        ARKIT_RAW_ROOT / "Validation" / video_id,
    ]
    for scene_dir in direct_candidates:
        if (scene_dir / "lowres_wide.traj").exists():
            return scene_dir

    hits = list(ARKIT_RAW_ROOT.glob(f"**/{video_id}/lowres_wide.traj"))
    if not hits:
        raise FileNotFoundError(f"missing_arkit_traj:{video_id}")
    return hits[0].parent


def nearest_value_index(sorted_values: List[float], target: float) -> int:
    pos = bisect.bisect_left(sorted_values, target)
    if pos <= 0:
        return 0
    if pos >= len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[pos - 1]
    after = sorted_values[pos]
    return pos - 1 if abs(before - target) <= abs(after - target) else pos


def build_arkit_bundle(
    video_id: str,
    *,
    sample_every: int,
    max_items: int | None,
    pose_indices: List[int] | None,
    output_dir: Path,
    max_residual_seconds: float,
) -> Path:
    scene_dir = locate_arkit_scene(video_id)
    video_path = VSI_ROOT / "arkitscenes" / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"missing_vsi_video:{video_path}")

    traj_path = scene_dir / "lowres_wide.traj"
    intr_dir = scene_dir / "lowres_wide_intrinsics"
    intr_files = sorted(intr_dir.glob("*.pincam"), key=lambda p: float(p.stem.split("_")[-1]))
    if not intr_files:
        raise FileNotFoundError(f"missing_arkit_intrinsics:{intr_dir}")

    traj_records = [parse_arkit_traj_line(line) for line in traj_path.read_text().splitlines() if line.strip()]
    traj_timestamps = [record["timestamp"] for record in traj_records]
    intr_timestamps = [float(path.stem.split("_")[-1]) for path in intr_files]
    video_info = read_video_info(video_path)

    frame_count = video_info["frame_count"]
    if frame_count < 2:
        raise RuntimeError(f"not_enough_video_frames:{video_path}")

    official_start = intr_timestamps[0]
    official_end = intr_timestamps[-1]
    official_span = official_end - official_start
    if official_span <= 0:
        raise RuntimeError(f"invalid_intrinsics_timespan:{video_id}")

    video_span = (frame_count - 1) / video_info["fps"]
    if abs(video_span - official_span) > 0.2:
        raise RuntimeError(
            f"arkit_time_span_mismatch:{video_id}:video_span={video_span:.6f}:official_span={official_span:.6f}"
        )

    if pose_indices is None:
        chosen_pose_indices = downsample_indices(len(traj_records), sample_every, max_items)
    else:
        chosen_pose_indices = [idx for idx in pose_indices if 0 <= idx < len(traj_records)]
    if not chosen_pose_indices:
        raise RuntimeError(f"no_pose_indices_selected:{video_id}")

    def frame_official_time(frame_index: int) -> float:
        if frame_count == 1:
            return official_start
        return official_start + frame_index * official_span / (frame_count - 1)

    items = []
    residuals = []
    for pose_idx in chosen_pose_indices:
        pose_record = traj_records[pose_idx]
        pose_ts = pose_record["timestamp"]
        frame_index = round((pose_ts - official_start) / official_span * (frame_count - 1))
        frame_index = max(0, min(frame_count - 1, frame_index))
        mapped_official_ts = frame_official_time(frame_index)
        residual = abs(mapped_official_ts - pose_ts)
        if residual > max_residual_seconds:
            raise RuntimeError(
                f"arkit_residual_too_large:{video_id}:pose_idx={pose_idx}:residual={residual:.6f}"
            )
        intr_idx = nearest_value_index(intr_timestamps, mapped_official_ts)
        residuals.append(residual)
        items.append(
            {
                "pose_index": pose_idx,
                "pose_timestamp": pose_ts,
                "frame_index": frame_index,
                "mapped_official_timestamp": mapped_official_ts,
                "time_residual_seconds": residual,
                "intrinsics_timestamp": intr_timestamps[intr_idx],
                "intrinsics_path": str(intr_files[intr_idx]),
                "traj_path": str(traj_path),
                "pose_raw_values": pose_record["raw_values"],
                "pose_raw_line": pose_record["raw_line"],
            }
        )

    frame_indices = [item["frame_index"] for item in items]
    frame_paths = [output_dir / f"frame_{item['frame_index']:06d}.png" for item in items]
    extract_frames_by_index(video_path, frame_indices, frame_paths)

    for item, frame_path in zip(items, frame_paths):
        item["frame_path"] = str(frame_path)

    bundle = {
        "dataset": "arkitscenes",
        "scene_id": video_id,
        "video_path": str(video_path),
        "video_info": video_info,
        "alignment": {
            "type": "timestamp_affine_from_intrinsics",
            "official_start_timestamp": official_start,
            "official_end_timestamp": official_end,
            "official_span_seconds": official_span,
            "video_span_seconds": video_span,
            "max_residual_seconds": max(residuals),
            "mean_residual_seconds": sum(residuals) / len(residuals),
        },
        "items": items,
    }
    metadata_path = output_dir / "metadata.json"
    ensure_dir(output_dir)
    metadata_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def ensure_scannet_scene_unpacked(scene_id: str) -> Path:
    scene_dir = SCANNET_CACHE_ROOT / scene_id
    pose_dir = scene_dir / "pose"
    if pose_dir.is_dir():
        return scene_dir

    pack_path = SCANNET_PACK_ROOT / f"{scene_id}.tar.zst"
    if not pack_path.exists():
        raise FileNotFoundError(f"missing_scannet_pack:{pack_path}")

    ensure_dir(SCANNET_CACHE_ROOT)
    subprocess.run(
        ["tar", "--use-compress-program=unzstd", "-xf", str(pack_path), "-C", str(SCANNET_CACHE_ROOT)],
        check=True,
    )
    if not pose_dir.is_dir():
        raise RuntimeError(f"failed_to_unpack_scannet_pose:{scene_id}")
    return scene_dir


def build_scannet_bundle(
    scene_id: str,
    *,
    sample_every: int,
    max_items: int | None,
    pose_indices: List[int] | None,
    output_dir: Path,
) -> Path:
    scene_dir = ensure_scannet_scene_unpacked(scene_id)
    video_path = VSI_ROOT / "scannet" / f"{scene_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"missing_vsi_video:{video_path}")

    pose_dir = scene_dir / "pose"
    pose_files = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
    pose_ids = [int(path.stem) for path in pose_files]
    video_info = read_video_info(video_path)
    frame_count = video_info["frame_count"]

    expected_ids = list(range(frame_count))
    if pose_ids != expected_ids:
        raise RuntimeError(
            f"scannet_pose_video_mismatch:{scene_id}:video_frames={frame_count}:pose_files={len(pose_ids)}"
        )

    if pose_indices is None:
        chosen_pose_indices = downsample_indices(len(pose_files), sample_every, max_items)
    else:
        chosen_pose_indices = [idx for idx in pose_indices if 0 <= idx < len(pose_files)]
    if not chosen_pose_indices:
        raise RuntimeError(f"no_pose_indices_selected:{scene_id}")

    intrinsic_color = scene_dir / "intrinsic_color.txt"
    if not intrinsic_color.exists():
        raise FileNotFoundError(f"missing_intrinsic_color:{intrinsic_color}")

    items = []
    for pose_idx in chosen_pose_indices:
        pose_path = pose_files[pose_idx]
        items.append(
            {
                "pose_index": pose_idx,
                "frame_index": pose_idx,
                "pose_path": str(pose_path),
                "pose_matrix": pose_path.read_text().splitlines(),
            }
        )

    frame_indices = [item["frame_index"] for item in items]
    frame_paths = [output_dir / f"frame_{item['frame_index']:06d}.png" for item in items]
    extract_frames_by_index(video_path, frame_indices, frame_paths)

    for item, frame_path in zip(items, frame_paths):
        item["frame_path"] = str(frame_path)

    copied_intrinsic = output_dir / "intrinsic_color.txt"
    ensure_dir(output_dir)
    shutil.copy2(intrinsic_color, copied_intrinsic)

    bundle = {
        "dataset": "scannet",
        "scene_id": scene_id,
        "video_path": str(video_path),
        "video_info": video_info,
        "alignment": {
            "type": "exact_frame_index_equals_pose_id",
            "validated_pose_count": len(pose_files),
        },
        "intrinsic_color_path": str(copied_intrinsic),
        "items": items,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames aligned with official ARKitScenes / ScanNet pose assets.")
    parser.add_argument("--dataset", choices=["arkitscenes", "scannet"], required=True)
    parser.add_argument("--scene-id", required=True, help="ARKit video id or ScanNet scene id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--pose-indices", type=str, default="", help="Comma-separated pose indices to extract")
    parser.add_argument("--max-residual-seconds", type=float, default=0.05)
    args = parser.parse_args()

    pose_indices = parse_indices_arg(args.pose_indices)
    output_dir = args.output_dir / args.dataset / args.scene_id

    if args.dataset == "arkitscenes":
        metadata_path = build_arkit_bundle(
            args.scene_id,
            sample_every=args.sample_every,
            max_items=args.max_items,
            pose_indices=pose_indices,
            output_dir=output_dir,
            max_residual_seconds=args.max_residual_seconds,
        )
    else:
        metadata_path = build_scannet_bundle(
            args.scene_id,
            sample_every=args.sample_every,
            max_items=args.max_items,
            pose_indices=pose_indices,
            output_dir=output_dir,
        )

    print(metadata_path)


if __name__ == "__main__":
    main()
