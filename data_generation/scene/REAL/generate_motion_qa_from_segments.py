#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

cv2.setNumThreads(1)

SEG_ROOT = Path("/path/to/workspace/MOTION_SEGMENTS_033_FAST")
OUT_ROOT = Path("/path/to/workspace/DATA/MOTION_QA_FROM_SEGMENTS")

SCANNETV2_ROOT = Path("/path/to/workspace/Scannet-V2数据集/extracted_rgb_pose_stride10")
SCANNETPP_ROOT = Path("/path/to/workspace/Scannet++数据集/data")
MULTISCAN_ROOT = Path("/path/to/workspace/MultiScan/scans")
MULTISCAN_CACHE = Path("/path/to/workspace/MultiScan_motion_qa_cache")
THREED_SEQUENCE_ROOT = Path("/path/to/workspace/3RScan_sequence_only")
THREED_EXTRACTED_ROOT = Path("/path/to/workspace/3RScan_public_sample_rgb_pose")
ARKIT_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")
DEFAULT_ARKIT_ORIENTATION_AUDIT = Path("/path/to/workspace/SFT_QA/orientation_audit_qwen3vl8b.jsonl")

ACTION_OPTIONS = ["向前移动", "向后移动", "向左移动", "向右移动", "向左转动", "向右转动", "向上转动", "向下转动"]

_VIDEO_CACHE: dict[str, tuple[str, cv2.VideoCapture]] = {}

# Candidate magnitudes for movement-degree comparison. Older data always used
# middle-vs-end, which made the smaller/larger ratio almost always 0.5. These
# pairs produce easy/medium/hard comparisons while keeping both candidates on
# the same pure segment.
DEGREE_FRACTION_PAIRS = [
    (0.25, 1.00, "easy_025_100"),
    (0.33, 1.00, "easy_033_100"),
    (0.40, 0.80, "medium_040_080"),
    (0.50, 1.00, "medium_050_100"),
    (0.60, 1.00, "hard_060_100"),
    (0.67, 1.00, "hard_067_100"),
]

ROTATION_HORIZONTAL_LABELS = {"向左转动", "向右转动"}
ROTATION_VERTICAL_LABELS = {"向上转动", "向下转动"}
ORIENTATION_LABEL_MAP = {
    "normal": {},
    "rot180": {
        "向上转动": "向下转动",
        "向下转动": "向上转动",
        "向左转动": "向右转动",
        "向右转动": "向左转动",
        "向左移动": "向右移动",
        "向右移动": "向左移动",
    },
    "cw90": {
        "向上转动": "向右转动",
        "向下转动": "向左转动",
        "向左转动": "向上转动",
        "向右转动": "向下转动",
    },
    "ccw90": {
        "向上转动": "向左转动",
        "向下转动": "向右转动",
        "向左转动": "向下转动",
        "向右转动": "向上转动",
    },
}

# Direct two-frame tasks need the start/end images to stay visually connected.
DIRECT_ROTATION_CAP_DEG = {"horizontal": 60.0, "vertical": 40.0}
DIRECT_ROTATION_MIN_REMAINDER_DEG = {"horizontal": 10.0, "vertical": 8.0}

# Sorting is judged along the temporal chain. Each adjacent pair in the true
# order must remain visually connected, so a four-frame sorting QA may span up
# to three adjacent edges.
SORTING_ROTATION_EDGE_CAP_DEG = {"horizontal": 40.0, "vertical": 20.0}
SORTING_ROTATION_MIN_REMAINDER_DEG = {"horizontal": 30.0, "vertical": 15.0}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_exists(path: Path, retries: int = 5, delay: float = 0.2) -> bool:
    for attempt in range(retries):
        try:
            return path.exists()
        except BlockingIOError:
            if attempt + 1 == retries:
                return False
            time.sleep(delay)
        except OSError:
            return False
    return False


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def existing_segment_ids(path: Path) -> set[str]:
    segment_ids: set[str] = set()
    if not path.exists():
        return segment_ids
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            seg = row.get("gt", {}).get("segment", {})
            segment_id = seg.get("segment_id") or row.get("segment_id")
            if segment_id:
                segment_ids.add(str(segment_id))
    return segment_ids


def label_family(label: str) -> str:
    return "translation" if label in {"向前移动", "向后移动", "向左移动", "向右移动"} else "rotation"


def rotation_axis(label: str) -> str | None:
    if label in ROTATION_HORIZONTAL_LABELS:
        return "horizontal"
    if label in ROTATION_VERTICAL_LABELS:
        return "vertical"
    return None


def load_arkit_orientation_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            scene = str(row.get("scene") or "")
            orientation = str(row.get("orientation") or row.get("correction") or "normal")
            if orientation not in ORIENTATION_LABEL_MAP:
                orientation = "normal"
            if scene:
                mapping[scene] = orientation
    return mapping


def segment_orientation(seg: dict, arkit_orientation_map: dict[str, str]) -> str:
    dataset = str(seg.get("dataset") or "")
    if dataset == "3rscan":
        return "cw90"
    if dataset == "arkit":
        return arkit_orientation_map.get(str(seg.get("scene") or ""), "normal")
    return "normal"


def remap_label_for_orientation(label: str, orientation: str) -> str:
    return ORIENTATION_LABEL_MAP.get(orientation, {}).get(label, label)


def final_rotation_axis(seg: dict, orientation: str) -> str | None:
    label = str(seg.get("label") or seg.get("action_label") or "")
    return rotation_axis(remap_label_for_orientation(label, orientation))


def rotation_magnitude(seg: dict) -> float:
    return abs(float(seg.get("cumulative_value") or 0.0))


def segment_frames(seg: dict) -> list[int]:
    frames = [int(x) for x in seg.get("frame_indices") or seg.get("resampled_frame_indices") or []]
    if not frames:
        frames = [int(seg["start_frame_index"]), int(seg["end_frame_index"])]
    out: list[int] = []
    seen: set[int] = set()
    for frame in frames:
        if frame not in seen:
            out.append(frame)
            seen.add(frame)
    return out


def actual_fraction_for_frame(seg: dict, frame: int) -> float:
    start = int(seg["start_frame_index"])
    end = int(seg["end_frame_index"])
    denom = end - start
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, (int(frame) - start) / denom))


def frame_at_global_fraction(seg: dict, fraction: float) -> int:
    frames = segment_frames(seg)
    fraction = max(0.0, min(1.0, fraction))
    if len(frames) >= 2:
        pos = round(fraction * (len(frames) - 1))
        return int(frames[pos])
    start = int(seg["start_frame_index"])
    end = int(seg["end_frame_index"])
    return int(round(start + fraction * (end - start)))


def frames_for_fraction_window(seg: dict, start_fraction: float, end_fraction: float, count: int) -> list[int] | None:
    if count < 2:
        return None
    start_fraction = max(0.0, min(1.0, start_fraction))
    end_fraction = max(0.0, min(1.0, end_fraction))
    if end_fraction <= start_fraction:
        return None

    wanted = [
        frame_at_global_fraction(seg, start_fraction + (end_fraction - start_fraction) * i / (count - 1))
        for i in range(count)
    ]
    if len(set(wanted)) == count:
        return [int(x) for x in wanted]

    start_frame = frame_at_global_fraction(seg, start_fraction)
    end_frame = frame_at_global_fraction(seg, end_fraction)
    lo, hi = sorted((start_frame, end_frame))
    available = [frame for frame in segment_frames(seg) if lo <= frame <= hi]
    if len(available) < count:
        return None
    picked = uniform_pick(available, count)
    if len(set(picked)) < count:
        return None
    return picked


def rotation_value_between_frames(seg: dict, start_frame: int, end_frame: int) -> float:
    full_value = rotation_magnitude(seg)
    start_fraction = actual_fraction_for_frame(seg, start_frame)
    end_fraction = actual_fraction_for_frame(seg, end_frame)
    return full_value * abs(end_fraction - start_fraction)


def adjacent_rotation_values(seg: dict, frame_indices: Sequence[int]) -> list[float]:
    frames = [int(x) for x in frame_indices]
    return [rotation_value_between_frames(seg, frames[i], frames[i + 1]) for i in range(len(frames) - 1)]


def scaled_number(value: object, scale: float) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) * scale
    return value


def scaled_numeric_dict(value: object, scale: float) -> object:
    if not isinstance(value, dict):
        return value
    return {key: scaled_number(item, scale) for key, item in value.items()}


def windowed_segment(
    seg: dict,
    frame_indices: Sequence[int],
    tag: str,
    window_index: int,
    total_windows: int,
    *,
    orientation_for_cap: str = "normal",
    cap_axis: str | None = None,
    cap_value: float | None = None,
) -> dict:
    frames = [int(x) for x in frame_indices]
    out = copy.deepcopy(seg)
    original_segment_id = str(seg.get("segment_id") or "")
    start_fraction = actual_fraction_for_frame(seg, frames[0])
    end_fraction = actual_fraction_for_frame(seg, frames[-1])
    span_fraction = max(0.0, min(1.0, abs(end_fraction - start_fraction)))
    full_value = rotation_magnitude(seg)
    value = full_value * span_fraction if full_value > 0 else abs(float(seg.get("cumulative_value") or 0.0))
    scale = value / full_value if full_value > 1e-9 else span_fraction

    out["segment_id"] = f"{original_segment_id}__{tag}_{window_index:02d}"
    out["original_segment_id"] = original_segment_id
    out["start_frame_index"] = frames[0]
    out["end_frame_index"] = frames[-1]
    out["frame_gap"] = abs(frames[-1] - frames[0])
    out["frame_indices"] = [int(x) for x in segment_frames(seg) if min(frames[0], frames[-1]) <= int(x) <= max(frames[0], frames[-1])]
    if not out["frame_indices"]:
        out["frame_indices"] = frames
    out["resampled_frame_indices"] = out["frame_indices"]
    out["cumulative_value"] = float(value)
    out["component_sums"] = scaled_numeric_dict(out.get("component_sums"), scale)
    out["component_abs_sums"] = scaled_numeric_dict(out.get("component_abs_sums"), scale)
    out["num_steps_033"] = max(1, int(round(float(seg.get("num_steps_033") or len(out["frame_indices"]) - 1) * span_fraction)))
    label = str(out.get("label") or out.get("action_label") or "")
    if label:
        out["step_label_counts"] = {label: int(out["num_steps_033"])}
        out["action_label"] = label
    out["rotation_cap_window"] = {
        "version": "rotation_cap_window_v1",
        "tag": tag,
        "window_index": window_index,
        "total_windows": total_windows,
        "original_segment_id": original_segment_id,
        "original_cumulative_value": float(seg.get("cumulative_value") or 0.0),
        "orientation_for_cap": orientation_for_cap,
        "final_label_for_cap": remap_label_for_orientation(label, orientation_for_cap) if label else None,
        "cap_axis": cap_axis,
        "cap_value": cap_value,
        "start_fraction": float(start_fraction),
        "end_fraction": float(end_fraction),
        "span_fraction": float(span_fraction),
        "window_cumulative_value": float(value),
    }
    return out


def rotation_fraction_windows(total_value: float, cap: float, min_remainder: float) -> list[tuple[float, float]]:
    if total_value <= 1e-9:
        return [(0.0, 1.0)]
    if total_value <= cap:
        return [(0.0, 1.0)]
    windows: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor + cap <= total_value + 1e-9:
        windows.append((cursor / total_value, min(1.0, (cursor + cap) / total_value)))
        cursor += cap
    remainder = total_value - cursor
    if remainder >= min_remainder:
        windows.append((cursor / total_value, 1.0))
    return windows or [(0.0, 1.0)]


def direct_motion_segments(seg: dict, orientation_for_cap: str = "normal") -> list[dict]:
    axis = final_rotation_axis(seg, orientation_for_cap)
    if seg.get("family") != "rotation" or axis is None:
        return [seg]
    total = rotation_magnitude(seg)
    windows = rotation_fraction_windows(total, DIRECT_ROTATION_CAP_DEG[axis], DIRECT_ROTATION_MIN_REMAINDER_DEG[axis])
    out: list[dict] = []
    for idx, (start_fraction, end_fraction) in enumerate(windows, start=1):
        frames = frames_for_fraction_window(seg, start_fraction, end_fraction, 2)
        if frames is None or len(set(frames)) < 2:
            continue
        if rotation_value_between_frames(seg, frames[0], frames[-1]) > DIRECT_ROTATION_CAP_DEG[axis] + 1e-6:
            continue
        out.append(
            windowed_segment(
                seg,
                frames,
                "direct_rotcap",
                idx,
                len(windows),
                orientation_for_cap=orientation_for_cap,
                cap_axis=axis,
                cap_value=DIRECT_ROTATION_CAP_DEG[axis],
            )
        )
    return out


def action_value_text(label: str, unit: str, value: float) -> str:
    if unit == "deg":
        return f"{label}约{value:.1f}度"
    return f"{label}约{value:.2f}米"


def read_image(path: Path) -> np.ndarray | None:
    if not path_exists(path):
        return None
    try:
        return cv2.imread(str(path))
    except Exception:
        return None


def read_scannetv2(scene: str, frame_index: int) -> np.ndarray | None:
    color_dir = SCANNETV2_ROOT / scene / "color"
    idx = int(frame_index)
    for name in (f"{idx:06d}.jpg", f"{idx}.jpg", f"{idx:06d}.png", f"{idx}.png"):
        frame = read_image(color_dir / name)
        if frame is not None:
            return frame
    return None


def read_scannetpp(scene: str, frame_index: int) -> np.ndarray | None:
    return read_video_cached("scannetpp", SCANNETPP_ROOT / scene / "iphone" / "rgb.mkv", frame_index)


def multiscan_mp4(scene: str) -> Path | None:
    ensure_dir(MULTISCAN_CACHE)
    cached = MULTISCAN_CACHE / f"{scene}.mp4"
    if cached.exists() and cached.stat().st_size > 1024 * 1024:
        return cached
    zip_path = MULTISCAN_ROOT / f"{scene}.zip"
    if not zip_path.exists():
        return None
    tmp = cached.with_suffix(".mp4.tmp")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"{scene}/{scene}.mp4") as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        tmp.replace(cached)
        return cached
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def read_multiscan(scene: str, frame_index: int) -> np.ndarray | None:
    mp4 = multiscan_mp4(scene)
    if mp4 is None:
        return None
    return read_video_cached("multiscan", mp4, frame_index)


def read_video_cached(cache_key: str, video_path: Path, frame_index: int) -> np.ndarray | None:
    video_path_str = str(video_path)
    cached = _VIDEO_CACHE.get(cache_key)
    if cached is None or cached[0] != video_path_str or not cached[1].isOpened():
        if cached is not None:
            cached[1].release()
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            cap.release()
            _VIDEO_CACHE.pop(cache_key, None)
            return None
        _VIDEO_CACHE[cache_key] = (video_path_str, cap)
    else:
        cap = cached[1]

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def read_3rscan(scene: str, frame_index: int) -> np.ndarray | None:
    extracted = THREED_EXTRACTED_ROOT / scene / "sequence" / f"frame-{int(frame_index):06d}.color.jpg"
    if extracted.exists():
        return read_image(extracted)
    zip_path = THREED_SEQUENCE_ROOT / scene / "sequence.zip"
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            data = zf.read(f"frame-{int(frame_index):06d}.color.jpg")
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def read_arkit(scene: str, frame_index: int) -> np.ndarray | None:
    return read_image(ARKIT_ROOT / scene / "color" / f"{int(frame_index):06d}.jpg")


def read_frame(dataset: str, scene: str, frame_index: int) -> np.ndarray | None:
    if dataset == "scannetv2":
        return read_scannetv2(scene, frame_index)
    if dataset == "scannetpp":
        return read_scannetpp(scene, frame_index)
    if dataset == "multiscan":
        return read_multiscan(scene, frame_index)
    if dataset == "3rscan":
        return read_3rscan(scene, frame_index)
    if dataset == "arkit":
        return read_arkit(scene, frame_index)
    return None


def load_frames(dataset: str, scene: str, frame_indices: Sequence[int]) -> list[np.ndarray] | None:
    frames = []
    for frame_index in frame_indices:
        frame = read_frame(dataset, scene, int(frame_index))
        if frame is None:
            return None
        frames.append(frame)
    return frames


def resize_for_sheet(frame: np.ndarray, width: int = 360) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    height = max(1, int(round(h * (width / w))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def save_sample(sample_dir: Path, names_and_frames: Sequence[tuple[str, np.ndarray]]) -> dict[str, str]:
    ensure_dir(sample_dir)
    outputs: dict[str, str] = {}
    smalls = []
    for name, frame in names_and_frames:
        path = sample_dir / f"{name}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        outputs[name] = str(path)
        smalls.append(resize_for_sheet(frame))
    min_h = min(x.shape[0] for x in smalls)
    smalls = [x[:min_h] for x in smalls]
    contact = sample_dir / "contact_sheet.jpg"
    cv2.imwrite(str(contact), np.concatenate(smalls, axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    outputs["contact_sheet"] = str(contact)
    return outputs


def uniform_pick(values: Sequence[int], count: int) -> list[int]:
    if len(values) <= count:
        return [int(x) for x in values]
    return [int(values[round(i * (len(values) - 1) / (count - 1))]) for i in range(count)]


def middle_frame(seg: dict) -> int:
    frames = [int(x) for x in seg.get("frame_indices") or seg.get("resampled_frame_indices") or []]
    if frames:
        return frames[len(frames) // 2]
    return int(round((int(seg["start_frame_index"]) + int(seg["end_frame_index"])) / 2))


def degree_fraction_pair(seg: dict) -> tuple[float, float, str]:
    key = str(seg.get("segment_id") or f"{seg.get('dataset')}:{seg.get('scene')}:{seg.get('start_frame_index')}:{seg.get('end_frame_index')}")
    idx = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(DEGREE_FRACTION_PAIRS)
    return DEGREE_FRACTION_PAIRS[idx]


def frame_at_fraction(seg: dict, fraction: float) -> tuple[int, float]:
    frames = [int(x) for x in seg.get("frame_indices") or seg.get("resampled_frame_indices") or []]
    if len(frames) >= 2:
        pos = round(max(0.0, min(1.0, fraction)) * (len(frames) - 1))
        frame = int(frames[pos])
    else:
        start = int(seg["start_frame_index"])
        end = int(seg["end_frame_index"])
        frame = int(round(start + max(0.0, min(1.0, fraction)) * (end - start)))

    start = int(seg["start_frame_index"])
    end = int(seg["end_frame_index"])
    denom = end - start
    if denom == 0:
        actual_fraction = max(0.0, min(1.0, fraction))
    else:
        actual_fraction = max(0.0, min(1.0, (frame - start) / denom))
    return frame, actual_fraction


def degree_candidates(seg: dict) -> tuple[list[tuple[str, int, float]], str] | None:
    f_small, f_large, difficulty = degree_fraction_pair(seg)
    small_frame, small_fraction = frame_at_fraction(seg, f_small)
    large_frame, large_fraction = frame_at_fraction(seg, f_large)
    start = int(seg["start_frame_index"])
    if len({start, small_frame, large_frame}) < 3 or small_fraction >= large_fraction:
        mid = middle_frame(seg)
        end = int(seg["end_frame_index"])
        if len({start, mid, end}) < 3:
            return None
        denom = max(1, end - start)
        small_frame, large_frame = mid, end
        small_fraction = max(0.05, min(0.95, (mid - start) / denom))
        large_fraction = 1.0
        difficulty = "fallback_050_100"

    full_value = float(seg["cumulative_value"])
    return [
        ("A", small_frame, full_value * small_fraction),
        ("B", large_frame, full_value * large_fraction),
    ], difficulty


def iter_segments(segment_root: Path, datasets: Sequence[str]) -> Iterable[dict]:
    for dataset in datasets:
        if dataset == "arkit":
            paths = sorted((segment_root / "_arkit_shards").glob("shard_*/arkit/segments.jsonl"))
        else:
            paths = [segment_root / dataset / "segments.jsonl"]
        for path in paths:
            if not path_exists(path):
                continue
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("is_pure") and row.get("segment_type") in {"短", "长"}:
                        yield row


def build_action(seg: dict, out_root: Path, counters: Counter) -> dict | None:
    dataset, scene = seg["dataset"], seg["scene"]
    start = int(seg["start_frame_index"])
    end = int(seg["end_frame_index"])
    frames = load_frames(dataset, scene, [start, end])
    if frames is None:
        return None
    qa_id = f"{dataset}_action_inference_{counters['action_inference'] + 1:06d}"
    paths = save_sample(out_root / "action_inference" / qa_id, [("frame_A", frames[0]), ("frame_B", frames[1])])
    return {
        "qa_id": qa_id,
        "task_type": "action_inference",
        "dataset": dataset,
        "scene": scene,
        "segment_id": seg["segment_id"],
        "question": "图A到图B之间，相机主要执行了什么动作？",
        "answer": seg["label"],
        "answer_with_value": action_value_text(seg["label"], seg["unit"], float(seg["cumulative_value"])),
        "options": ACTION_OPTIONS,
        "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], "frame_A": paths["frame_A"], "frame_B": paths["frame_B"], "contact_sheet": paths["contact_sheet"]},
        "gt": {"segment": seg, "unit": seg["unit"], "value": float(seg["cumulative_value"]), "frame_indices": [start, end]},
    }


def build_short_rows(seg: dict, out_root: Path, counters: Counter, rng: random.Random) -> tuple[dict | None, dict | None]:
    dataset, scene = seg["dataset"], seg["scene"]
    start = int(seg["start_frame_index"])
    mid = middle_frame(seg)
    end = int(seg["end_frame_index"])
    if mid == start or mid == end:
        frames = load_frames(dataset, scene, [start, end])
        if frames is None:
            return None, None
        qa_id = f"{dataset}_action_inference_{counters['action_inference'] + 1:06d}"
        paths = save_sample(out_root / "action_inference" / qa_id, [("frame_A", frames[0]), ("frame_B", frames[1])])
        action_row = {
            "qa_id": qa_id,
            "task_type": "action_inference",
            "dataset": dataset,
            "scene": scene,
            "segment_id": seg["segment_id"],
            "question": "图A到图B之间，相机主要执行了什么动作？",
            "answer": seg["label"],
            "answer_with_value": action_value_text(seg["label"], seg["unit"], float(seg["cumulative_value"])),
            "options": ACTION_OPTIONS,
            "input": {"frame_paths": [paths["frame_A"], paths["frame_B"]], "frame_A": paths["frame_A"], "frame_B": paths["frame_B"], "contact_sheet": paths["contact_sheet"]},
            "gt": {"segment": seg, "unit": seg["unit"], "value": float(seg["cumulative_value"]), "frame_indices": [start, end]},
        }
        return action_row, None

    action_frames = load_frames(dataset, scene, [start, end])
    if action_frames is None:
        return None, None
    qa_id = f"{dataset}_action_inference_{counters['action_inference'] + 1:06d}"
    action_paths = save_sample(out_root / "action_inference" / qa_id, [("frame_A", action_frames[0]), ("frame_B", action_frames[1])])
    action_row = {
        "qa_id": qa_id,
        "task_type": "action_inference",
        "dataset": dataset,
        "scene": scene,
        "segment_id": seg["segment_id"],
        "question": "图A到图B之间，相机主要执行了什么动作？",
        "answer": seg["label"],
        "answer_with_value": action_value_text(seg["label"], seg["unit"], float(seg["cumulative_value"])),
        "options": ACTION_OPTIONS,
        "input": {"frame_paths": [action_paths["frame_A"], action_paths["frame_B"]], "frame_A": action_paths["frame_A"], "frame_B": action_paths["frame_B"], "contact_sheet": action_paths["contact_sheet"]},
        "gt": {"segment": seg, "unit": seg["unit"], "value": float(seg["cumulative_value"]), "frame_indices": [start, end]},
    }

    cand, difficulty = degree_candidates(seg) or (None, None)
    if cand is None:
        return action_row, None
    candidate_indices = [idx for _label, idx, _value in cand]
    frames = load_frames(dataset, scene, [start] + candidate_indices)
    if frames is None:
        return action_row, None

    candidates = [(label, frames[i + 1], idx, value) for i, (label, idx, value) in enumerate(cand)]
    if rng.random() < 0.5:
        candidates.reverse()
    answer_label = candidates[0][0] if candidates[0][3] > candidates[1][3] else candidates[1][0]
    qa_id = f"{dataset}_movement_degree_comparison_{counters['movement_degree_comparison'] + 1:06d}"
    names = [("frame_start", frames[0])] + [(f"frame_{label}", frame) for label, frame, _idx, _value in candidates]
    paths = save_sample(out_root / "movement_degree_comparison" / qa_id, names)
    unit_cn = "度" if seg["unit"] == "deg" else "米"
    compare_row = {
        "qa_id": qa_id,
        "task_type": "movement_degree_comparison",
        "dataset": dataset,
        "scene": scene,
        "segment_id": seg["segment_id"],
        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
        "answer": f"候选图{answer_label}更大。",
        "answer_with_value": f"候选图{candidates[0][0]}约{candidates[0][3]:.1f}{unit_cn}，候选图{candidates[1][0]}约{candidates[1][3]:.1f}{unit_cn}。" if seg["unit"] == "deg" else f"候选图{candidates[0][0]}约{candidates[0][3]:.2f}{unit_cn}，候选图{candidates[1][0]}约{candidates[1][3]:.2f}{unit_cn}。",
        "input": {
            "frame_paths": [paths["frame_start"], paths["frame_A"], paths["frame_B"]],
            "first_frame": paths["frame_start"],
            "candidate_frames": {"A": paths["frame_A"], "B": paths["frame_B"]},
            "contact_sheet": paths["contact_sheet"],
        },
        "gt": {
            "segment": seg,
            "larger": answer_label,
            "unit": seg["unit"],
            "candidate_frame_indices": {label: int(idx) for label, _frame, idx, _value in candidates},
            "candidate_values_approx": {label: float(value) for label, _frame, _idx, value in candidates},
            "candidate_fraction_difficulty": difficulty,
            "candidate_fractions_approx": {
                label: float(value) / max(abs(float(seg["cumulative_value"])), 1e-9)
                for label, _frame, _idx, value in candidates
            },
            "construction": "same pure segment: diversified candidate fractions",
        },
    }
    return action_row, compare_row


def build_comparison(seg: dict, out_root: Path, counters: Counter, rng: random.Random) -> dict | None:
    dataset, scene = seg["dataset"], seg["scene"]
    start = int(seg["start_frame_index"])
    cand, difficulty = degree_candidates(seg) or (None, None)
    if cand is None:
        return None
    candidate_indices = [idx for _label, idx, _value in cand]
    frames = load_frames(dataset, scene, [start] + candidate_indices)
    if frames is None:
        return None
    candidates = [
        (label, f"fraction_{i}", frames[i + 1], idx, value)
        for i, (label, idx, value) in enumerate(cand)
    ]
    if rng.random() < 0.5:
        candidates.reverse()
    answer_label = candidates[0][0] if candidates[0][4] > candidates[1][4] else candidates[1][0]
    qa_id = f"{dataset}_movement_degree_comparison_{counters['movement_degree_comparison'] + 1:06d}"
    names = [("frame_start", frames[0])] + [(f"frame_{label}", frame) for label, _kind, frame, _idx, _value in candidates]
    paths = save_sample(out_root / "movement_degree_comparison" / qa_id, names)
    unit_cn = "度" if seg["unit"] == "deg" else "米"
    return {
        "qa_id": qa_id,
        "task_type": "movement_degree_comparison",
        "dataset": dataset,
        "scene": scene,
        "segment_id": seg["segment_id"],
        "question": "给定第一张起始图，以及候选图A和候选图B。相对于起始图，哪一张候选图对应的相机运动幅度更大？",
        "answer": f"候选图{answer_label}更大。",
        "answer_with_value": f"候选图{candidates[0][0]}约{candidates[0][4]:.1f}{unit_cn}，候选图{candidates[1][0]}约{candidates[1][4]:.1f}{unit_cn}。" if seg["unit"] == "deg" else f"候选图{candidates[0][0]}约{candidates[0][4]:.2f}{unit_cn}，候选图{candidates[1][0]}约{candidates[1][4]:.2f}{unit_cn}。",
        "input": {
            "frame_paths": [paths["frame_start"], paths["frame_A"], paths["frame_B"]],
            "first_frame": paths["frame_start"],
            "candidate_frames": {"A": paths["frame_A"], "B": paths["frame_B"]},
            "contact_sheet": paths["contact_sheet"],
        },
        "gt": {
            "segment": seg,
            "larger": answer_label,
            "unit": seg["unit"],
            "candidate_frame_indices": {label: int(idx) for label, _kind, _frame, idx, _value in candidates},
            "candidate_values_approx": {label: float(value) for label, _kind, _frame, _idx, value in candidates},
            "candidate_fraction_difficulty": difficulty,
            "candidate_fractions_approx": {
                label: float(value) / max(abs(float(seg["cumulative_value"])), 1e-9)
                for label, _kind, _frame, _idx, value in candidates
            },
            "construction": "same pure segment: diversified candidate fractions",
        },
    }


def sorting_windows(seg: dict, orientation_for_cap: str = "normal") -> list[tuple[dict, list[int]]]:
    axis = final_rotation_axis(seg, orientation_for_cap)
    if seg.get("family") == "rotation" and axis is not None:
        total = rotation_magnitude(seg)
        cap = SORTING_ROTATION_EDGE_CAP_DEG[axis] * 3.0
        windows = rotation_fraction_windows(total, cap, SORTING_ROTATION_MIN_REMAINDER_DEG[axis])
        out: list[tuple[dict, list[int]]] = []
        for idx, (start_fraction, end_fraction) in enumerate(windows, start=1):
            frame_indices = frames_for_fraction_window(seg, start_fraction, end_fraction, 4)
            if frame_indices is None or len(set(frame_indices)) < 4:
                continue
            if any(value > SORTING_ROTATION_EDGE_CAP_DEG[axis] + 1e-6 for value in adjacent_rotation_values(seg, frame_indices)):
                continue
            window_seg = windowed_segment(
                seg,
                frame_indices,
                "sorting_rotcap",
                idx,
                len(windows),
                orientation_for_cap=orientation_for_cap,
                cap_axis=axis,
                cap_value=SORTING_ROTATION_EDGE_CAP_DEG[axis],
            )
            out.append((window_seg, frame_indices))
        return out

    frames = [int(x) for x in seg.get("frame_indices") or seg.get("resampled_frame_indices") or []]
    if len(frames) < 4:
        frames = [int(seg["start_frame_index"]), int(seg["end_frame_index"])]
    if len(frames) < 4:
        return []
    steps = int(seg.get("num_steps_033") or 0)
    if steps >= 8 and len(frames) >= 8:
        mid = len(frames) // 2
        windows = [frames[: mid + 1], frames[mid:]]
        return [(seg, uniform_pick(window, 4)) for window in windows if len(window) >= 4]
    return [(seg, uniform_pick(frames, 4))]


def build_sorting(seg: dict, out_root: Path, counters: Counter, rng: random.Random, frame_indices: Sequence[int]) -> dict | None:
    dataset, scene = seg["dataset"], seg["scene"]
    frames = load_frames(dataset, scene, frame_indices)
    if frames is None:
        return None
    candidate_items = [("A", int(frame_indices[1]), frames[1]), ("B", int(frame_indices[2]), frames[2]), ("C", int(frame_indices[3]), frames[3])]
    rng.shuffle(candidate_items)
    correct_order = [label for label, _idx, _frame in sorted(candidate_items, key=lambda x: x[1])]
    qa_id = f"{dataset}_movement_sequence_sorting_{counters['movement_sequence_sorting'] + 1:06d}"
    names = [("frame_start", frames[0])] + [(f"frame_{label}", frame) for label, _idx, frame in candidate_items]
    paths = save_sample(out_root / "movement_sequence_sorting" / qa_id, names)
    return {
        "qa_id": qa_id,
        "task_type": "movement_sequence_sorting",
        "dataset": dataset,
        "scene": scene,
        "segment_id": seg["segment_id"],
        "question": "已知第一张图是起始帧。请将其余三张候选图按真实视频中的时间先后排序。",
        "answer": " -> ".join(correct_order),
        "input": {
            "frame_paths": [paths["frame_start"]] + [paths[f"frame_{label}"] for label, _idx, _frame in candidate_items],
            "first_frame": paths["frame_start"],
            "candidate_frames": {label: paths[f"frame_{label}"] for label, _idx, _frame in candidate_items},
            "contact_sheet": paths["contact_sheet"],
        },
        "gt": {
            "segment": seg,
            "shown_frame_indices": [int(x) for x in frame_indices],
            "candidate_frame_indices": {label: int(idx) for label, idx, _frame in candidate_items},
            "correct_order": correct_order,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA rows from precomputed pure motion segments.")
    parser.add_argument("--segment-root", type=Path, default=SEG_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--datasets", default="3rscan,scannetv2,scannetpp,multiscan,arkit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append to existing qa_data.jsonl files instead of clearing them.")
    parser.add_argument("--skip-existing-segments", action="store_true", help="When appending, skip rows whose segment_id already exists in a task jsonl.")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--arkit-orientation-audit", type=Path, default=DEFAULT_ARKIT_ORIENTATION_AUDIT)
    args = parser.parse_args()

    if args.overwrite and args.append:
        raise ValueError("--overwrite and --append cannot be used together")

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)
    for task in ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]:
        ensure_dir(args.output_root / task)
        if not args.append:
            (args.output_root / task / "qa_data.jsonl").unlink(missing_ok=True)

    rng = random.Random(20260517)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    counters: Counter = Counter()
    if args.append:
        for task in ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]:
            counters[task] = count_jsonl(args.output_root / task / "qa_data.jsonl")
    existing_by_task = {
        task: existing_segment_ids(args.output_root / task / "qa_data.jsonl") if args.skip_existing_segments else set()
        for task in ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]
    }
    skipped: Counter = Counter()
    label_counts: dict[str, Counter] = defaultdict(Counter)
    processed = 0
    bad_visual_segments: set[str] = set()
    arkit_orientation_map = load_arkit_orientation_map(args.arkit_orientation_audit)

    for seg in iter_segments(args.segment_root, datasets):
        processed += 1
        if args.max_segments and processed > args.max_segments:
            break
        segment_id = str(seg.get("segment_id", ""))
        orientation_for_cap = segment_orientation(seg, arkit_orientation_map)
        if seg["segment_type"] == "短":
            if segment_id and segment_id in existing_by_task["action_inference"] and segment_id in existing_by_task["movement_degree_comparison"]:
                skipped["existing_short_segment"] += 1
                continue
            direct_segments = direct_motion_segments(seg, orientation_for_cap)
            if not direct_segments:
                skipped["direct_rotation_window_empty"] += 1
                continue
            for direct_seg in direct_segments:
                action_row, compare_row = build_short_rows(direct_seg, args.output_root, counters, rng)
                if action_row is None:
                    skipped["action_inference"] += 1
                    bad_visual_segments.add(direct_seg["segment_id"])
                else:
                    append_jsonl(args.output_root / "action_inference" / "qa_data.jsonl", action_row)
                    counters["action_inference"] += 1
                    label_counts["action_inference"][action_row["answer"]] += 1
                if compare_row is None:
                    if direct_seg["segment_id"] in bad_visual_segments:
                        skipped["movement_degree_comparison_visual_precheck"] += 1
                    else:
                        skipped["movement_degree_comparison"] += 1
                else:
                    append_jsonl(args.output_root / "movement_degree_comparison" / "qa_data.jsonl", compare_row)
                    counters["movement_degree_comparison"] += 1
                    label_counts["movement_degree_comparison"][direct_seg["label"]] += 1
        elif seg["segment_type"] == "长":
            if segment_id and segment_id in existing_by_task["movement_sequence_sorting"]:
                skipped["existing_long_segment"] += 1
                continue
            made = 0
            for sorting_seg, window in sorting_windows(seg, orientation_for_cap):
                row = build_sorting(sorting_seg, args.output_root, counters, rng, window)
                if row is None:
                    skipped["movement_sequence_sorting"] += 1
                    continue
                append_jsonl(args.output_root / "movement_sequence_sorting" / "qa_data.jsonl", row)
                counters["movement_sequence_sorting"] += 1
                label_counts["movement_sequence_sorting"][seg["label"]] += 1
                made += 1
            if made == 0:
                skipped["movement_sequence_sorting_empty"] += 1
        if processed % 100 == 0:
            print(f"[progress] segments={processed} action={counters['action_inference']} compare={counters['movement_degree_comparison']} sort={counters['movement_sequence_sorting']} skipped={dict(skipped)}", flush=True)

    for task in ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]:
        rows = []
        path = args.output_root / task / "qa_data.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        write_json(args.output_root / task / "qa_data.json", rows)

    summary = {
        "output_root": str(args.output_root),
        "segment_root": str(args.segment_root),
        "arkit_orientation_audit": str(args.arkit_orientation_audit),
        "datasets": datasets,
        "pure_segments_processed": processed,
        "qa_counts": dict(counters),
        "skipped_counts": dict(skipped),
        "label_counts": {task: dict(counter) for task, counter in label_counts.items()},
        "policy": {
            "short_segments": [
                "action_inference_direct_rotation_cap_horizontal_60_vertical_40",
                "movement_degree_comparison_direct_rotation_cap_horizontal_60_vertical_40_diversified_fraction_pairs",
            ],
            "long_segments": [
                "movement_sequence_sorting_adjacent_rotation_cap_horizontal_40_vertical_20",
                "two_sorting_windows_when_num_steps_033_at_least_8_for_non_rotation",
            ],
        },
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
