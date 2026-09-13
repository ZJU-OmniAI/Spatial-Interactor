from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


DATA_ROOT = Path("/path/to/workspace/DATA")
VSI_ROOT = DATA_ROOT / "VSI-590K"
VLM_ROOT = DATA_ROOT / "VLM-3R-DATA"

SCANNET_VIDEO_ROOT = VSI_ROOT / "scannet"
SCANNETPP_VIDEO_ROOT = VSI_ROOT / "scannetppv2"
ARKIT_VIDEO_ROOT = VSI_ROOT / "arkitscenes"

ANNOTATION_FILES: Dict[str, Path] = {
    "action_inference": VLM_ROOT / "vstibench_train" / "qa_camera_movement_direction_v1.json",
    "parallax_depth_inference": VLM_ROOT / "vstibench_train" / "qa_camera_obj_rel_dist_v3.json",
    "movement_degree_source": VLM_ROOT / "vstibench_train" / "qa_camera_displacement.json",
    "overlap_localization": VSI_ROOT / "vsi_590k.jsonl",
    "route_planning": VLM_ROOT / "vsibench_train" / "merged_qa_route_plan_train.json",
}

FRAME_RANGE_RE = re.compile(r"frame\s+(\d+)\s+and\s+frame\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
FRAME_SINGLE_RE = re.compile(r"frame\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
OPTION_LINE_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$")


@dataclass
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_source_name(video_ref: str) -> str:
    ref = str(video_ref).strip()
    if not ref:
        return "unknown"
    lowered = ref.lower()
    if lowered.startswith("scannetpp") or lowered.startswith("scannetppv2"):
        return "scannetpp"
    if lowered.startswith("scannet"):
        return "scannet"
    if lowered.startswith("arkitscenes"):
        return "arkitscenes"
    return ref.split("/")[0].lower()


def resolve_video_path(video_ref: str) -> Optional[Path]:
    ref = str(video_ref).strip()
    if not ref:
        return None
    name = Path(ref).name
    lowered = ref.lower()
    if lowered.startswith("scannetpp") or lowered.startswith("scannetppv2"):
        path = SCANNETPP_VIDEO_ROOT / name
    elif lowered.startswith("scannet"):
        path = SCANNET_VIDEO_ROOT / name
    elif lowered.startswith("arkitscenes"):
        path = ARKIT_VIDEO_ROOT / name
    else:
        return None
    return path if path.exists() else None


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return VideoInfo(video_path, frame_count, fps, width, height)
    finally:
        cap.release()


def _uniform_frame_indices(frame_count: int, total_slots: int = 32) -> List[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if total_slots <= 1:
        return [0]
    if frame_count == 1:
        return [0] * total_slots
    return [
        min(frame_count - 1, max(0, int(round(i * (frame_count - 1) / (total_slots - 1)))))
        for i in range(total_slots)
    ]


def frame_slot_to_index(frame_count: int, slot_id_1based: int, total_slots: int = 32) -> int:
    slot = max(1, min(total_slots, int(slot_id_1based)))
    return _uniform_frame_indices(frame_count, total_slots=total_slots)[slot - 1]


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed_to_read_frame:{video_path}:{frame_index}")
        return frame
    finally:
        cap.release()


def save_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    frame = read_frame(video_path, frame_index)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise RuntimeError(f"failed_to_write_frame:{output_path}")


def parse_frame_range(question: str) -> Optional[Tuple[int, int, int]]:
    match = FRAME_RANGE_RE.search(question)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_single_frame(question: str) -> Optional[Tuple[int, int]]:
    match = FRAME_SINGLE_RE.search(question)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for line in question.splitlines():
        match = OPTION_LINE_RE.match(line)
        if match:
            options[match.group(1).strip()] = match.group(2).strip()
    return options


def answer_text_from_record(record: dict) -> str:
    conversations = record.get("conversations") or []
    if len(conversations) < 2:
        return ""
    raw_answer = str(conversations[1].get("value", "")).strip()
    options = parse_options(str(conversations[0].get("value", "")))
    if raw_answer in options:
        return options[raw_answer]
    return raw_answer


def clean_question_text(question: str) -> str:
    text = str(question).replace("<image>", "").strip()
    return re.sub(r"\n{2,}", "\n", text)


def sample_records(
    records: Sequence[dict],
    *,
    limit: int,
    source_order: Optional[Sequence[str]] = None,
    predicate=None,
) -> List[dict]:
    chosen: List[dict] = []
    seen_videos = set()
    ordered_sources = list(source_order or [])
    if ordered_sources:
        grouped: Dict[str, List[dict]] = {src: [] for src in ordered_sources}
        fallback: List[dict] = []
        for record in records:
            if predicate and not predicate(record):
                continue
            source = extract_source_name(record.get("video", ""))
            if source in grouped:
                grouped[source].append(record)
            else:
                fallback.append(record)
        for source in ordered_sources:
            for record in grouped[source]:
                video = record.get("video")
                if video in seen_videos:
                    continue
                chosen.append(record)
                seen_videos.add(video)
                if len(chosen) >= limit:
                    return chosen
        for record in fallback:
            video = record.get("video")
            if video in seen_videos:
                continue
            chosen.append(record)
            seen_videos.add(video)
            if len(chosen) >= limit:
                return chosen
        return chosen

    for record in records:
        if predicate and not predicate(record):
            continue
        video = record.get("video")
        if video in seen_videos:
            continue
        chosen.append(record)
        seen_videos.add(video)
        if len(chosen) >= limit:
            break
    return chosen


def build_row_base(
    *,
    qa_id: str,
    task_type: str,
    source_dataset: str,
    source_name: str,
    video_ref: str,
    video_path: Path,
    question: str,
    answer: str,
    input_paths: List[Path],
    metadata: dict,
) -> dict:
    return {
        "qa_id": qa_id,
        "task_type": task_type,
        "source_dataset": source_dataset,
        "source_name": source_name,
        "video_ref": video_ref,
        "video_path": str(video_path),
        "question": question,
        "answer": answer,
        "input": {
            "frame_paths": [str(path) for path in input_paths],
        },
        "metadata": metadata,
    }


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def try_parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except Exception:
        return None


def shuffled_labels(count: int, seed: int) -> List[str]:
    labels = [chr(ord("A") + i) for i in range(count)]
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels
