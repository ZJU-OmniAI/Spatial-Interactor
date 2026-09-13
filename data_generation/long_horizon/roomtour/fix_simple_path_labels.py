#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SRC = Path("/path/to/workspace/combined_camera_roomtour_metric_m_no_boundary_20260529/combined_camera_roomtour_metric_m_no_boundary.jsonl")
OUT_DIR = Path("/path/to/workspace/combined_camera_roomtour_simplepath_fixed_20260603")
OUT = OUT_DIR / "combined_camera_roomtour_simplepath_fixed.jsonl"
SUMMARY = OUT_DIR / "simplepath_fix_summary.json"
GEO_PKL = Path("/path/to/workspace/roomtour3d_video_pose_20260527/roomtour3d/geometry/geo_trajectory.pkl")

SIMPLE_PATH_LABELS = [
    "moves mostly straight forward",
    "moves forward, turns left, then continues forward",
    "moves forward, turns right, then continues forward",
    "moves forward, turns left, continues forward, then turns left and continues forward",
    "moves forward, turns left, continues forward, then turns right and continues forward",
    "moves forward, turns right, continues forward, then turns left and continues forward",
    "moves forward, turns right, continues forward, then turns right and continues forward",
]

LABEL_BY_TURNS = {
    tuple(): SIMPLE_PATH_LABELS[0],
    ("left",): SIMPLE_PATH_LABELS[1],
    ("right",): SIMPLE_PATH_LABELS[2],
    ("left", "left"): SIMPLE_PATH_LABELS[3],
    ("left", "right"): SIMPLE_PATH_LABELS[4],
    ("right", "left"): SIMPLE_PATH_LABELS[5],
    ("right", "right"): SIMPLE_PATH_LABELS[6],
}

PARAMS = {
    "rdp_tolerance_m": 0.35,
    "min_translation_m": 1.2,
    "straight_max_path_to_straight_ratio": 1.28,
    "simple_path_max_path_to_straight_ratio": 2.65,
    "min_segment_len_m": 0.55,
    "min_turn_deg": 32.0,
    "max_turn_deg": 132.0,
    "max_turn_count": 2,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def path_id_from_qid(qid: str) -> str:
    qid = qid.removeprefix("roomtour3d_")
    for suffix in ("_coarse_video_motion", "_metric_m_pose_interval_summary"):
        if qid.endswith(suffix):
            return qid[: -len(suffix)]
    return qid


def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def path_metrics(points: np.ndarray) -> dict[str, float]:
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    path_len = float(seg_lens.sum())
    straight = float(np.linalg.norm(points[-1] - points[0]))
    return {
        "straight_line_distance_m": straight,
        "cumulative_path_length_m": path_len,
        "path_to_straight_ratio": path_len / max(straight, 1e-6),
    }


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-10:
        return vec_norm(p - a)
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return vec_norm(p - proj)


def rdp(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    distances = [point_line_distance(points[i], a, b) for i in range(1, len(points) - 1)]
    if not distances:
        return points[[0, -1]]
    idx = int(np.argmax(distances)) + 1
    max_dist = distances[idx - 1]
    if max_dist > eps:
        left = rdp(points[: idx + 1], eps)
        right = rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    return points[[0, -1]]


def remove_short_segments(points: np.ndarray, min_len: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    pts = [points[0]]
    for p in points[1:-1]:
        if vec_norm(p - pts[-1]) >= min_len:
            pts.append(p)
    pts.append(points[-1])
    changed = True
    while changed and len(pts) > 3:
        changed = False
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            if vec_norm(pts[i] - new_pts[-1]) < min_len or vec_norm(pts[i + 1] - pts[i]) < min_len:
                changed = True
                continue
            new_pts.append(pts[i])
        new_pts.append(pts[-1])
        pts = new_pts
    return np.asarray(pts, dtype=float)


def signed_turn_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = vec_norm(v1)
    n2 = vec_norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    a = v1 / n1
    b = v2 / n2
    cross = float(a[0] * b[1] - a[1] * b[0])
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.atan2(cross, dot))


def classify(points: np.ndarray) -> tuple[str | None, dict[str, Any]]:
    m = path_metrics(points)
    if m["straight_line_distance_m"] < PARAMS["min_translation_m"]:
        return None, {"drop_reason": "too_little_translation", **m}

    simplified = rdp(points, PARAMS["rdp_tolerance_m"])
    simplified = remove_short_segments(simplified, PARAMS["min_segment_len_m"])
    if len(simplified) < 2:
        return None, {"drop_reason": "too_few_simplified_points", **m}

    if len(simplified) == 2:
        if m["path_to_straight_ratio"] <= PARAMS["straight_max_path_to_straight_ratio"]:
            return LABEL_BY_TURNS[tuple()], {
                **m,
                "turns": [],
                "turn_degrees": [],
                "simplified_point_count": int(len(simplified)),
            }
        return None, {"drop_reason": "straight_candidate_too_curvy", **m, "simplified_point_count": int(len(simplified))}

    segs = np.diff(simplified, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    if np.any(seg_lens < PARAMS["min_segment_len_m"]):
        return None, {"drop_reason": "short_segment", **m, "segment_lengths_m": [float(x) for x in seg_lens]}

    turns: list[str] = []
    turn_degs: list[float] = []
    for v1, v2 in zip(segs, segs[1:]):
        deg = signed_turn_deg(v1, v2)
        abs_deg = abs(deg)
        if abs_deg < PARAMS["min_turn_deg"]:
            continue
        if abs_deg > PARAMS["max_turn_deg"]:
            return None, {"drop_reason": "uturn_or_too_sharp", **m, "turn_degrees": turn_degs + [float(deg)]}
        turns.append("left" if deg > 0 else "right")
        turn_degs.append(float(deg))

    if not turns:
        if m["path_to_straight_ratio"] <= PARAMS["straight_max_path_to_straight_ratio"]:
            return LABEL_BY_TURNS[tuple()], {
                **m,
                "turns": [],
                "turn_degrees": [],
                "simplified_point_count": int(len(simplified)),
                "segment_lengths_m": [float(x) for x in seg_lens],
            }
        return None, {"drop_reason": "no_clear_turn_but_curvy", **m, "simplified_point_count": int(len(simplified))}

    if len(turns) > PARAMS["max_turn_count"]:
        return None, {"drop_reason": "more_than_two_turns", **m, "turns": turns, "turn_degrees": turn_degs}
    if m["path_to_straight_ratio"] > PARAMS["simple_path_max_path_to_straight_ratio"]:
        return None, {"drop_reason": "too_loopy_for_simple_path", **m, "turns": turns, "turn_degrees": turn_degs}

    label = LABEL_BY_TURNS.get(tuple(turns))
    if label is None:
        return None, {"drop_reason": "unsupported_turn_pattern", **m, "turns": turns, "turn_degrees": turn_degs}
    return label, {
        **m,
        "turns": turns,
        "turn_degrees": turn_degs,
        "simplified_point_count": int(len(simplified)),
        "segment_lengths_m": [float(x) for x in seg_lens],
    }


def options_for(label: str, qid: str) -> tuple[list[dict[str, str]], str]:
    rng = random.Random("simplepath-v1|" + qid)
    correct_idx = SIMPLE_PATH_LABELS.index(label)
    candidates = [label]
    scored = []
    for i, cand in enumerate(SIMPLE_PATH_LABELS):
        if cand == label:
            continue
        # Prefer nearby confusions: same number of turns or left/right swapped.
        dist = abs(i - correct_idx)
        if (i == 1 and correct_idx == 2) or (i == 2 and correct_idx == 1):
            dist = 1
        scored.append((dist, rng.random(), cand))
    for _, _, cand in sorted(scored):
        candidates.append(cand)
        if len(candidates) == 4:
            break
    rng.shuffle(candidates)
    letters = ["A", "B", "C", "D"]
    answer = letters[candidates.index(label)]
    return [{"key": k, "text": text} for k, text in zip(letters, candidates)], answer


def points_for_row(row: dict[str, Any], geo: dict[str, dict[int, dict[str, Any]]]) -> tuple[np.ndarray | None, str]:
    md = row.get("metadata") or {}
    pm = md.get("pose_metrics") or {}
    model_id = pm.get("geometry_model_id")
    frames = pm.get("pose_frame_indices_used_for_gt") or md.get("pose_frame_indices_used") or (row.get("input") or {}).get("sampled_frame_indices_32")
    if not model_id or model_id not in geo:
        return None, "missing_geometry_model"
    if not frames:
        return None, "missing_pose_frames"
    model = geo[model_id]
    pts = []
    for frame in frames:
        item = model.get(int(frame))
        if item is None:
            continue
        pos = np.asarray(item["real_world_position"], dtype=float)
        pts.append([float(pos[0]), float(pos[2])])
    if len(pts) < 3:
        return None, "not_enough_geometry_points"
    return np.asarray(pts, dtype=float), ""


def update_coarse_row(row: dict[str, Any], label: str, detail: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    old_answer = row.get("answer")
    old_answer_text = row.get("answer_text")
    old_options = row.get("options")
    options, answer = options_for(label, row.get("qa_id", ""))
    row["question"] = (
        "Which option best describes the simple ground-plane path shape of the camera in this first-person video? "
        "Reply with only one letter."
    )
    row["options"] = options
    row["answer"] = answer
    row["correct_option"] = answer
    row["answer_text"] = label
    row["task_type"] = "roomtour3d_simple_path_shape"
    row["qa_id"] = row["qa_id"].replace("_coarse_video_motion", "_simple_path_shape")
    md = dict(row.get("metadata") or {})
    md["simple_path_revision_20260603"] = {
        "old_task_type": "roomtour3d_coarse_video_motion",
        "old_answer": old_answer,
        "old_answer_text": old_answer_text,
        "old_options": old_options,
        "allowed_label_policy": "straight; straight then left/right and continue; straight then two left/right turns and continue",
        "classifier_params": PARAMS,
        "classifier_detail": detail,
    }
    row["metadata"] = md
    gt = dict(row.get("gt") or {})
    gt["simple_path_label"] = label
    gt["simple_path_classifier_detail"] = detail
    gt["old_motion_label"] = old_answer_text
    row["gt"] = gt
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SRC)
    print("loading geometry", GEO_PKL)
    with GEO_PKL.open("rb") as f:
        geo = pickle.load(f)

    out_rows: list[dict[str, Any]] = []
    kept_coarse = 0
    dropped_coarse = 0
    label_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    old_to_new: dict[str, Counter[str]] = defaultdict(Counter)
    examples_dropped: list[dict[str, Any]] = []

    for row in rows:
        if row.get("source") == "roomtour3d" and row.get("task_type") == "roomtour3d_coarse_video_motion":
            pts, reason = points_for_row(row, geo)
            if pts is None:
                dropped_coarse += 1
                drop_counts[reason] += 1
                if len(examples_dropped) < 20:
                    examples_dropped.append({"qa_id": row.get("qa_id"), "reason": reason, "old_answer_text": row.get("answer_text")})
                continue
            label, detail = classify(pts)
            if label is None:
                dropped_coarse += 1
                reason = detail.get("drop_reason", "unclassified")
                drop_counts[reason] += 1
                if len(examples_dropped) < 20:
                    examples_dropped.append({"qa_id": row.get("qa_id"), "reason": reason, "old_answer_text": row.get("answer_text"), "detail": detail})
                continue
            out_rows.append(update_coarse_row(row, label, detail))
            kept_coarse += 1
            label_counts[label] += 1
            old_to_new[str(row.get("answer_text"))][label] += 1
        else:
            out_rows.append(row)

    write_jsonl(OUT, out_rows)
    summary = {
        "source": str(SRC),
        "output": str(OUT),
        "total_input_rows": len(rows),
        "total_output_rows": len(out_rows),
        "roomtour_coarse_input_rows": kept_coarse + dropped_coarse,
        "roomtour_simple_path_kept_rows": kept_coarse,
        "roomtour_coarse_dropped_rows": dropped_coarse,
        "simple_path_label_counts": dict(label_counts),
        "drop_reason_counts": dict(drop_counts),
        "old_label_to_new_label_counts": {k: dict(v) for k, v in old_to_new.items()},
        "example_dropped": examples_dropped,
        "classifier_params": PARAMS,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
