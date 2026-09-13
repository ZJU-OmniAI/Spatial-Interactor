#!/usr/bin/env python3
"""
Derive two RoomTour3D temporal camera-motion QA families from the final video QA bundle.

1. turning_trajectory_interval:
   Ask which 8-frame interval contains an actual ground-plane turning trajectory.
   "No obvious turning trajectory" is included for clean straight paths. Turns are
   derived from ground-plane path geometry/classifier details, not camera yaw alone.

2. revisited_location_pair:
   Ask which pair of frames is closest to the same ground-plane camera location.
   This uses official RoomTour3D real_world_position when geometry ids are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INPUT = Path("/path/to/workspace/llamafactory_sft_bundle_20260608/data/llamafactory_videoqa_exact_clips.jsonl")
OUT_DIR = Path("/path/to/workspace/roomtour_temporal_motion_qa_20260616")
GEOMETRY_PKL = Path("/path/to/workspace/roomtour3d_video_pose_20260527/roomtour3d/geometry/geo_trajectory.pkl")

TURN_LABELS = {
    "moves forward, turns left, then continues forward",
    "moves forward, turns right, then continues forward",
    "moves forward, turns left, continues forward, then turns left and continues forward",
    "moves forward, turns right, continues forward, then turns right and continues forward",
    "moves forward, turns left, continues forward, then turns right and continues forward",
    "moves forward, turns right, continues forward, then turns left and continues forward",
}
STRAIGHT_LABEL = "moves mostly straight forward"

TURN_OPTIONS = [
    "the first quarter of the video, around frames 1-8",
    "the second quarter of the video, around frames 9-16",
    "the third quarter of the video, around frames 17-24",
    "the final quarter of the video, around frames 25-32",
    "no obvious turning trajectory; the camera moves mostly straight",
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def stable_shuffle(items: list[Any], seed_text: str) -> list[Any]:
    items = list(items)
    random.Random(stable_seed(seed_text)).shuffle(items)
    return items


def source_video(row: dict[str, Any]) -> str:
    videos = row.get("videos")
    if isinstance(videos, list) and videos:
        return str(videos[0])
    if isinstance(videos, str):
        return videos
    return ""


def get_gt(row: dict[str, Any]) -> dict[str, Any]:
    return ((row.get("metadata") or {}).get("gt") or {}) if isinstance(row.get("metadata"), dict) else {}


def get_detail(row: dict[str, Any]) -> dict[str, Any]:
    gt = get_gt(row)
    detail = gt.get("simple_path_classifier_detail") or gt.get("simple_path_classification") or {}
    return detail if isinstance(detail, dict) else {}


def get_exact_pose(row: dict[str, Any]) -> dict[str, Any]:
    gt = get_gt(row)
    exact = gt.get("exact_pose_values_meters") or gt.get("exact_pose_values") or {}
    return exact if isinstance(exact, dict) else {}


def metric_value(row: dict[str, Any], key: str) -> float | None:
    detail = get_detail(row)
    exact = get_exact_pose(row)
    for src in (detail, exact):
        if key in src and src[key] is not None:
            try:
                return float(src[key])
            except Exception:
                return None
    return None


def make_turn_options(answer_idx: int, row_id: str) -> tuple[list[dict[str, str]], str]:
    indexed = stable_shuffle(list(enumerate(TURN_OPTIONS)), row_id)
    options = []
    answer = ""
    for label, (idx, text) in zip("ABCDE", indexed):
        options.append({"label": label, "text": text})
        if idx == answer_idx:
            answer = label
    return options, answer


def turn_question(options: list[dict[str, str]]) -> str:
    lines = "\n".join(f"{o['label']}. {o['text']}" for o in options)
    return (
        "<video>\n"
        "Which interval contains an actual turning trajectory of the camera? "
        "A turning trajectory means the camera moves along the ground while changing direction; "
        "a pure in-place view rotation does not count. Reply with only one letter.\n"
        "Options:\n"
        f"{lines}"
    )


def turn_candidate(line_no: int, row: dict[str, Any], margin: float, min_seg_m: float) -> tuple[dict[str, Any] | None, str]:
    md = row.get("metadata") or {}
    if md.get("source") != "roomtour3d" or md.get("task_type") != "roomtour3d_simple_path_shape":
        return None, "not_roomtour_simple"

    row_id = str(row.get("id") or f"line_{line_no}")
    answer_text = str(md.get("answer_text") or "")
    detail = get_detail(row)
    exact = get_exact_pose(row)

    if answer_text == STRAIGHT_LABEL:
        ratio = metric_value(row, "path_to_straight_ratio")
        straight = metric_value(row, "straight_line_distance_m")
        if ratio is None or ratio > 1.12:
            return None, "straight_not_clean_enough"
        if straight is not None and straight < 2.0:
            return None, "straight_too_short"
        answer_idx = 4
        options, answer = make_turn_options(answer_idx, row_id)
        score = (1.12 - ratio) + (straight or 0.0) * 0.01
        out_id = row_id.replace("_simple_path_shape", "_turning_trajectory_interval")
        return {
            "id": out_id,
            "conversations": [{"from": "human", "value": turn_question(options)}, {"from": "gpt", "value": answer}],
            "videos": [source_video(row)],
            "metadata": {
                "source": "roomtour3d",
                "dataset": md.get("dataset"),
                "task_type": "roomtour3d_turning_trajectory_interval",
                "answer_text": TURN_OPTIONS[answer_idx],
                "source_simple_path_id": row_id,
                "source_simple_path_answer_text": answer_text,
                "gt": {
                    "has_turning_trajectory": False,
                    "turn_interval_index": answer_idx,
                    "turn_interval_text": TURN_OPTIONS[answer_idx],
                    "path_to_straight_ratio": ratio,
                    "straight_line_distance_m": straight,
                    "exact_pose_values": exact,
                    "simple_path_detail": detail,
                },
                "selection_score": score,
            },
        }, "ok_none"

    if answer_text not in TURN_LABELS:
        return None, "label_not_supported"

    turns = detail.get("turns") or []
    segs = detail.get("segment_lengths_m") or []
    if not isinstance(turns, list) or not isinstance(segs, list) or not turns or len(segs) < len(turns) + 1:
        return None, "missing_turn_detail"
    try:
        segs_f = [float(x) for x in segs]
    except Exception:
        return None, "bad_segment_lengths"
    total = sum(segs_f)
    if total <= 0:
        return None, "bad_total_length"

    quarters = []
    margins = []
    for i, _turn in enumerate(turns):
        if segs_f[i] < min_seg_m or segs_f[i + 1] < min_seg_m:
            return None, "turn_not_bracketed_by_translation"
        frac = sum(segs_f[: i + 1]) / total
        q = min(3, max(0, int(frac * 4)))
        m = min(abs(frac - b) for b in (0.25, 0.50, 0.75))
        if m < margin:
            return None, "near_interval_boundary"
        quarters.append(q)
        margins.append(m)

    if len(set(quarters)) != 1:
        return None, "turns_in_multiple_intervals"

    answer_idx = quarters[0]
    options, answer = make_turn_options(answer_idx, row_id)
    out_id = row_id.replace("_simple_path_shape", "_turning_trajectory_interval")
    turn_degrees = detail.get("turn_degrees") or []
    score = min(margins) + min(total, 12.0) * 0.005
    return {
        "id": out_id,
        "conversations": [{"from": "human", "value": turn_question(options)}, {"from": "gpt", "value": answer}],
        "videos": [source_video(row)],
        "metadata": {
            "source": "roomtour3d",
            "dataset": md.get("dataset"),
            "task_type": "roomtour3d_turning_trajectory_interval",
            "answer_text": TURN_OPTIONS[answer_idx],
            "source_simple_path_id": row_id,
            "source_simple_path_answer_text": answer_text,
            "gt": {
                "has_turning_trajectory": True,
                "turn_interval_index": answer_idx,
                "turn_interval_text": TURN_OPTIONS[answer_idx],
                "turns": turns,
                "turn_degrees": turn_degrees,
                "turn_interval_margins": margins,
                "segment_lengths_m": segs_f,
                "cumulative_path_length_m": total,
                "exact_pose_values": exact,
                "simple_path_detail": detail,
            },
            "selection_score": score,
        },
    }, "ok_turn"


def select_turn_rows(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_idx[int(row["metadata"]["gt"]["turn_interval_index"])].append(row)
    for rows in by_idx.values():
        rows.sort(key=lambda r: (float(r["metadata"].get("selection_score") or 0.0), r["id"]), reverse=True)

    # Keep most real-turn rows, then fill with clean straight/no-turn rows.
    selected: list[dict[str, Any]] = []
    interval_caps = {0: 170, 1: 330, 2: 300, 3: 80}
    for idx in range(4):
        selected.extend(by_idx.get(idx, [])[: interval_caps[idx]])
    if len(selected) > 820:
        selected = sorted(selected, key=lambda r: (float(r["metadata"].get("selection_score") or 0.0), r["id"]), reverse=True)[:820]

    remaining = target - len(selected)
    if remaining > 0:
        selected.extend(by_idx.get(4, [])[:remaining])
    if len(selected) < target:
        used = {r["id"] for r in selected}
        rest = [r for r in candidates if r["id"] not in used]
        rest.sort(key=lambda r: (float(r["metadata"].get("selection_score") or 0.0), r["id"]), reverse=True)
        selected.extend(rest[: target - len(selected)])
    return selected[:target]


def load_geometry(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def points_for_row(row: dict[str, Any], geo: dict[str, Any]) -> tuple[list[tuple[float, float]] | None, str]:
    exact = get_exact_pose(row)
    model_id = exact.get("geometry_model_id")
    frames = exact.get("pose_frame_indices_used_for_gt")
    if not frames:
        frames = ((row.get("metadata") or {}).get("original_input") or {}).get("sampled_frame_indices_32")
    if not frames:
        frames = (row.get("metadata") or {}).get("original_input", {}).get("sampled_frame_indices_32")
    if not frames:
        frames = (row.get("metadata") or {}).get("clip_dedup_key", {}).get("slots")
    if not model_id or model_id not in geo:
        return None, "missing_geometry_model"
    model = geo[model_id]
    pts: list[tuple[float, float]] = []
    for frame in frames:
        try:
            item = model.get(int(frame))
        except Exception:
            item = None
        if item is None:
            return None, "missing_frame_pose"
        pos = item.get("real_world_position")
        if pos is None:
            return None, "missing_position"
        pts.append((float(pos[0]), float(pos[2])))
    if len(pts) != 32:
        return None, "not_32_points"
    return pts, "ok"


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_len_between(points: list[tuple[float, float]], i: int, j: int) -> float:
    return sum(dist(points[k], points[k + 1]) for k in range(i, j))


def make_pair_options(true_pair: tuple[int, int], false_pairs: list[tuple[int, int]], row_id: str) -> tuple[list[dict[str, str]], str]:
    pairs = [(true_pair, True)] + [(p, False) for p in false_pairs[:3]]
    pairs = stable_shuffle(pairs, row_id + "::pair_options")
    opts = []
    ans = ""
    for label, (pair, is_true) in zip("ABCD", pairs):
        opts.append({"label": label, "text": f"frame {pair[0] + 1} and frame {pair[1] + 1}"})
        if is_true:
            ans = label
    return opts, ans


def pair_question(options: list[dict[str, str]]) -> str:
    lines = "\n".join(f"{o['label']}. {o['text']}" for o in options)
    return (
        "<video>\n"
        "Which pair of frames shows the camera at nearly the same ground-plane location? "
        "Reply with only one letter.\n"
        "Options:\n"
        f"{lines}"
    )


def revisit_candidate(row: dict[str, Any], geo: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    md = row.get("metadata") or {}
    if md.get("source") != "roomtour3d":
        return None, "not_roomtour"
    row_id = str(row.get("id") or "")
    video = source_video(row)
    points, reason = points_for_row(row, geo)
    if points is None:
        return None, reason

    true_pairs = []
    false_pairs = []
    for i in range(32):
        for j in range(i + 6, 32):
            d = dist(points[i], points[j])
            between = path_len_between(points, i, j)
            if d <= 0.35 and between >= 2.0:
                true_pairs.append((d, between, i, j))
            elif d >= 1.50:
                false_pairs.append((d, i, j))

    if not true_pairs:
        return None, "no_revisit_pair"
    true_pairs.sort(key=lambda x: (x[0], -x[1]))
    false_pairs = stable_shuffle(false_pairs, row_id + "::false_pairs")
    chosen_false: list[tuple[int, int]] = []
    chosen_false_distances: list[float] = []
    used_frames: set[int] = set()
    for d, i, j in false_pairs:
        if (i, j) == (true_pairs[0][2], true_pairs[0][3]):
            continue
        # Keep distractors away from the true pair and from each other.
        if d < 1.50:
            continue
        if len({i, j} & used_frames) > 1:
            continue
        chosen_false.append((i, j))
        chosen_false_distances.append(d)
        used_frames.update([i, j])
        if len(chosen_false) == 3:
            break
    if len(chosen_false) < 3:
        return None, "not_enough_false_pairs"

    true_distance, between, i, j = true_pairs[0]
    opts, ans = make_pair_options((i, j), chosen_false, row_id)
    out_id = row_id.replace("_simple_path_shape", "_revisited_location_pair").replace(
        "_metric_m_pose_interval_summary", "_revisited_location_pair"
    )
    return {
        "id": out_id,
        "conversations": [{"from": "human", "value": pair_question(opts)}, {"from": "gpt", "value": ans}],
        "videos": [video],
        "metadata": {
            "source": "roomtour3d",
            "dataset": md.get("dataset"),
            "task_type": "roomtour3d_revisited_location_pair",
            "answer_text": f"frame {i + 1} and frame {j + 1}",
            "source_videoqa_id": row_id,
            "gt": {
                "frame_pair_1based": [i + 1, j + 1],
                "pair_distance_m": true_distance,
                "path_length_between_pair_m": between,
                "false_pair_min_distance_m": 1.50,
                "chosen_false_pair_distances_m": chosen_false_distances,
                "temporal_gap_min_frames": 6,
                "geometry_model_id": get_exact_pose(row).get("geometry_model_id"),
            },
            "selection_score": (0.35 - true_distance) * 100.0 + min(between, 10.0),
        },
    }, "ok"


def select_revisit_rows(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_answer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_answer[row["conversations"][1]["value"]].append(row)
    for label_rows in by_answer.values():
        label_rows.sort(
            key=lambda r: (
                float((r["metadata"]["gt"] or {}).get("pair_distance_m") or 999.0) <= 0.25,
                float(r["metadata"].get("selection_score") or 0.0),
                r["id"],
            ),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    per_label = target // 4
    for label in "ABCD":
        selected.extend(by_answer.get(label, [])[:per_label])

    if len(selected) < target:
        used = {r["id"] for r in selected}
        rest = [r for r in rows if r["id"] not in used]
        rest.sort(key=lambda r: (float(r["metadata"].get("selection_score") or 0.0), r["id"]), reverse=True)
        selected.extend(rest[: target - len(selected)])
    return selected[:target]


def build_html(rows: list[dict[str, Any]], path: Path, media_root: Path, title: str, limit: int = 60) -> None:
    cards = []
    for row in rows[:limit]:
        q = row["conversations"][0]["value"]
        a = row["conversations"][1]["value"]
        video = source_video(row)
        video_src = os.path.relpath(media_root / video, path.parent) if video else ""
        md = row.get("metadata") or {}
        cards.append(
            f"""
<section class="card">
  <h2>{row.get('id')}</h2>
  <video src="{video_src}" controls muted preload="metadata"></video>
  <p class="q">{q.splitlines()[1]}</p>
  <pre>{q.split('Options:', 1)[-1].strip()}</pre>
  <p><b>Answer:</b> {a}. {md.get('answer_text')}</p>
  <pre class="gt">{json.dumps(md.get('gt') or {}, ensure_ascii=False, indent=2)[:900]}</pre>
</section>
"""
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1d2430; }}
    h1 {{ margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #dde1e7; border-radius: 8px; padding: 14px; }}
    h2 {{ font-size: 14px; margin: 0 0 10px; word-break: break-word; }}
    video {{ width: 100%; aspect-ratio: 16 / 9; background: #111; border-radius: 4px; }}
    .q {{ color: #c62828; line-height: 1.35; }}
    pre {{ white-space: pre-wrap; background: #f2f4f7; padding: 10px; border-radius: 6px; }}
    .gt {{ color: #465160; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--geometry", type=Path, default=GEOMETRY_PKL)
    ap.add_argument("--turn-target", type=int, default=1000)
    ap.add_argument("--revisit-target", type=int, default=1000)
    ap.add_argument("--turn-margin", type=float, default=0.04)
    ap.add_argument("--turn-min-segment-m", type=float, default=0.60)
    args = ap.parse_args()

    turn_candidates: list[dict[str, Any]] = []
    turn_reasons = Counter()
    rows_for_revisit: list[dict[str, Any]] = []
    seen_video = set()

    for line_no, row in read_jsonl(args.input):
        md = row.get("metadata") or {}
        if md.get("source") == "roomtour3d":
            v = source_video(row)
            if v and v not in seen_video:
                rows_for_revisit.append(row)
                seen_video.add(v)
        cand, reason = turn_candidate(line_no, row, args.turn_margin, args.turn_min_segment_m)
        turn_reasons[reason] += 1
        if cand:
            turn_candidates.append(cand)

    turn_rows = select_turn_rows(turn_candidates, args.turn_target)
    turn_out = args.out_dir / "roomtour_turning_trajectory_interval_mcq.jsonl"
    write_jsonl(turn_out, turn_rows)
    build_html(turn_rows, args.out_dir / "review_turning_trajectory_interval.html", args.input.parent, "RoomTour Turning-Trajectory Interval QA")

    geo = load_geometry(args.geometry)
    revisit_rows: list[dict[str, Any]] = []
    revisit_reasons = Counter()
    for row in rows_for_revisit:
        cand, reason = revisit_candidate(row, geo)
        revisit_reasons[reason] += 1
        if cand:
            revisit_rows.append(cand)
    revisit_rows = select_revisit_rows(revisit_rows, args.revisit_target)
    revisit_out = args.out_dir / "roomtour_revisited_location_pair_mcq.jsonl"
    write_jsonl(revisit_out, revisit_rows)
    build_html(revisit_rows, args.out_dir / "review_revisited_location_pair.html", args.input.parent, "RoomTour Revisited-Location Pair QA")

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(args.input),
        "geometry": str(args.geometry),
        "turning_trajectory_interval": {
            "output": str(turn_out),
            "html": str(args.out_dir / "review_turning_trajectory_interval.html"),
            "eligible_candidates": len(turn_candidates),
            "selected_rows": len(turn_rows),
            "skip_reasons": dict(turn_reasons),
            "answer_text_distribution": dict(Counter(r["metadata"]["answer_text"] for r in turn_rows)),
            "answer_letter_distribution": dict(Counter(r["conversations"][1]["value"] for r in turn_rows)),
            "has_turn_distribution": dict(Counter(str(r["metadata"]["gt"]["has_turning_trajectory"]) for r in turn_rows)),
        },
        "revisited_location_pair": {
            "output": str(revisit_out),
            "html": str(args.out_dir / "review_revisited_location_pair.html"),
            "selected_rows": len(revisit_rows),
            "skip_reasons": dict(revisit_reasons),
            "answer_letter_distribution": dict(Counter(r["conversations"][1]["value"] for r in revisit_rows)),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
