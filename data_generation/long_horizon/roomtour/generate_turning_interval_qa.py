#!/usr/bin/env python3
"""
Generate RoomTour3D one-turn temporal-localization QA.

The source is the final video QA bundle. We only derive from RoomTour simple-path
samples whose current final label is a single turn and whose stored one-turn GT
agrees with that direction. The turn location is approximated by the path-length
fraction before the turn, then mapped to video quarters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/path/to/workspace/llamafactory_sft_bundle_20260608/data/llamafactory_videoqa_exact_clips.jsonl")
DEFAULT_OUT_DIR = Path("/path/to/workspace/roomtour_one_turn_interval_qa_20260616")

LEFT_LABEL = "moves forward, turns left, then continues forward"
RIGHT_LABEL = "moves forward, turns right, then continues forward"
LABEL_TO_TURN = {LEFT_LABEL: "left", RIGHT_LABEL: "right"}

INTERVALS = [
    "the first quarter of the video, around frames 1-8",
    "the second quarter of the video, around frames 9-16",
    "the third quarter of the video, around frames 17-24",
    "the final quarter of the video, around frames 25-32",
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


def get_answer_text(row: dict[str, Any]) -> str:
    md = row.get("metadata") or {}
    return str(md.get("answer_text") or "")


def get_video(row: dict[str, Any]) -> list[str]:
    videos = row.get("videos")
    if isinstance(videos, list):
        return list(videos)
    if isinstance(videos, str):
        return [videos]
    return []


def get_detail(row: dict[str, Any]) -> dict[str, Any]:
    md = row.get("metadata") or {}
    gt = md.get("gt") or {}
    detail = gt.get("simple_path_classifier_detail") or {}
    return detail if isinstance(detail, dict) else {}


def turn_fraction(detail: dict[str, Any]) -> float | None:
    seg = detail.get("segment_lengths_m") or []
    if not isinstance(seg, list) or len(seg) < 2:
        return None
    try:
        seg = [float(x) for x in seg if float(x) >= 0]
    except Exception:
        return None
    total = sum(seg)
    if total <= 0:
        return None
    return seg[0] / total


def quarter_for_fraction(frac: float) -> int:
    return min(3, max(0, int(frac * 4)))


def boundary_margin(frac: float) -> float:
    return min(abs(frac - b) for b in (0.25, 0.50, 0.75))


def make_options(correct_idx: int, row_id: str) -> tuple[list[dict[str, str]], str]:
    labels = list("ABCD")
    items = list(enumerate(INTERVALS))
    rng = random.Random(stable_seed(row_id))
    rng.shuffle(items)
    options = []
    answer = ""
    for label, (idx, text) in zip(labels, items):
        options.append({"label": label, "text": text})
        if idx == correct_idx:
            answer = label
    return options, answer


def format_question(options: list[dict[str, str]]) -> str:
    option_lines = "\n".join(f"{o['label']}. {o['text']}" for o in options)
    return (
        "<video>\n"
        "The camera follows a path with one main turn. In which part of the video does the main turn happen? "
        "Reply with only one letter.\n"
        "Options:\n"
        f"{option_lines}"
    )


def convert_row(
    line_no: int, row: dict[str, Any], margin_threshold: float, source_input: Path
) -> tuple[dict[str, Any] | None, str]:
    md = row.get("metadata") or {}
    if md.get("source") != "roomtour3d" or md.get("task_type") != "roomtour3d_simple_path_shape":
        return None, "not_roomtour_simple_path"

    answer_text = get_answer_text(row)
    expected_turn = LABEL_TO_TURN.get(answer_text)
    if not expected_turn:
        return None, "current_label_not_one_turn"

    detail = get_detail(row)
    turns = detail.get("turns") or []
    if not isinstance(turns, list) or len(turns) != 1:
        return None, "gt_not_single_turn"
    gt_turn = str(turns[0]).lower()
    if gt_turn != expected_turn:
        return None, "turn_direction_mismatch"

    frac = turn_fraction(detail)
    if frac is None:
        return None, "missing_segment_lengths"
    margin = boundary_margin(frac)
    if margin < margin_threshold:
        return None, "near_quarter_boundary"

    qidx = quarter_for_fraction(frac)
    row_id = str(row.get("id") or f"line_{line_no}")
    options, answer = make_options(qidx, row_id)
    question = format_question(options)

    out_id = row_id.replace("_simple_path_shape", "_main_turn_interval")
    out = {
        "id": out_id,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
        "videos": get_video(row),
        "metadata": {
            "source_file": str(source_input),
            "source": "roomtour3d",
            "dataset": md.get("dataset"),
            "scene": md.get("scene"),
            "task_type": "roomtour3d_main_turn_interval",
            "answer_text": INTERVALS[qidx],
            "source_simple_path_id": row_id,
            "source_simple_path_answer_text": answer_text,
            "gt": {
                "turn_direction": gt_turn,
                "turn_degrees": (detail.get("turn_degrees") or [None])[0],
                "turn_path_fraction": frac,
                "turn_interval_index": qidx,
                "turn_interval_text": INTERVALS[qidx],
                "boundary_margin": margin,
                "simple_path_classifier_detail": detail,
            },
            "generation": {
                "script": Path(__file__).name,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "margin_threshold": margin_threshold,
                "rule": "single-turn RoomTour samples; turn interval from pre-turn path-length fraction",
            },
        },
    }
    return out, "ok"


def build_html(rows: list[dict[str, Any]], out_path: Path, title: str, media_root: Path, limit: int = 40) -> None:
    cards = []
    for row in rows[:limit]:
        conv = row["conversations"]
        md = row["metadata"]
        video = row.get("videos", [""])[0]
        video_src = os.path.relpath(media_root / video, out_path.parent)
        answer = conv[-1]["value"]
        options = "\n".join(conv[0]["value"].split("Options:\n", 1)[-1].splitlines())
        gt = md.get("gt") or {}
        cards.append(
            f"""
<section class="card">
  <h2>{row['id']}</h2>
  <video src="{video_src}" controls muted preload="metadata"></video>
  <p class="q">The camera follows a path with one main turn. In which part of the video does the main turn happen?</p>
  <pre>{options}</pre>
  <p><b>Answer:</b> {answer}. {md.get('answer_text')}</p>
  <p class="meta">turn={gt.get('turn_direction')}, fraction={gt.get('turn_path_fraction'):.3f}, margin={gt.get('boundary_margin'):.3f}</p>
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
    h2 {{ font-size: 15px; margin: 0 0 10px; word-break: break-word; }}
    video {{ width: 100%; aspect-ratio: 16 / 9; background: #111; border-radius: 4px; }}
    .q {{ color: #c62828; font-size: 15px; line-height: 1.35; }}
    pre {{ white-space: pre-wrap; background: #f2f4f7; padding: 10px; border-radius: 6px; }}
    .meta {{ color: #5d6675; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    reasons = Counter()
    answer_labels = Counter()
    intervals = Counter()
    turns = Counter()

    for line_no, row in read_jsonl(args.input):
        out, reason = convert_row(line_no, row, args.margin, args.input)
        reasons[reason] += 1
        if out:
            rows.append(out)
            answer_labels[out["conversations"][-1]["value"]] += 1
            intervals[out["metadata"]["answer_text"]] += 1
            turns[out["metadata"]["gt"]["turn_direction"]] += 1

    out_jsonl = args.out_dir / "roomtour_one_turn_main_turn_interval_mcq.jsonl"
    report_path = args.out_dir / "generation_summary.json"
    html_path = args.out_dir / "review_examples.html"

    write_jsonl(out_jsonl, rows)
    build_html(rows, html_path, "RoomTour One-Turn Main-Turn Interval QA", args.input.parent)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(args.input),
        "output": str(out_jsonl),
        "html": str(html_path),
        "margin_threshold": args.margin,
        "rows": len(rows),
        "skip_reasons": dict(reasons),
        "answer_label_distribution": dict(answer_labels),
        "interval_distribution": dict(intervals),
        "turn_direction_distribution": dict(turns),
        "sample_ids": [r["id"] for r in rows[:20]],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
