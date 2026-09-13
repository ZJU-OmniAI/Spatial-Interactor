#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SRC = Path("/path/to/workspace/roomtour3d_video_pose_20260527/generated_video_pose_qa_20260527")
OUT = Path("/path/to/workspace/roomtour3d_video_pose_20260527/generated_video_pose_qa_curated_10k_20260528")
TARGET_WINDOWS = 5000
SEED = 20260528


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


def parse_pose_bins(answer_text: str) -> tuple[str, str, str]:
    straight = re.search(r"straight-line displacement ([^,]+)", answer_text)
    angle = re.search(r"axis angle ([^,]+)", answer_text)
    path = re.search(r"total path length (.+)$", answer_text)
    return (
        straight.group(1) if straight else "unknown",
        angle.group(1) if angle else "unknown",
        path.group(1) if path else "unknown",
    )


def balanced_targets(counts: Counter[str], total: int) -> dict[str, int]:
    remaining_labels = set(counts)
    targets = {k: 0 for k in counts}
    remaining_total = total
    while remaining_labels and remaining_total > 0:
        quota = remaining_total / len(remaining_labels)
        fixed = []
        for label in list(remaining_labels):
            if counts[label] <= quota:
                targets[label] = counts[label]
                remaining_total -= counts[label]
                fixed.append(label)
        if not fixed:
            base = remaining_total // len(remaining_labels)
            extra = remaining_total % len(remaining_labels)
            for i, label in enumerate(sorted(remaining_labels)):
                targets[label] = base + (1 if i < extra else 0)
            remaining_total = 0
            break
        for label in fixed:
            remaining_labels.remove(label)
    return targets


def is_reasonable_window(w: dict[str, Any]) -> bool:
    m = w["metrics"]
    return (
        len(w["sampled_frame_indices_32"]) >= 24
        and len(w["pose_frame_indices_used"]) >= 24
        and 0.2 <= float(m["straight_colmap_units"]) <= 50.0
        and float(m["path_colmap_units"]) <= 100.0
        and float(m["path_to_straight_ratio"]) <= 20.0
    )


def enrich_windows(windows: list[dict[str, Any]], qa_by_path: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    enriched = []
    for w in windows:
        if not is_reasonable_window(w):
            continue
        pair = qa_by_path.get(w["path_id"], {})
        pose = pair.get("roomtour3d_pose_interval_summary")
        motion = pair.get("roomtour3d_coarse_video_motion")
        if not pose or not motion:
            continue
        straight_bin, angle_bin, path_bin = parse_pose_bins(pose["answer_text"])
        item = dict(w)
        item["_motion_label"] = motion["answer_text"]
        item["_straight_bin"] = straight_bin
        item["_angle_bin"] = angle_bin
        item["_path_bin"] = path_bin
        enriched.append(item)
    return enriched


def choose_windows(candidates: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    motion_counts = Counter(x["_motion_label"] for x in candidates)
    motion_targets = balanced_targets(motion_counts, TARGET_WINDOWS)
    straight_targets = balanced_targets(Counter(x["_straight_bin"] for x in candidates), TARGET_WINDOWS)
    angle_targets = balanced_targets(Counter(x["_angle_bin"] for x in candidates), TARGET_WINDOWS)
    path_targets = balanced_targets(Counter(x["_path_bin"] for x in candidates), TARGET_WINDOWS)

    selected: list[dict[str, Any]] = []
    selected_keys = set()
    video_counts = Counter()
    straight_counts = Counter()
    angle_counts = Counter()
    path_counts = Counter()
    motion_selected = Counter()

    def component_score(item: dict[str, Any]) -> float:
        score = 0.0
        for key, counts, targets, weight in [
            ("_motion_label", motion_selected, motion_targets, 1.35),
            ("_straight_bin", straight_counts, straight_targets, 1.20),
            ("_path_bin", path_counts, path_targets, 1.20),
            ("_angle_bin", angle_counts, angle_targets, 0.75),
        ]:
            target = max(targets.get(item[key], 1), 1)
            score += weight * max(0.0, (target - counts[item[key]]) / target)
        score += 0.10 * (len(item["sampled_frame_indices_32"]) / 32.0)
        score += 0.10 * (len(item["pose_frame_indices_used"]) / 32.0)
        score -= 0.35 * video_counts[item["video_id"]]
        return score

    def add(item: dict[str, Any]) -> None:
        if item["path_id"] in selected_keys:
            return
        selected.append(item)
        selected_keys.add(item["path_id"])
        video_counts[item["video_id"]] += 1
        straight_counts[item["_straight_bin"]] += 1
        angle_counts[item["_angle_bin"]] += 1
        path_counts[item["_path_bin"]] += 1
        motion_selected[item["_motion_label"]] += 1

    rare_motion = {k for k, v in motion_counts.items() if v <= motion_targets.get(k, 0)}
    rare_straight = {k for k, v in Counter(x["_straight_bin"] for x in candidates).items() if v <= straight_targets.get(k, 0)}
    rare_path = {k for k, v in Counter(x["_path_bin"] for x in candidates).items() if v <= path_targets.get(k, 0)}

    def rare_score(item: dict[str, Any]) -> float:
        return (
            (2.0 if item["_motion_label"] in rare_motion else 0.0)
            + (1.3 if item["_straight_bin"] in rare_straight else 0.0)
            + (1.3 if item["_path_bin"] in rare_path else 0.0)
            + component_score(item)
            + rng.random() * 0.01
        )

    rare_first = [
        x
        for x in candidates
        if x["_motion_label"] in rare_motion or x["_straight_bin"] in rare_straight or x["_path_bin"] in rare_path
    ]
    for item in sorted(rare_first, key=rare_score, reverse=True):
        if len(selected) >= TARGET_WINDOWS:
            break
        add(item)

    if len(selected) < TARGET_WINDOWS:
        remaining = [x for x in candidates if x["path_id"] not in selected_keys]
        rng.shuffle(remaining)
        for cap in [3, 4, 6, 8, 12, 9999]:
            while len(selected) < TARGET_WINDOWS:
                eligible = [
                    item
                    for item in remaining
                    if item["path_id"] not in selected_keys and video_counts[item["video_id"]] < cap
                ]
                if not eligible:
                    break
                eligible.sort(key=lambda item: component_score(item) + rng.random() * 0.01, reverse=True)
                for item in eligible[: min(300, TARGET_WINDOWS - len(selected))]:
                    add(item)
            if len(selected) >= TARGET_WINDOWS:
                break

    return selected[:TARGET_WINDOWS]


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def write_preview(qas: list[dict[str, Any]], windows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    by_path = defaultdict(list)
    for q in qas:
        by_path[q["path_id"]].append(q)
    cards = []
    for w in windows[:120]:
        rows = by_path[w["path_id"]]
        qhtml = "".join(
            "<div class='qa'><b>{}</b><p>{}</p><ol>{}</ol><p>answer: {}</p></div>".format(
                html.escape(q["task_type"]),
                html.escape(q["question"]),
                "".join(f"<li>{html.escape(o['key'])}. {html.escape(o['text'])}</li>" for o in q["options"]),
                html.escape(q["answer"] + " - " + q["answer_text"]),
            )
            for q in rows
        )
        m = w["metrics"]
        cards.append(
            f"<section><h2>{html.escape(w['path_id'])}</h2>"
            f"<p><b>{html.escape(w['_motion_label'])}</b> | {html.escape(w['_straight_bin'])} | {html.escape(w['_angle_bin'])} | {html.escape(w['_path_bin'])}</p>"
            f"<p>video={html.escape(w['video_id'])}, sampled={len(w['sampled_frame_indices_32'])}, pose={len(w['pose_frame_indices_used'])}, "
            f"straight={m['straight_colmap_units']:.2f}, path={m['path_colmap_units']:.2f}, ratio={m['path_to_straight_ratio']:.2f}</p>"
            f"<p>{html.escape(w['source_instruction'][:450])}</p>{qhtml}</section>"
        )
    text = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomTour3D Curated 10k QA</title><style>
body{{font-family:Arial,sans-serif;background:#f5f6f7;margin:0;color:#111}}main{{max-width:1200px;margin:0 auto;padding:18px}}
section{{background:#fff;border:1px solid #d8dde3;border-radius:6px;margin:12px 0;padding:12px}}pre{{background:#fff;padding:12px;border:1px solid #ddd;overflow:auto}}
.qa{{border-top:1px solid #eee;margin-top:8px;padding-top:8px}}li{{margin:4px 0}}
</style></head><body><main><h1>RoomTour3D Curated 10k QA</h1><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>{''.join(cards)}</main></body></html>"""
    (OUT / "preview.html").write_text(text, encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    qas = read_jsonl(SRC / "roomtour3d_video_pose_qa.jsonl")
    windows = read_jsonl(SRC / "roomtour3d_video_pose_windows.jsonl")
    qa_by_path: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for q in qas:
        qa_by_path[q["path_id"]][q["task_type"]] = q

    candidates = enrich_windows(windows, qa_by_path)
    selected = choose_windows(candidates, rng)
    selected_ids = {w["path_id"] for w in selected}
    curated_qas = []
    curation = {
        "name": "roomtour3d_video_pose_qa_curated_10k_20260528",
        "policy": (
            "filtered COLMAP outliers, preserved rare motion labels and rare pose interval bins, "
            "reduced per-video redundancy, and balanced remaining motion labels and pose interval bins"
        ),
    }
    for q in qas:
        if q["path_id"] in selected_ids:
            q = dict(q)
            q["curation"] = curation
            q["metadata"] = dict(q.get("metadata", {}))
            q["metadata"]["curation"] = curation
            curated_qas.append(q)
    curated_qas.sort(key=lambda q: (q["path_id"], q["task_type"]))
    selected_public = [strip_private(w) for w in selected]

    qa_out = OUT / "roomtour3d_video_pose_qa_curated_10k.jsonl"
    win_out = OUT / "roomtour3d_video_pose_windows_curated_5k.jsonl"
    write_jsonl(qa_out, curated_qas)
    write_jsonl(win_out, selected_public)
    pd.DataFrame(curated_qas).to_parquet(OUT / "roomtour3d_video_pose_qa_curated_10k.parquet", index=False)
    pd.DataFrame(selected_public).to_parquet(OUT / "roomtour3d_video_pose_windows_curated_5k.parquet", index=False)

    motion_counts = Counter(w["_motion_label"] for w in selected)
    straight_counts = Counter(w["_straight_bin"] for w in selected)
    angle_counts = Counter(w["_angle_bin"] for w in selected)
    path_counts = Counter(w["_path_bin"] for w in selected)
    per_video = Counter(w["video_id"] for w in selected)
    summary = {
        "source_dir": str(SRC),
        "out_dir": str(OUT),
        "candidate_windows_after_filter": len(candidates),
        "kept_windows": len(selected),
        "qa_rows": len(curated_qas),
        "unique_videos": len(per_video),
        "task_counts": dict(Counter(q["task_type"] for q in curated_qas)),
        "answer_counts": dict(Counter(q["answer"] for q in curated_qas)),
        "answer_counts_by_task": {
            task: dict(Counter(q["answer"] for q in curated_qas if q["task_type"] == task))
            for task in sorted({q["task_type"] for q in curated_qas})
        },
        "motion_label_counts": dict(motion_counts),
        "pose_straight_bin_counts": dict(straight_counts),
        "pose_angle_bin_counts": dict(angle_counts),
        "pose_path_bin_counts": dict(path_counts),
        "per_video_path_count": {
            "max": max(per_video.values()),
            "min": min(per_video.values()),
            "mean": sum(per_video.values()) / len(per_video),
            "videos_with_1": sum(1 for v in per_video.values() if v == 1),
            "videos_with_2_to_4": sum(1 for v in per_video.values() if 2 <= v <= 4),
            "videos_with_5_plus": sum(1 for v in per_video.values() if v >= 5),
        },
        "filter_policy": {
            "sampled_frame_count_min": 24,
            "pose_frame_count_min": 24,
            "straight_colmap_units": "[0.2, 50]",
            "path_colmap_units_max": 100,
            "path_to_straight_ratio_max": 20,
            "target": "5000 windows / 10000 QA rows",
        },
        "outputs": {
            "qa_jsonl": str(qa_out),
            "qa_parquet": str(OUT / "roomtour3d_video_pose_qa_curated_10k.parquet"),
            "windows_jsonl": str(win_out),
            "windows_parquet": str(OUT / "roomtour3d_video_pose_windows_curated_5k.parquet"),
            "preview_html": str(OUT / "preview.html"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(curated_qas, selected, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
