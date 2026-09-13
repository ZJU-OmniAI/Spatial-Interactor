#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_motion_run_033_pilot as pilot  # noqa: E402
import build_motion_run_index as base  # noqa: E402


OUT_ROOT = Path("/path/to/workspace/DATA")


def retry_eagain(fn, *, attempts: int = 20, delay_s: float = 0.5):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


def write_json(path: Path, payload: object) -> None:
    retry_eagain(lambda: path.parent.mkdir(parents=True, exist_ok=True))
    retry_eagain(lambda: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    retry_eagain(lambda: path.parent.mkdir(parents=True, exist_ok=True))

    def _write() -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    retry_eagain(_write)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    retry_eagain(lambda: path.parent.mkdir(parents=True, exist_ok=True))

    def _append() -> None:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    retry_eagain(_append)


def read_existing_index(path: Path) -> tuple[set[str], int, Counter, Counter]:
    scenes: set[str] = set()
    segment_count = 0
    type_counts: Counter = Counter()
    action_counts: Counter = Counter()
    if not retry_eagain(lambda: path.exists()):
        return scenes, segment_count, type_counts, action_counts
    with retry_eagain(lambda: path.open("r", encoding="utf-8")) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("scene"):
                scenes.add(str(row["scene"]))
            segment_count += 1
            type_counts[row.get("segment_type")] += 1
            if row.get("action_label"):
                action_counts[row["action_label"]] += 1
    return scenes, segment_count, type_counts, action_counts


def label_from_axis(axis: str, sign: int) -> tuple[str, str, str]:
    if axis == "forward_m":
        return ("向前移动" if sign > 0 else "向后移动", "translation", "m")
    if axis == "right_m":
        return ("向右移动" if sign > 0 else "向左移动", "translation", "m")
    if axis == "right_turn_deg":
        return ("向右转动" if sign > 0 else "向左转动", "rotation", "deg")
    if axis == "up_turn_deg":
        return ("向上转动" if sign > 0 else "向下转动", "rotation", "deg")
    raise ValueError(axis)


def segment_kind(run: dict, args: argparse.Namespace) -> str:
    if run["unit"] == "m":
        if run["num_steps_033"] >= args.long_steps or run["cumulative_value"] >= args.long_translation_m:
            return "长"
    else:
        if run["num_steps_033"] >= args.long_steps or run["cumulative_value"] >= args.long_rotation_deg:
            return "长"
    return "短"


def summarize_window(
    loaded: base.LoadedScene,
    steps: list[dict],
    start: int,
    end: int,
    args: argparse.Namespace,
) -> dict:
    component_sums = {"forward_m": 0.0, "right_m": 0.0, "right_turn_deg": 0.0, "up_turn_deg": 0.0}
    component_abs = {k: 0.0 for k in component_sums}
    pos = {k: 0.0 for k in component_sums}
    neg = {k: 0.0 for k in component_sums}
    total_norm = 0.0
    step_labels = Counter()
    for step in steps[start:end]:
        if step["label"]:
            step_labels[step["label"]] += 1
        for key, value in step["axes"].items():
            component_sums[key] += value
            component_abs[key] += abs(value)
            if value >= 0:
                pos[key] += value
            else:
                neg[key] += -value
        total_norm += abs(step["axes"]["forward_m"]) / args.trans_unit_m
        total_norm += abs(step["axes"]["right_m"]) / args.trans_unit_m
        total_norm += abs(step["axes"]["right_turn_deg"]) / args.rot_unit_deg
        total_norm += abs(step["axes"]["up_turn_deg"]) / args.rot_unit_deg

    candidates = []
    for axis in component_sums:
        for sign in (1, -1):
            unit_scale = args.trans_unit_m if axis.endswith("_m") else args.rot_unit_deg
            value = pos[axis] if sign > 0 else neg[axis]
            label, family, unit = label_from_axis(axis, sign)
            candidates.append((value / unit_scale, value, axis, sign, label, family, unit))
    primary_norm, primary_value, axis, sign, label, family, unit = max(candidates, key=lambda x: x[0])
    opposite = neg[axis] if sign > 0 else pos[axis]
    unit_scale = args.trans_unit_m if unit == "m" else args.rot_unit_deg
    opposite_norm = opposite / unit_scale
    purity = primary_norm / max(total_norm, 1e-8)
    monotonicity = primary_norm / max(primary_norm + opposite_norm, 1e-8)

    start_slot = steps[start]["start_slot"]
    end_slot = steps[end - 1]["end_slot"]
    return {
        "dataset": loaded.dataset,
        "scene": loaded.scene_id,
        "candidate_label": label,
        "candidate_family": family,
        "candidate_unit": unit,
        "candidate_cumulative_value": float(primary_value),
        "purity": float(purity),
        "monotonicity": float(monotonicity),
        "num_steps_033": int(end - start),
        "slot_start": int(start_slot),
        "slot_end": int(end_slot),
        "start_frame_index": int(loaded.frame_ids[start_slot]),
        "end_frame_index": int(loaded.frame_ids[end_slot]),
        "frame_gap": int(loaded.frame_ids[end_slot] - loaded.frame_ids[start_slot]),
        "resampled_frame_indices": [int(steps[i]["start_frame_index"]) for i in range(start, end)] + [int(steps[end - 1]["end_frame_index"])],
        "frame_indices": [int(x) for x in loaded.frame_ids[start_slot : end_slot + 1]],
        "component_sums": {k: float(v) for k, v in component_sums.items()},
        "component_abs_sums": {k: float(v) for k, v in component_abs.items()},
        "step_label_counts": dict(step_labels),
        "target_frame_gap_per_step": pilot.TARGET_FRAME_GAP[loaded.dataset],
        "rule_metadata": loaded.metadata,
    }


def best_pure_from(
    loaded: base.LoadedScene,
    steps: list[dict],
    start: int,
    args: argparse.Namespace,
) -> dict | None:
    candidates = []
    max_end = min(len(steps), start + args.max_run_steps)
    for end in range(start + args.min_steps, max_end + 1):
        run = pilot.evaluate_window(loaded, steps, start, end, args)
        if run is None:
            continue
        candidates.append(run)
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            r["num_steps_033"],
            r["cumulative_value"],
            r["purity"],
            r["monotonicity"],
        ),
        reverse=True,
    )
    return candidates[0]


def segment_scene(loaded: base.LoadedScene, args: argparse.Namespace) -> tuple[list[dict], dict]:
    target_gap = pilot.TARGET_FRAME_GAP[loaded.dataset]
    slots = pilot.resample_slots(loaded.frame_ids, target_gap)
    steps = pilot.step_axes(loaded, slots, args)
    best_cache: dict[int, dict | None] = {}

    def get_best(start: int) -> dict | None:
        if start not in best_cache:
            best_cache[start] = best_pure_from(loaded, steps, start, args)
        return best_cache[start]

    segments: list[dict] = []
    cursor = 0
    while cursor < len(steps):
        pure = get_best(cursor)
        if pure is not None:
            pure["segment_type"] = segment_kind(pure, args)
            pure["is_pure"] = True
            pure["action_label"] = pure["label"]
            pure["segment_id"] = f"{loaded.dataset}_{loaded.scene_id}_{len(segments):05d}"
            pure["resampled_frame_indices"] = [
                int(steps[i]["start_frame_index"]) for i in range(cursor, cursor + pure["num_steps_033"])
            ] + [int(steps[cursor + pure["num_steps_033"] - 1]["end_frame_index"])]
            segments.append(pure)
            cursor += int(pure["num_steps_033"])
            continue

        next_pure = None
        search_limit = min(len(steps), cursor + args.max_impure_steps)
        for nxt in range(cursor + 1, search_limit + 1):
            if get_best(nxt) is not None:
                next_pure = nxt
                break
        end = next_pure if next_pure is not None else search_limit
        if end <= cursor:
            end = min(len(steps), cursor + 1)
        impure = summarize_window(loaded, steps, cursor, end, args)
        impure["segment_type"] = "不纯"
        impure["is_pure"] = False
        impure["action_label"] = None
        impure["unit"] = None
        impure["cumulative_value"] = None
        impure["segment_id"] = f"{loaded.dataset}_{loaded.scene_id}_{len(segments):05d}"
        segments.append(impure)
        cursor = end

    stats = {
        "raw_slots": int(len(loaded.frame_ids)),
        "resampled_slots": int(len(slots)),
        "resampled_steps": int(len(steps)),
        "segments": int(len(segments)),
        "target_frame_gap": int(target_gap),
    }
    return segments, stats


def dataset_limit(name: str, args: argparse.Namespace) -> int:
    specific = {
        "scannetv2": args.max_scenes_scannetv2,
        "scannetpp": args.max_scenes_scannetpp,
        "multiscan": args.max_scenes_multiscan,
        "3rscan": args.max_scenes_3rscan,
        "arkit": args.max_scenes_arkit,
    }[name]
    return specific if specific is not None else args.scenes_per_dataset


def dataset_iter(name: str, args: argparse.Namespace):
    limit = dataset_limit(name, args)
    if name == "scannetv2":
        return base.iter_scannetv2(limit, args.skip_scenes)
    if name == "scannetpp":
        return base.iter_scannetpp(limit, args.skip_scenes)
    if name == "multiscan":
        return base.iter_multiscan(limit, args.skip_scenes)
    if name == "3rscan":
        return base.iter_3rscan(limit, args.skip_scenes)
    if name == "arkit":
        return base.iter_arkit(limit, args.skip_scenes)
    raise ValueError(name)


def build_dataset(name: str, args: argparse.Namespace) -> dict:
    scene_stats = []
    segment_type_counts = Counter()
    action_counts = Counter()
    scene_count = 0
    out_dir = args.output_root / name
    segment_path = out_dir / "segments.jsonl"
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    existing_scenes: set[str] = set()
    existing_segments = 0
    if args.overwrite:
        retry_eagain(lambda: segment_path.write_text("", encoding="utf-8"))
    else:
        existing_scenes, existing_segments, segment_type_counts, action_counts = read_existing_index(segment_path)
    for loaded in dataset_iter(name, args):
        if loaded.scene_id in existing_scenes:
            continue
        scene_count += 1
        scene_segments, stats = segment_scene(loaded, args)
        for segment in scene_segments:
            segment_type_counts[segment["segment_type"]] += 1
            if segment.get("action_label"):
                action_counts[segment["action_label"]] += 1
        append_jsonl(segment_path, scene_segments)
        stats.update({"scene": loaded.scene_id})
        scene_stats.append(stats)
        print(
            f"[segments] {name}/{loaded.scene_id}: {len(scene_segments)} segments "
            f"({dict(Counter(s['segment_type'] for s in scene_segments))})",
            flush=True,
        )

    summary = {
        "dataset": name,
        "scenes_scanned_this_run": scene_count,
        "existing_scenes_skipped": len(existing_scenes),
        "segments_total": existing_segments + sum(stats["segments"] for stats in scene_stats),
        "segment_type_counts": dict(segment_type_counts),
        "action_counts_pure_segments": dict(action_counts),
        "target_frame_gap": pilot.TARGET_FRAME_GAP[name],
        "approx_step_seconds": "about 0.33s",
        "thresholds": {
            "min_purity_for_pure": args.min_purity,
            "min_monotonicity_for_pure": args.min_monotonicity,
            "min_translation_m_for_pure": args.min_translation_m,
            "min_rotation_deg_for_pure": args.min_rotation_deg,
            "long_steps": args.long_steps,
            "long_translation_m": args.long_translation_m,
            "long_rotation_deg": args.long_rotation_deg,
        },
        "scene_stats": scene_stats,
        "segments_jsonl": str(out_dir / "segments.jsonl"),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-coverage 0.33s motion segmentation pilot.")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["scannetv2", "scannetpp", "multiscan", "3rscan", "arkit"])
    parser.add_argument("--scenes-per-dataset", type=int, default=10)
    parser.add_argument("--skip-scenes", type=int, default=0)
    parser.add_argument("--max-scenes-scannetv2", type=int)
    parser.add_argument("--max-scenes-scannetpp", type=int)
    parser.add_argument("--max-scenes-multiscan", type=int)
    parser.add_argument("--max-scenes-3rscan", type=int)
    parser.add_argument("--max-scenes-arkit", type=int)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--trans-unit-m", type=float, default=0.05)
    parser.add_argument("--rot-unit-deg", type=float, default=4.0)
    parser.add_argument("--min-step-norm", type=float, default=0.12)
    parser.add_argument("--step-ratio", type=float, default=1.15)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-run-steps", type=int, default=90)
    parser.add_argument("--max-impure-steps", type=int, default=18)
    parser.add_argument("--min-purity", type=float, default=0.76)
    parser.add_argument("--min-monotonicity", type=float, default=0.94)
    parser.add_argument("--min-translation-m", type=float, default=0.35)
    parser.add_argument("--min-rotation-deg", type=float, default=30.0)
    parser.add_argument("--long-steps", type=int, default=24)
    parser.add_argument("--long-translation-m", type=float, default=1.2)
    parser.add_argument("--long-rotation-deg", type=float, default=90.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name in args.datasets:
        if args.overwrite:
            out_dir = args.output_root / name
            if out_dir.exists():
                shutil.rmtree(out_dir)
        summaries[name] = build_dataset(name, args)
    write_json(args.output_root / "motion_segments_033_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
