#!/usr/bin/env python3
import argparse
import json
import math
import os
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


ALLOWED_DIRS = {"forward", "backward", "left", "right", "up", "down"}
ALLOWED_VIDEO_ANSWERS = {
    "turn around, move forward",
    "turn around, move forward, then turn left and move forward",
    "turn around, move forward, then turn right and move forward",
    "turn around, move forward, then turn left and move forward, then turn left and move forward",
    "turn around, move forward, then turn left and move forward, then turn right and move forward",
    "turn around, move forward, then turn right and move forward, then turn left and move forward",
    "turn around, move forward, then turn right and move forward, then turn right and move forward",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as exc:
                yield ln, {"__bad_json__": repr(exc)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mcq_text(opts: list[dict[str, Any]]) -> str:
    return "\n".join(f"{o['label']}. {o.get('text', 'Candidate ' + o['label'])}" for o in opts)


def make_lf_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = row["input"].get("frame_labels") or [str(i + 1) for i in range(len(row["input"]["frame_paths"]))]
    user = "\n".join(f"<image> {lab}" for lab in labels)
    user += "\n" + row["question"] + "\n" + mcq_text(row["options"]) + "\nAnswer with the option letter only."
    return {
        "id": row["id"],
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": row["answer"]}],
        "images": row["input"]["frame_paths"],
        "metadata": {
            "task_type": row["task_type"],
            "subcat": row["subcat"],
            "source": row["source"],
            "gt": row["gt"],
            "answer_text": row["answer_text"],
            "quality_filter": row.get("quality_filter", {}),
        },
    }


def resolve_path(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else base / pp


def image_ok(path: Path, cache: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists() or path.stat().st_size < 1024:
        cache[key] = (False, "missing_or_tiny_image")
        return cache[key]
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if im.width < 96 or im.height < 96:
                cache[key] = (False, "image_too_small")
                return cache[key]
            stat = ImageStat.Stat(im.convert("L").resize((64, 64)))
            if stat.stddev[0] < 3.0:
                cache[key] = (False, "near_blank_image")
                return cache[key]
    except Exception as exc:
        cache[key] = (False, f"bad_image:{type(exc).__name__}")
        return cache[key]
    cache[key] = (True, "")
    return cache[key]


def zip_ok(path: Path, cache: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.exists() or path.stat().st_size < 10_000:
        cache[key] = (False, "missing_or_tiny_zip")
        return cache[key]
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) < 8:
                cache[key] = (False, "too_few_zip_frames")
                return cache[key]
    except Exception as exc:
        cache[key] = (False, f"bad_zip:{type(exc).__name__}")
        return cache[key]
    cache[key] = (True, "")
    return cache[key]


def common_checks(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("__bad_json__"):
        return False, "bad_json"
    for k in ["id", "source", "task_type", "subcat", "question", "options", "answer", "answer_text", "input", "gt"]:
        if k not in row:
            return False, f"missing_{k}"
    opts = row["options"]
    if not isinstance(opts, list) or len(opts) != 4:
        return False, "options_not_4"
    labels = [o.get("label") for o in opts]
    if labels != ["A", "B", "C", "D"]:
        return False, "bad_option_labels"
    if row["answer"] not in labels:
        return False, "answer_not_in_options"
    texts = [str(o.get("text", "")) for o in opts]
    if len(set(texts)) != 4:
        return False, "duplicate_option_text"
    if any("none of the above" in t.lower() for t in texts):
        return False, "none_option"
    correct_text = next(o.get("text") for o in opts if o.get("label") == row["answer"])
    if row.get("answer_text") and str(correct_text) != str(row["answer_text"]):
        return False, "answer_text_mismatch"
    paths = row.get("input", {}).get("frame_paths")
    if not isinstance(paths, list) or not paths:
        return False, "empty_frame_paths"
    return True, ""


def media_checks(row: dict[str, Any], base: Path, image_cache: dict[str, tuple[bool, str]], zip_cache: dict[str, tuple[bool, str]]) -> tuple[bool, str, list[str]]:
    resolved = []
    for p in row["input"]["frame_paths"]:
        rp = resolve_path(base, p)
        resolved.append(str(rp))
        suffix = rp.suffix.lower()
        if suffix == ".zip":
            ok, reason = zip_ok(rp, zip_cache)
        elif suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            ok, reason = image_ok(rp, image_cache)
        else:
            ok, reason = (rp.exists() and rp.stat().st_size > 0, "missing_media")
        if not ok:
            return False, reason, resolved
    return True, "", resolved


def recover_missing_media(row: dict[str, Any]) -> None:
    paths = row.get("input", {}).get("frame_paths")
    if paths:
        return
    gt = row.get("gt", {})
    candidates = [
        (gt.get("source_input") or {}).get("video_path"),
        (gt.get("source_metadata") or {}).get("video_path"),
        (gt.get("source_input") or {}).get("video_zip"),
    ]
    for p in candidates:
        if p:
            row.setdefault("input", {})["frame_paths"] = [p]
            row["input"]["frame_labels"] = ["video"]
            row.setdefault("quality_filter_fixes", []).append("recovered_media_from_source_gt")
            return


def finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def task_checks(row: dict[str, Any]) -> tuple[bool, str]:
    task = row["task_type"]
    gt = row["gt"]
    if task == "node_reverse_path_planning":
        dist = gt.get("straight_line_distance_m")
        plen = gt.get("cumulative_path_length_m")
        ans = row.get("answer_text")
        if row.get("source") not in {"roomtour3d", "SIMS-V", "simsv"}:
            return False, "bad_video_source"
        if ans not in ALLOWED_VIDEO_ANSWERS or gt.get("reverse_program") != ans or row.get("subcat") != ans:
            return False, "bad_reverse_answer"
        if not finite_number(dist) or not finite_number(plen):
            return False, "bad_video_distance"
        dist = float(dist)
        plen = float(plen)
        if dist < 3.0 or plen < 4.0:
            return False, "video_path_too_short"
        if plen + 1e-6 < dist:
            return False, "path_shorter_than_chord"
        if plen > max(25.0, dist * 3.2):
            return False, "video_path_too_indirect"
        return True, ""

    if task == "pure_two_frame_motion_magnitude":
        direction = gt.get("direction") or row.get("subcat")
        if direction not in ALLOWED_DIRS:
            return False, "bad_direction"
        dist = gt.get("distance_cm")
        if not finite_number(dist) or float(dist) < 4.0 or float(dist) > 35.0:
            return False, "bad_motion_distance"
        off = float(gt.get("off_axis_cm", 99.0))
        zoff = float(gt.get("vertical_off_cm", 99.0))
        grip = abs(float(gt.get("gripper_delta", 99.0)))
        if direction in {"up", "down"}:
            if off > 2.5:
                return False, "impure_vertical_motion"
        elif off > 2.5 or zoff > 2.5:
            return False, "impure_horizontal_motion"
        if grip > 0.08:
            return False, "gripper_changed"
        if len(row["input"]["frame_paths"]) != 2:
            return False, "bad_two_frame_count"
        return True, ""

    if task == "action_to_image_choice":
        direction = gt.get("direction") or row.get("subcat")
        if direction not in ALLOWED_DIRS:
            return False, "bad_direction"
        target = gt.get("target_distance_cm")
        if not finite_number(target) or float(target) < 4.0 or float(target) > 24.0:
            return False, "bad_target_distance"
        metrics = gt.get("candidate_metrics")
        if not isinstance(metrics, list) or len(metrics) != 4:
            return False, "bad_candidate_metrics"
        correct_metric = None
        for m in metrics:
            if m.get("label") == row["answer"]:
                correct_metric = m
            if float(m.get("off_axis_cm", 99.0)) > 2.5:
                return False, "candidate_impure_off_axis"
            if direction not in {"up", "down"} and float(m.get("vertical_off_cm", 99.0)) > 2.5:
                return False, "candidate_impure_vertical"
            if abs(float(m.get("gripper_delta", 99.0))) > 0.08:
                return False, "candidate_gripper_changed"
        if not correct_metric:
            return False, "missing_correct_candidate_metric"
        if abs(float(correct_metric["distance_cm"]) - float(target)) > 1.5:
            return False, "correct_candidate_not_target"
        if len(row["input"]["frame_paths"]) != 5:
            return False, "bad_candidate_frame_count"
        return True, ""

    if task == "temporal_sequence_sorting":
        if row.get("subcat") != "start_grasp_place_return":
            return False, "bad_sorting_subcat"
        if len(row["input"]["frame_paths"]) != 4:
            return False, "bad_sorting_frame_count"
        if float(gt.get("gripper_range", 0.0)) < 0.35:
            return False, "weak_gripper_state_change"
        if "Instruction:" not in row.get("question", ""):
            return False, "missing_instruction"
        if "->" not in row.get("answer_text", ""):
            return False, "bad_sorting_answer"
        return True, ""

    if task == "long_horizon_manipulation_program":
        s1 = gt.get("segment_1", {})
        s2 = gt.get("segment_2", {})
        if s1.get("direction") in {None, "slightly"} or s2.get("direction") in {None, "slightly"}:
            return False, "weak_segment_direction"
        if float(s1.get("distance_cm", 0.0)) < 10.0 or float(s2.get("distance_cm", 0.0)) < 10.0:
            return False, "segment_too_short"
        if float(gt.get("turn_angle_deg", 0.0)) < 30.0:
            return False, "weak_turn"
        frames = gt.get("sampled_frame_indices")
        if not isinstance(frames, list) or len(frames) < 6:
            return False, "too_few_video_frames"
        if "Which coarse horizontal manipulation program" not in row.get("question", ""):
            return False, "bad_long_program_question"
        return True, ""

    return False, "unknown_task"


def filter_dir(src_dir: Path, out_dir: Path, prefix: str, image_cache: dict[str, tuple[bool, str]], zip_cache: dict[str, tuple[bool, str]]) -> dict[str, Any]:
    kept = []
    rejected = []
    qa_path = src_dir / "qa_data.jsonl"
    if not qa_path.exists():
        return {"prefix": prefix, "missing": str(qa_path), "kept": 0, "rejected": 0}

    for ln, row in read_jsonl(qa_path):
        recover_missing_media(row)
        ok, reason = common_checks(row)
        resolved = []
        if ok:
            ok, reason, resolved = media_checks(row, src_dir, image_cache, zip_cache)
        if ok:
            ok, reason = task_checks(row)
        if ok:
            row = json.loads(json.dumps(row, ensure_ascii=False))
            row["id"] = f"{prefix}_{row['id']}"
            row["input"]["frame_paths"] = resolved
            row["quality_filter"] = {
                "kept": True,
                "source_dir": str(src_dir),
                "source_line": ln,
                "filtered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            kept.append(row)
        else:
            rejected.append({"line": ln, "id": row.get("id"), "task_type": row.get("task_type"), "subcat": row.get("subcat"), "reason": reason})

    write_jsonl(src_dir / "qa_data.filtered.jsonl", kept)
    write_jsonl(src_dir / "llamafactory_data.filtered.jsonl", [make_lf_row(r) for r in kept])
    write_jsonl(src_dir / "qa_data.rejected.jsonl", rejected)

    return {
        "prefix": prefix,
        "src_dir": str(src_dir),
        "kept": len(kept),
        "rejected": len(rejected),
        "by_task_type": dict(Counter(r["task_type"] for r in kept)),
        "by_task_type_subcat": {f"{k[0]}::{k[1]}": v for k, v in sorted(Counter((r["task_type"], r["subcat"]) for r in kept).items())},
        "reject_reasons": dict(Counter(r["reason"] for r in rejected)),
    }, kept, rejected


def make_tar(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with tarfile.open(dst, "w") as tar:
        tar.add(src, arcname=src.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=Path("/path/to/workspace/high_quality_new_interaction_video_reverse_full_20260615_work"))
    ap.add_argument("--robot-dir", type=Path, default=Path("/path/to/workspace/high_quality_new_interaction_robot_full_20260615_work"))
    ap.add_argument("--out", type=Path, default=Path("/path/to/workspace/high_quality_new_interaction_filtered_20260616_work"))
    ap.add_argument("--tar", type=Path, default=Path("/path/to/workspace/high_quality_new_interaction_filtered_20260616_metadata.tar"))
    ap.add_argument("--tar-to-dataset2", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, tuple[bool, str]] = {}
    zip_cache: dict[str, tuple[bool, str]] = {}
    all_kept: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    reports = []

    for src, prefix in [(args.video_dir, "video"), (args.robot_dir, "robot")]:
        report, kept, rejected = filter_dir(src, args.out, prefix, image_cache, zip_cache)
        reports.append(report)
        all_kept.extend(kept)
        all_rejected.extend({"prefix": prefix, **r} for r in rejected)

    write_jsonl(args.out / "qa_data.filtered.jsonl", all_kept)
    write_jsonl(args.out / "llamafactory_data.filtered.jsonl", [make_lf_row(r) for r in all_kept])
    write_jsonl(args.out / "qa_data.rejected.jsonl", all_rejected)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_kept": len(all_kept),
        "total_rejected": len(all_rejected),
        "reports": reports,
        "by_source": dict(Counter(r["source"] for r in all_kept)),
        "by_task_type": dict(Counter(r["task_type"] for r in all_kept)),
        "by_task_type_subcat": {f"{k[0]}::{k[1]}": v for k, v in sorted(Counter((r["task_type"], r["subcat"]) for r in all_kept).items())},
        "answer_distribution_by_task_type": {task: dict(cnt) for task, cnt in sorted(defaultdict(Counter, {
            task: Counter(r["answer"] for r in all_kept if r["task_type"] == task)
            for task in set(r["task_type"] for r in all_kept)
        }).items())},
        "media_mode": "filtered rows reference media in place by absolute path; original generated directories are not deleted.",
    }
    (args.out / "quality_filter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.tar_to_dataset2:
        make_tar(args.out, args.tar)
        print(f"[filter] wrote {args.tar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
