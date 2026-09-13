#!/usr/bin/env python3
from __future__ import annotations

import html
import io
import json
import math
import pickle
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/path/to/workspace/roomtour3d_video_pose_20260527")
TRAJ_JSON = ROOT / "roomtour3d/trajectories/p1_train_colmap_trajectory_tour3d.json"
GEO_PKL = ROOT / "roomtour3d/geometry/geo_trajectory.pkl"
VIDEO_DIR = ROOT / "room_tour_video_3fps"
OUT = ROOT / "generated_video_pose_qa_metric_continuous_60_180s_20260527"

FPS = 3
SAMPLE_FRAMES = 32
MIN_SECONDS = 60.0
MAX_SECONDS = 180.0
MAX_POSE_GAP_FRAMES = 3
EXPORT_ALL_IMAGES = True

STRAIGHT_BINS = [
    (0.0, 3.0, "0-3 meters"),
    (3.0, 6.0, "3-6 meters"),
    (6.0, 9.0, "6-9 meters"),
    (9.0, float("inf"), "more than 9 meters"),
]
PATH_BINS = [
    (0.0, 4.0, "0-4 meters"),
    (4.0, 8.0, "4-8 meters"),
    (8.0, 12.0, "8-12 meters"),
    (12.0, float("inf"), "more than 12 meters"),
]
ANGLE_BINS = [
    (0.0, 15.0, "0-15 degrees"),
    (15.0, 30.0, "15-30 degrees"),
    (30.0, float("inf"), "30-45 degrees"),
]

MOTION_LABELS = [
    "mostly moves forward through the scene",
    "moves forward and turns left",
    "moves forward and turns right",
    "moves toward a visible target area or object",
    "moves from one room into another room",
    "mostly scans or pivots with limited translation",
]

ROOM_WORDS = [
    "hallway",
    "living room",
    "bedroom",
    "bathroom",
    "office",
    "kitchen",
    "dining room",
    "stairwell",
    "closet",
    "entryway",
    "corridor",
    "balcony",
    "laundry room",
    "garage",
]


def frame_num(name: str) -> int | None:
    m = re.search(r"output_frame_(\d+)\.png$", name)
    return int(m.group(1)) if m else None


def frame_name(num: int) -> str:
    return f"output_frame_{num:04d}.png"


def model_id(long_id: str) -> str:
    return long_id.split("|", 1)[0]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    return x


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def split_runs(frames: list[int]) -> list[list[int]]:
    frames = sorted(set(frames))
    if not frames:
        return []
    runs: list[list[int]] = []
    cur = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b - a <= MAX_POSE_GAP_FRAMES:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


def sample_32(start: int, end: int, available: set[int]) -> list[int]:
    raw = [int(round(x)) for x in np.linspace(start, end, SAMPLE_FRAMES)]
    out: list[int] = []
    used = set()
    sorted_available = sorted(x for x in available if start <= x <= end)
    if not sorted_available:
        return []
    for target in raw:
        best = min(sorted_available, key=lambda x: abs(x - target))
        if best not in used:
            out.append(best)
            used.add(best)
    for x in sorted_available:
        if len(out) >= SAMPLE_FRAMES:
            break
        if x not in used:
            out.append(x)
            used.add(x)
    return sorted(out[:SAMPLE_FRAMES])


def load_video_frame_nums(video_zip: Path, video_id: str) -> set[int]:
    if not video_zip.exists():
        return set()
    prefix = f"{video_id}/imgs_3fps_360p/"
    nums = set()
    with zipfile.ZipFile(video_zip) as zf:
        for name in zf.namelist():
            if name.startswith(prefix):
                n = frame_num(name)
                if n is not None:
                    nums.add(n)
    return nums


def positions_for_run(geo_model: dict[int, dict[str, Any]], frames: list[int]) -> list[tuple[int, np.ndarray]]:
    out = []
    for f in frames:
        if f not in geo_model:
            continue
        pos = np.asarray(geo_model[f]["real_world_position"], dtype=float)
        out.append((f, pos))
    return out


def path_metrics(points: list[tuple[int, np.ndarray]]) -> dict[str, float | int]:
    positions = [p for _, p in points]
    diffs = [float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]
    path_len = float(sum(diffs))
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    delta = positions[-1] - positions[0]
    dx, dz = float(delta[0]), float(delta[2])
    angle = math.degrees(math.atan2(abs(dz), abs(dx))) if abs(dx) + abs(dz) > 1e-8 else 0.0
    angle = min(angle, 90.0 - angle)
    return {
        "straight_meters": straight,
        "path_meters": path_len,
        "axis_angle_deg": angle,
        "path_to_straight_ratio": path_len / max(straight, 1e-6),
        "pose_frame_count": len(points),
    }


def bin_index(value: float, bins: list[tuple[float, float, str]]) -> int:
    for i, (lo, hi, _) in enumerate(bins):
        if lo <= value < hi:
            return i
    return len(bins) - 1


def interval_text(choice: tuple[int, int, int]) -> str:
    return (
        f"straight-line displacement {STRAIGHT_BINS[choice[0]][2]}, "
        f"axis angle {ANGLE_BINS[choice[1]][2]}, "
        f"total path length {PATH_BINS[choice[2]][2]}"
    )


def interval_options(correct: tuple[int, int, int], rng: random.Random) -> tuple[list[dict[str, str]], str]:
    candidates = [correct]
    scored = []
    for s in range(len(STRAIGHT_BINS)):
        for a in range(len(ANGLE_BINS)):
            for p in range(len(PATH_BINS)):
                cand = (s, a, p)
                if cand != correct:
                    scored.append((sum(abs(cand[i] - correct[i]) for i in range(3)), cand))
    rng.shuffle(scored)
    for _, cand in sorted(scored, key=lambda x: x[0]):
        if cand not in candidates:
            candidates.append(cand)
        if len(candidates) == 4:
            break
    rng.shuffle(candidates)
    letters = ["A", "B", "C", "D"]
    answer = letters[candidates.index(correct)]
    return [{"key": k, "text": interval_text(c)} for k, c in zip(letters, candidates)], answer


def find_rooms(text: str) -> list[str]:
    low = text.lower()
    return [room for room in ROOM_WORDS if room in low]


def motion_label(text: str, metrics: dict[str, float | int]) -> str:
    low = text.lower()
    rooms = find_rooms(text)
    if "turn left" in low or "veer left" in low or "to the left" in low:
        return "moves forward and turns left"
    if "turn right" in low or "veer right" in low or "to the right" in low:
        return "moves forward and turns right"
    if len(set(rooms)) >= 2:
        return "moves from one room into another room"
    if "approach" in low or "towards" in low or "toward" in low:
        return "moves toward a visible target area or object"
    if float(metrics["straight_meters"]) < 0.8 or float(metrics["path_to_straight_ratio"]) > 3.0:
        return "mostly scans or pivots with limited translation"
    return "mostly moves forward through the scene"


def motion_options(correct: str, rng: random.Random) -> tuple[list[dict[str, str]], str]:
    opts = [correct]
    pool = [x for x in MOTION_LABELS if x != correct]
    rng.shuffle(pool)
    opts.extend(pool[:3])
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    answer = letters[opts.index(correct)]
    return [{"key": k, "text": o} for k, o in zip(letters, opts)], answer


def make_qas(window: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    metrics = window["metrics"]
    base = {
        "dataset": "roomtour3d_metric_continuous_video_pose_qa_20260527",
        "source": "roomtour3d",
        "video_id": window["video_id"],
        "path_id": window["path_id"],
        "clip_id": window["clip_id"],
        "input": {
            "media_type": "image_sequence_32frames",
            "video_zip": window["video_zip"],
            "frame_rate_fps": FPS,
            "sampled_frame_indices_32": window["sampled_frame_indices_32"],
            "sampled_frame_names_32": window["sampled_frame_names_32"],
            "sampled_image_paths_32": window.get("sampled_image_paths_32", []),
        },
        "question_language": "en",
        "answer_language": "en",
        "answer_style": "mcq",
        "answer_format": "multiple_choice",
        "metadata": {
            "source_instruction": window["source_instruction"],
            "geometry_pickle": str(GEO_PKL),
            "geometry_model_id": window["geometry_model_id"],
            "pose_source": "official geometry real_world_position, in meters",
            "pose_frame_indices_used_for_gt": window["pose_frame_indices_used_for_gt"],
            "pose_metrics": metrics,
            "continuity_policy": f"continuous geometry run, max pose gap <= {MAX_POSE_GAP_FRAMES} frames",
        },
    }
    correct = (
        bin_index(float(metrics["straight_meters"]), STRAIGHT_BINS),
        bin_index(float(metrics["axis_angle_deg"]), ANGLE_BINS),
        bin_index(float(metrics["path_meters"]), PATH_BINS),
    )
    options, answer = interval_options(correct, rng)
    qa1 = dict(base)
    qa1.update(
        {
            "qa_id": f"roomtour3d_{window['clip_id']}_metric_pose_interval_summary",
            "task_type": "roomtour3d_metric_pose_interval_summary",
            "question": (
                "Which option best describes the rough camera trajectory in this indoor first-person video? "
                "Estimate straight-line displacement, the angle of that displacement to the nearest horizontal/vertical "
                "ground-plane axis, and total path length. Reply with only one letter."
            ),
            "options": options,
            "answer": answer,
            "correct_option": answer,
            "answer_text": next(o["text"] for o in options if o["key"] == answer),
            "gt": {"coarse_answer_text": next(o["text"] for o in options if o["key"] == answer), "exact_pose_values": metrics},
        }
    )
    motion = motion_label(window["source_instruction"], metrics)
    options2, answer2 = motion_options(motion, rng)
    qa2 = dict(base)
    qa2.update(
        {
            "qa_id": f"roomtour3d_{window['clip_id']}_coarse_video_motion",
            "task_type": "roomtour3d_coarse_video_motion",
            "question": "Which broad camera motion pattern is most consistent with this indoor first-person video? Reply with only one letter.",
            "options": options2,
            "answer": answer2,
            "correct_option": answer2,
            "answer_text": motion,
            "gt": {"motion_label": motion, "source_instruction": window["source_instruction"], "exact_pose_values": metrics},
        }
    )
    return [qa1, qa2]


def export_images(video_zip: Path, video_id: str, clip_id: str, frames: list[int], dst_root: Path) -> list[str]:
    dst_dir = dst_root / clip_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(video_zip) as zf:
        names = set(zf.namelist())
        for i, n in enumerate(frames, 1):
            inner = f"{video_id}/imgs_3fps_360p/{frame_name(n)}"
            if inner not in names:
                continue
            dst = dst_dir / f"{i:02d}_{frame_name(n)}"
            if not dst.exists():
                with zf.open(inner) as src, dst.open("wb") as f:
                    shutil.copyfileobj(src, f)
            out.append(str(dst))
    return out


def trajectory_svg(points: list[tuple[int, np.ndarray]], sampled: list[int]) -> str:
    if not points:
        return ""
    w, h, pad = 420, 420, 32
    pts = np.array([[p[0], p[2]] for _, p in points], dtype=float)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    scale = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1])

    def mapp(pos: np.ndarray) -> tuple[float, float]:
        xy = (np.array([pos[0], pos[2]]) - mn) * scale + pad
        return float(xy[0]), float(h - xy[1])

    elems = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fbfbfa" stroke="#ccc"/>']
    poly = " ".join(f"{mapp(p)[0]:.1f},{mapp(p)[1]:.1f}" for _, p in points)
    elems.append(f'<polyline points="{poly}" fill="none" stroke="#999" stroke-width="2"/>')
    sampled_set = set(sampled)
    for fn, p in points:
        x, y = mapp(p)
        color, r = ("#1a73e8", 4) if fn in sampled_set else ("#bbb", 2)
        elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" opacity="0.9"><title>{fn}</title></circle>')
    sx, sy = mapp(points[0][1])
    ex, ey = mapp(points[-1][1])
    elems.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="#188038"/>')
    elems.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="7" fill="#d93025"/>')
    elems.append('<text x="12" y="402" font-size="12" fill="#555">green=start, red=end, blue=sampled</text>')
    return f'<svg viewBox="0 0 {w} {h}">' + "".join(elems) + "</svg>"


def write_preview(windows: list[dict[str, Any]], qas_by_clip: dict[str, list[dict[str, Any]]], out: Path) -> None:
    cards = []
    for i, win in enumerate(windows[:20], 1):
        imgs = "".join(
            f"<figure><img src='{html.escape(str(Path(p).relative_to(out)))}'><figcaption>{html.escape(Path(p).name)}</figcaption></figure>"
            for p in win.get("sampled_image_paths_32", [])
        )
        qas = []
        for q in qas_by_clip.get(win["clip_id"], []):
            opts = "".join(f"<li><b>{o['key']}</b>. {html.escape(o['text'])}</li>" for o in q["options"])
            qas.append(
                f"<div class='qa'><div class='task'>{html.escape(q['task_type'])} | answer {html.escape(q['answer'])}</div>"
                f"<p>{html.escape(q['question'])}</p><ol type='A'>{opts}</ol></div>"
            )
        m = win["metrics"]
        cards.append(
            f"<section><h2>{i}. {html.escape(win['clip_id'])}</h2><div class='meta'>video={html.escape(win['video_id'])} | "
            f"duration={win['clip_duration_seconds']:.1f}s | frames={win['clip_start_frame']}-{win['clip_end_frame']} | "
            f"straight={m['straight_meters']:.2f}m | path={m['path_meters']:.2f}m | angle={m['axis_angle_deg']:.1f} deg</div>"
            f"<div class='top'><div>{win['trajectory_svg']}</div><div>{''.join(qas)}</div></div><div class='grid'>{imgs}</div></section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomTour3D Metric Continuous Preview</title>
<style>
body{{margin:0;font-family:Arial,sans-serif;background:#f5f5f2;color:#1d1d1b}}header{{position:sticky;top:0;background:#202124;color:white;padding:14px 20px;z-index:5}}
main{{padding:18px;max-width:1600px;margin:auto}}section{{background:white;border:1px solid #d8d8d2;border-radius:6px;padding:16px;margin-bottom:22px}}
h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:18px;margin:0 0 8px}}.note,.meta{{font-size:13px;color:#555;line-height:1.45}}header .note{{color:#ddd}}
.top{{display:grid;grid-template-columns:440px 1fr;gap:14px;align-items:start}}.qa{{border:1px solid #ddd;border-radius:6px;background:#fbfbfa;padding:10px;margin-bottom:10px}}
.task{{font-size:12px;color:#555}}.grid{{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:8px;margin-top:12px}}
figure{{margin:0;background:#111;border-radius:4px;overflow:hidden}}img{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain}}figcaption{{font-size:11px;background:#222;color:#eee;padding:4px}}
@media(max-width:1000px){{.top{{grid-template-columns:1fr}}.grid{{grid-template-columns:repeat(4,1fr)}}}}
</style></head><body><header><h1>RoomTour3D Metric Continuous 60-180s Preview</h1>
<div class="note">GT uses official real_world_position in meters; clips are continuous geometry runs with max pose gap <= {MAX_POSE_GAP_FRAMES} frames.</div></header>
<main>{''.join(cards)}</main></body></html>"""
    (out / "preview_metric_continuous.html").write_text(page, encoding="utf-8")


def main() -> None:
    rng = random.Random(20260527)
    OUT.mkdir(parents=True, exist_ok=True)
    image_root = OUT / "sampled_32frames"
    if EXPORT_ALL_IMAGES:
        image_root.mkdir(parents=True, exist_ok=True)

    traj_rows = json.loads(TRAJ_JSON.read_text(encoding="utf-8"))
    with GEO_PKL.open("rb") as f:
        geo = pickle.load(f)

    video_frame_cache: dict[str, set[int]] = {}
    windows: list[dict[str, Any]] = []
    skipped = Counter()

    for row in traj_rows:
        video_id = row["videoId"]
        mid = model_id(row["longId"])
        video_zip = VIDEO_DIR / f"{video_id}.zip"
        if not video_zip.exists():
            skipped["missing_video_zip"] += 1
            continue
        if mid not in geo:
            skipped["missing_geometry_model"] += 1
            continue
        raw_nums = [frame_num(x) for x in row.get("path", [])]
        path_nums = [x for x in raw_nums if x is not None]
        if not path_nums:
            skipped["empty_path"] += 1
            continue
        start0, end0 = min(path_nums), max(path_nums)
        geo_model = geo[mid]
        available_geo = [int(k) for k in geo_model.keys() if start0 <= int(k) <= end0]
        if not available_geo:
            skipped["no_geometry_frames_in_path_span"] += 1
            continue
        if video_id not in video_frame_cache:
            video_frame_cache[video_id] = load_video_frame_nums(video_zip, video_id)
        available_video = video_frame_cache[video_id]
        for run_i, run in enumerate(split_runs(available_geo)):
            start, end = run[0], run[-1]
            duration = (end - start + 1) / FPS
            if duration < MIN_SECONDS or duration > MAX_SECONDS:
                skipped["run_duration_outside_60_180s"] += 1
                continue
            sampled = sample_32(start, end, available_video)
            if len(sampled) != SAMPLE_FRAMES:
                skipped["could_not_sample_32_video_frames"] += 1
                continue
            points = positions_for_run(geo_model, run)
            if len(points) < 2:
                skipped["too_few_metric_pose_points"] += 1
                continue
            metrics = path_metrics(points)
            clip_id = f"{row['path_id']}_contig{run_i:02d}"
            image_paths = export_images(video_zip, video_id, clip_id, sampled, image_root) if EXPORT_ALL_IMAGES else []
            if EXPORT_ALL_IMAGES and len(image_paths) != SAMPLE_FRAMES:
                skipped["image_export_incomplete"] += 1
                shutil.rmtree(image_root / clip_id, ignore_errors=True)
                continue
            window = {
                "source": "roomtour3d",
                "video_id": video_id,
                "path_id": row["path_id"],
                "clip_id": clip_id,
                "video_zip": str(video_zip),
                "geometry_model_id": mid,
                "sampled_frame_indices_32": sampled,
                "sampled_frame_names_32": [frame_name(n) for n in sampled],
                "sampled_image_paths_32": image_paths,
                "pose_frame_indices_used_for_gt": [fn for fn, _ in points],
                "metrics": metrics,
                "source_instruction": clean_text(row["instructions"][0]),
                "original_path_frames": row.get("path", []),
                "longId": row["longId"],
                "clip_duration_seconds": duration,
                "clip_start_frame": start,
                "clip_end_frame": end,
                "continuity": {
                    "max_pose_gap_frames": max([b - a for a, b in zip(run, run[1:])] or [0]),
                    "max_allowed_pose_gap_frames": MAX_POSE_GAP_FRAMES,
                    "pose_frame_count": len(run),
                },
                "trajectory_svg": trajectory_svg(points, sampled),
            }
            windows.append(window)

    qa_rows: list[dict[str, Any]] = []
    qas_by_clip: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        qas = make_qas(window, rng)
        qas_by_clip[window["clip_id"]] = qas
        qa_rows.extend(qas)

    # SVG is only for preview; keep the machine-readable window files smaller.
    windows_to_write = [{k: v for k, v in w.items() if k != "trajectory_svg"} for w in windows]
    qa_jsonl = OUT / "roomtour3d_metric_continuous_qa_60_180s.jsonl"
    win_jsonl = OUT / "roomtour3d_metric_continuous_windows_60_180s.jsonl"
    write_jsonl(qa_jsonl, qa_rows)
    write_jsonl(win_jsonl, windows_to_write)
    pd.DataFrame(json_safe(qa_rows)).to_parquet(OUT / "roomtour3d_metric_continuous_qa_60_180s.parquet", index=False)
    pd.DataFrame(json_safe(windows_to_write)).to_parquet(OUT / "roomtour3d_metric_continuous_windows_60_180s.parquet", index=False)

    durs = [w["clip_duration_seconds"] for w in windows]
    summary = {
        "trajectory_json": str(TRAJ_JSON),
        "geometry_pickle": str(GEO_PKL),
        "output_dir": str(OUT),
        "kept_windows": len(windows),
        "qa_rows": len(qa_rows),
        "unique_videos": len({w["video_id"] for w in windows}),
        "task_counts": dict(Counter(q["task_type"] for q in qa_rows)),
        "answer_counts": dict(Counter(q["answer"] for q in qa_rows)),
        "duration_seconds": {
            "min": min(durs) if durs else None,
            "median": float(np.median(durs)) if durs else None,
            "mean": float(np.mean(durs)) if durs else None,
            "max": max(durs) if durs else None,
        },
        "frame_policy": f"sample exactly {SAMPLE_FRAMES} frames uniformly from each continuous 60-180s run",
        "gt_policy": "use all official geometry real_world_position points inside the continuous run; units are meters",
        "continuity_policy": f"split geometry frames when gap > {MAX_POSE_GAP_FRAMES} frames; keep only runs between 60 and 180 seconds",
        "export_all_images": EXPORT_ALL_IMAGES,
        "sampled_image_root": str(image_root),
        "skipped": dict(skipped),
        "outputs": {
            "qa_jsonl": str(qa_jsonl),
            "qa_parquet": str(OUT / "roomtour3d_metric_continuous_qa_60_180s.parquet"),
            "windows_jsonl": str(win_jsonl),
            "windows_parquet": str(OUT / "roomtour3d_metric_continuous_windows_60_180s.parquet"),
            "preview_html": str(OUT / "preview_metric_continuous.html"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(windows, qas_by_clip, OUT)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
