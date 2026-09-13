#!/usr/bin/env python3
from __future__ import annotations

import html
import io
import json
import math
import random
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/path/to/workspace/roomtour3d_video_pose_20260527")
TRAJ_JSON = ROOT / "roomtour3d/trajectories/p1_train_colmap_trajectory_tour3d.json"
VIDEO_DIR = ROOT / "room_tour_video_3fps"
COLMAP_DIR = ROOT / "roomtour3d/colmap_reconstruction"
OUT = ROOT / "generated_video_pose_qa_20260527"

SAMPLE_FRAMES = 32
MIN_UNIQUE_FRAMES = 12

STRAIGHT_BINS = [
    (0.0, 1.0, "0-1 COLMAP units"),
    (1.0, 2.0, "1-2 COLMAP units"),
    (2.0, 4.0, "2-4 COLMAP units"),
    (4.0, float("inf"), "more than 4 COLMAP units"),
]
PATH_BINS = [
    (0.0, 2.0, "0-2 COLMAP units"),
    (2.0, 4.0, "2-4 COLMAP units"),
    (4.0, 8.0, "4-8 COLMAP units"),
    (8.0, float("inf"), "more than 8 COLMAP units"),
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def frame_num(name: str) -> int | None:
    m = re.search(r"output_frame_(\d+)\.png$", name)
    return int(m.group(1)) if m else None


def frame_name(num: int) -> str:
    return f"output_frame_{num:04d}.png"


def colmap_name(path_frame: str) -> str:
    n = frame_num(path_frame)
    return frame_name(n) if n is not None else path_frame


def qvec2rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def read_images_bin(data: bytes) -> dict[str, np.ndarray]:
    f = io.BytesIO(data)
    image_count = struct.unpack("<Q", f.read(8))[0]
    images: dict[str, np.ndarray] = {}
    for _ in range(image_count):
        f.read(4)
        qvec = struct.unpack("<dddd", f.read(32))
        tvec = np.array(struct.unpack("<ddd", f.read(24)), dtype=float)
        f.read(4)
        chunks = []
        while True:
            c = f.read(1)
            if c == b"\x00":
                break
            chunks.append(c)
        name = b"".join(chunks).decode("utf-8")
        num_points2d = struct.unpack("<Q", f.read(8))[0]
        f.seek(num_points2d * 24, 1)
        rot = qvec2rotmat(qvec)
        images[name] = -rot.T @ tvec
    return images


def load_colmap_models(path: Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    if not path.exists():
        return []
    models = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("images.bin"):
                try:
                    images = read_images_bin(zf.read(name))
                except Exception:
                    continue
                if images:
                    models.append((name, images))
    return models


def load_video_frames(video_zip: Path, video_id: str) -> tuple[list[int], dict[int, str]]:
    if not video_zip.exists():
        return [], {}
    nums = []
    names = {}
    prefix = f"{video_id}/imgs_3fps_360p/"
    with zipfile.ZipFile(video_zip) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix):
                continue
            n = frame_num(name)
            if n is not None:
                nums.append(n)
                names[n] = name
    nums = sorted(set(nums))
    return nums, names


def nearest_available(target: float, nums: list[int]) -> int:
    if not nums:
        raise ValueError("empty frame list")
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    cand = nums[lo]
    if lo > 0 and abs(nums[lo - 1] - target) <= abs(cand - target):
        cand = nums[lo - 1]
    return cand


def sample_frame_nums(path_frames: list[str], available: list[int], n: int = SAMPLE_FRAMES) -> list[int]:
    raw = [frame_num(x) for x in path_frames]
    raw_nums = [x for x in raw if x is not None]
    if not raw_nums or not available:
        return []
    start, end = min(raw_nums), max(raw_nums)
    if end < start:
        start, end = end, start
    if end - start + 1 < n:
        pad = (n - (end - start + 1) + 1) // 2
        start -= pad
        end += pad
    start = max(min(available), start)
    end = min(max(available), end)
    targets = np.linspace(start, end, n)
    chosen = [nearest_available(float(t), available) for t in targets]
    dedup = []
    seen = set()
    for x in chosen:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    if len(dedup) >= n:
        return dedup[:n]
    in_window = [x for x in available if start <= x <= end]
    for x in in_window:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
        if len(dedup) >= n:
            break
    return sorted(dedup)


def choose_model(models: list[tuple[str, dict[str, np.ndarray]]], frame_nums: list[int]) -> tuple[str, dict[str, np.ndarray]] | None:
    wanted = {frame_name(x) for x in frame_nums}
    best = None
    best_count = 0
    for model_name, images in models:
        count = sum(1 for w in wanted if w in images)
        if count > best_count:
            best = (model_name, images)
            best_count = count
    if best_count < 2:
        return None
    return best


def positions_for_frames(images: dict[str, np.ndarray], frame_nums: list[int]) -> tuple[list[int], list[np.ndarray]]:
    nums = []
    positions = []
    for n in frame_nums:
        name = frame_name(n)
        if name in images:
            nums.append(n)
            positions.append(images[name])
    return nums, positions


def path_metrics(positions: list[np.ndarray]) -> dict[str, float]:
    diffs = [float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]
    path_len = sum(diffs)
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    delta = positions[-1] - positions[0]
    dx, dz = float(delta[0]), float(delta[2])
    angle = math.degrees(math.atan2(abs(dz), abs(dx))) if abs(dx) + abs(dz) > 1e-8 else 0.0
    angle = min(angle, 90.0 - angle)
    return {
        "straight_colmap_units": straight,
        "path_colmap_units": path_len,
        "axis_angle_deg": angle,
        "path_to_straight_ratio": path_len / max(straight, 1e-6),
        "pose_frame_count": len(positions),
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
                if cand == correct:
                    continue
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


def motion_label(text: str, metrics: dict[str, float]) -> str:
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
    if metrics["straight_colmap_units"] < 0.8 or metrics["path_to_straight_ratio"] > 3.0:
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


def make_base(row: dict[str, Any], sampled_nums: list[int], pose_nums: list[int], model_name: str, metrics: dict[str, float]) -> dict[str, Any]:
    video_id = row["videoId"]
    sampled_names = [frame_name(n) for n in sampled_nums]
    return {
        "dataset": "roomtour3d_video_pose_qa_20260527",
        "source": "roomtour3d",
        "video_id": video_id,
        "path_id": row["path_id"],
        "input": {
            "media_type": "video_zip_32frame_sequence",
            "video_zip": str(VIDEO_DIR / f"{video_id}.zip"),
            "frame_rate_fps": 3,
            "sampled_frame_indices_32": sampled_nums,
            "sampled_frame_names_32": sampled_names,
            "original_path_frames": row.get("path", []),
            "optView": row.get("optView"),
        },
        "question_language": "en",
        "answer_language": "en",
        "answer_style": "mcq",
        "answer_format": "multiple_choice",
        "metadata": {
            "source_instruction": clean_text(row["instructions"][0]),
            "colmap_zip": str(COLMAP_DIR / f"{video_id}.zip"),
            "colmap_images_bin": model_name,
            "pose_frame_indices_used": pose_nums,
            "pose_metrics": metrics,
            "pose_scale_note": "COLMAP reconstruction scale; values are not guaranteed to be metric meters.",
        },
    }


def make_qas(
    row: dict[str, Any],
    sampled_nums: list[int],
    pose_nums: list[int],
    model_name: str,
    metrics: dict[str, float],
    rng: random.Random,
) -> list[dict[str, Any]]:
    base = make_base(row, sampled_nums, pose_nums, model_name, metrics)
    correct = (
        bin_index(metrics["straight_colmap_units"], STRAIGHT_BINS),
        bin_index(metrics["axis_angle_deg"], ANGLE_BINS),
        bin_index(metrics["path_colmap_units"], PATH_BINS),
    )
    options, answer = interval_options(correct, rng)
    qa1 = dict(base)
    qa1.update(
        {
            "qa_id": f"roomtour3d_{row['path_id']}_pose_interval_summary",
            "task_type": "roomtour3d_pose_interval_summary",
            "question": (
                "Which option best describes the rough camera trajectory in this indoor first-person video? "
                "Estimate straight-line displacement, the angle of that displacement to the nearest horizontal/vertical "
                "ground-plane axis, and total path length. Reply with only one letter."
            ),
            "options": options,
            "answer": answer,
            "correct_option": answer,
            "answer_text": next(o["text"] for o in options if o["key"] == answer),
            "gt": {
                "coarse_answer_text": next(o["text"] for o in options if o["key"] == answer),
                "exact_pose_values": metrics,
            },
        }
    )

    motion = motion_label(row["instructions"][0], metrics)
    options2, answer2 = motion_options(motion, rng)
    qa2 = dict(base)
    qa2.update(
        {
            "qa_id": f"roomtour3d_{row['path_id']}_coarse_video_motion",
            "task_type": "roomtour3d_coarse_video_motion",
            "question": "Which broad camera motion pattern is most consistent with this indoor first-person video? Reply with only one letter.",
            "options": options2,
            "answer": answer2,
            "correct_option": answer2,
            "answer_text": motion,
            "gt": {"motion_label": motion, "source_instruction": clean_text(row["instructions"][0]), "exact_pose_values": metrics},
        }
    )
    return [qa1, qa2]


def extract_preview_images(video_zip: Path, video_id: str, frame_nums: list[int], dst_dir: Path, path_id: str, count: int = 12) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not frame_nums:
        return []
    pick = [frame_nums[int(round(i))] for i in np.linspace(0, len(frame_nums) - 1, min(count, len(frame_nums)))]
    out = []
    with zipfile.ZipFile(video_zip) as zf:
        names = set(zf.namelist())
        for n in pick:
            inner = f"{video_id}/imgs_3fps_360p/{frame_name(n)}"
            if inner not in names:
                continue
            dst = dst_dir / f"{path_id}_{frame_name(n)}"
            if not dst.exists():
                with Image.open(io.BytesIO(zf.read(inner))) as im:
                    im = im.convert("RGB")
                    im.thumbnail((360, 240))
                    im.save(dst, quality=86)
            out.append(str(dst))
    return out


def write_preview_html(samples: list[dict[str, Any]], out: Path, summary: dict[str, Any]) -> None:
    cards = []
    for sample in samples:
        imgs = "".join(f"<img src='{html.escape(str(Path(p).relative_to(out)))}'>" for p in sample["thumbs"])
        qas = "".join(
            "<div class='qa'><b>{}</b><p>{}</p><ol>{}</ol><p>answer: {}</p></div>".format(
                html.escape(q["task_type"]),
                html.escape(q["question"]),
                "".join(f"<li>{html.escape(o['key'])}. {html.escape(o['text'])}</li>" for o in q["options"]),
                html.escape(q["answer"] + " - " + q["answer_text"]),
            )
            for q in sample["qas"]
        )
        m = sample["metrics"]
        cards.append(
            f"<section><h2>{html.escape(sample['path_id'])}</h2><div class='imgs'>{imgs}</div>"
            f"<p class='metric'>straight={m['straight_colmap_units']:.2f}, path={m['path_colmap_units']:.2f}, "
            f"angle={m['axis_angle_deg']:.1f}, pose_frames={m['pose_frame_count']}</p>"
            f"<p>{html.escape(sample['instruction'][:700])}</p>{qas}</section>"
        )
    text = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoomTour3D Video Pose QA Full Preview</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f5f6;margin:0;color:#111}}main{{max-width:1400px;margin:0 auto;padding:18px}}
section{{background:#fff;border:1px solid #d7dbe0;border-radius:6px;margin:14px 0;padding:12px}}h1{{font-size:24px}}h2{{font-size:18px}}
.imgs{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}}img{{width:100%;border:1px solid #ddd;background:#ddd}}.metric{{color:#555}}
.qa{{border-top:1px solid #eee;margin-top:10px;padding-top:10px}}li{{margin:4px 0}}@media(max-width:900px){{.imgs{{grid-template-columns:repeat(3,1fr)}}}}
</style></head><body><main><h1>RoomTour3D Video Pose QA Full Preview</h1>
<p>Full output uses 32 sampled frames per path when available. Distances are in COLMAP reconstruction units, not guaranteed meters.</p>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
{''.join(cards)}</main></body></html>"""
    (out / "preview.html").write_text(text, encoding="utf-8")


def main() -> None:
    rng = random.Random(20260527)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(TRAJ_JSON)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["videoId"]].append(row)

    qa_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    skipped = Counter()
    preview_samples = []

    for video_i, (video_id, video_rows) in enumerate(sorted(by_video.items())):
        video_zip = VIDEO_DIR / f"{video_id}.zip"
        colmap_zip = COLMAP_DIR / f"{video_id}.zip"
        if not video_zip.exists():
            skipped["missing_video_zip"] += len(video_rows)
            continue
        if not colmap_zip.exists():
            skipped["missing_colmap_zip"] += len(video_rows)
            continue
        available_nums, _ = load_video_frames(video_zip, video_id)
        if len(available_nums) < MIN_UNIQUE_FRAMES:
            skipped["too_few_video_frames"] += len(video_rows)
            continue
        models = load_colmap_models(colmap_zip)
        if not models:
            skipped["no_colmap_models"] += len(video_rows)
            continue
        for row in video_rows:
            sampled = sample_frame_nums(row.get("path", []), available_nums)
            if len(sampled) < MIN_UNIQUE_FRAMES:
                skipped["too_few_sampled_frames"] += 1
                continue
            chosen = choose_model(models, sampled)
            if chosen is None:
                skipped["insufficient_pose_overlap"] += 1
                continue
            model_name, images = chosen
            pose_nums, positions = positions_for_frames(images, sampled)
            if len(positions) < MIN_UNIQUE_FRAMES:
                skipped["too_few_pose_frames"] += 1
                continue
            metrics = path_metrics(positions)
            qas = make_qas(row, sampled, pose_nums, model_name, metrics, rng)
            qa_rows.extend(qas)
            window_rows.append(
                {
                    "source": "roomtour3d",
                    "video_id": video_id,
                    "path_id": row["path_id"],
                    "video_zip": str(video_zip),
                    "colmap_zip": str(colmap_zip),
                    "colmap_images_bin": model_name,
                    "sampled_frame_indices_32": sampled,
                    "pose_frame_indices_used": pose_nums,
                    "metrics": metrics,
                    "source_instruction": clean_text(row["instructions"][0]),
                }
            )
            if len(preview_samples) < 40 and rng.random() < 0.08:
                thumbs = extract_preview_images(video_zip, video_id, sampled, OUT / "preview_thumbs", row["path_id"])
                preview_samples.append(
                    {
                        "video_id": video_id,
                        "path_id": row["path_id"],
                        "instruction": clean_text(row["instructions"][0]),
                        "metrics": metrics,
                        "thumbs": thumbs,
                        "qas": qas,
                    }
                )
        if (video_i + 1) % 100 == 0:
            print(json.dumps({"processed_videos": video_i + 1, "qa_rows": len(qa_rows), "skipped": dict(skipped)}, ensure_ascii=False), flush=True)

    qa_jsonl = OUT / "roomtour3d_video_pose_qa.jsonl"
    windows_jsonl = OUT / "roomtour3d_video_pose_windows.jsonl"
    write_jsonl(qa_jsonl, qa_rows)
    write_jsonl(windows_jsonl, window_rows)
    pd.DataFrame(qa_rows).to_parquet(OUT / "roomtour3d_video_pose_qa.parquet", index=False)
    pd.DataFrame(window_rows).to_parquet(OUT / "roomtour3d_video_pose_windows.parquet", index=False)

    summary = {
        "trajectory_json": str(TRAJ_JSON),
        "video_dir": str(VIDEO_DIR),
        "colmap_dir": str(COLMAP_DIR),
        "source_rows": len(rows),
        "kept_windows": len(window_rows),
        "qa_rows": len(qa_rows),
        "unique_videos": len({r["video_id"] for r in window_rows}),
        "task_counts": dict(Counter(q["task_type"] for q in qa_rows)),
        "answer_counts": dict(Counter(q["answer"] for q in qa_rows)),
        "skipped": dict(skipped),
        "frame_policy": f"sample up to {SAMPLE_FRAMES} frames uniformly across each path's video-frame span; require at least {MIN_UNIQUE_FRAMES} sampled frames with COLMAP poses",
        "pose_scale_note": "COLMAP reconstruction scale; values are not guaranteed to be metric meters.",
        "outputs": {
            "qa_jsonl": str(qa_jsonl),
            "qa_parquet": str(OUT / "roomtour3d_video_pose_qa.parquet"),
            "windows_jsonl": str(windows_jsonl),
            "windows_parquet": str(OUT / "roomtour3d_video_pose_windows.parquet"),
            "preview_html": str(OUT / "preview.html"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview_html(preview_samples, OUT, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
