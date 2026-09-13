#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
OUTPUT_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10")
RAW_DIRNAME = "raw_minimal"
EXTRACTED_DIRNAME = "extracted_rgb_pose_stride10"


def retry_eagain(fn, *, attempts: int = 10, delay_s: float = 1.0):
    for attempt in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt + 1 >= attempts:
                raise
            time.sleep(delay_s * (attempt + 1))


def ensure_dir(path: Path) -> None:
    retry_eagain(lambda: path.mkdir(parents=True, exist_ok=True))


def exists_with_retry(path: Path) -> bool:
    return bool(retry_eagain(lambda: path.exists()))


def write_text_with_retry(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    retry_eagain(lambda: path.write_text(text, encoding="utf-8"))


def append_jsonl_with_retry(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)

    def _write() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    retry_eagain(_write)


def curl_download(url: str, dst: Path, timeout: int = 7200) -> None:
    ensure_dir(dst.parent)
    if exists_with_retry(dst) and retry_eagain(lambda: dst.stat().st_size) > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    env = os.environ.copy()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env[key] = ""
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-C",
        "-",
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True, env=env, timeout=timeout)
    retry_eagain(lambda: tmp.rename(dst))


def download_metadata(root: Path) -> Path:
    metadata_path = root / RAW_DIRNAME / "metadata.csv"
    curl_download(f"{BASE_URL}/raw/metadata.csv", metadata_path, timeout=120)
    return metadata_path


def load_metadata(path: Path, max_scenes: int = 0, skip_scenes: int = 0) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("video_id") and row.get("fold") in {"Training", "Validation"}]
    rows.sort(key=lambda row: (row["fold"], row["video_id"]))
    if skip_scenes > 0:
        rows = rows[skip_scenes:]
    if max_scenes > 0:
        rows = rows[:max_scenes]
    return rows


def parse_traj(path: Path) -> list[dict[str, Any]]:
    records = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        if len(values) != 7:
            continue
        records.append({"pose_index": idx, "timestamp": values[0], "raw_values": values[1:], "raw_line": line.strip()})
    if not records:
        raise RuntimeError(f"empty_traj:{path}")
    return records


def parse_pincam(path: Path) -> dict[str, Any]:
    values = [float(x) for x in path.read_text(encoding="utf-8").split()]
    if len(values) < 6:
        raise RuntimeError(f"bad_pincam:{path}")
    width, height, fx, fy, cx, cy = values[:6]
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return {
        "width": int(round(width)),
        "height": int(round(height)),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "intrinsic_4x4": intrinsic,
    }


def axis_angle_to_matrix(axis_angle: list[float]) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rot.astype(np.float64)


def arkit_pose_matrix(raw_values: list[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = axis_angle_to_matrix(raw_values[:3])
    mat[:3, 3] = np.asarray(raw_values[3:6], dtype=np.float64)
    return mat


def nearest_index(sorted_values: list[float], target: float) -> int:
    if not sorted_values:
        raise ValueError("empty sorted_values")
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo <= 0:
        return 0
    if lo >= len(sorted_values):
        return len(sorted_values) - 1
    return lo - 1 if abs(sorted_values[lo - 1] - target) <= abs(sorted_values[lo] - target) else lo


def safe_extract_zip(zip_path: Path, dst: Path) -> None:
    if exists_with_retry(dst) and retry_eagain(lambda: any(dst.iterdir())):
        return
    tmp = dst.with_name(dst.name + ".tmp")
    for attempt in range(8):
        try:
            if exists_with_retry(tmp):
                shutil.rmtree(tmp)
            ensure_dir(tmp)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            if exists_with_retry(dst):
                shutil.rmtree(dst)
            retry_eagain(lambda: tmp.rename(dst))
            return
        except OSError as exc:
            if exc.errno != errno.EAGAIN or attempt == 7:
                raise
            time.sleep(1.5 * (attempt + 1))


def download_scene_assets(root: Path, split: str, video_id: str) -> dict[str, Path]:
    scene_raw = root / RAW_DIRNAME / split / video_id
    ensure_dir(scene_raw)
    prefix = f"{BASE_URL}/raw/{split}/{video_id}"
    mov = scene_raw / f"{video_id}.mov"
    traj = scene_raw / "lowres_wide.traj"
    intr_zip = scene_raw / "lowres_wide_intrinsics.zip"
    curl_download(f"{prefix}/{video_id}.mov", mov)
    curl_download(f"{prefix}/lowres_wide.traj", traj, timeout=300)
    curl_download(f"{prefix}/lowres_wide_intrinsics.zip", intr_zip, timeout=600)
    intr_dir = scene_raw / "lowres_wide_intrinsics"
    safe_extract_zip(intr_zip, intr_dir)
    return {"mov": mov, "traj": traj, "intr_dir": intr_dir, "raw_dir": scene_raw}


def extract_scene(root: Path, row: dict[str, str], stride: int, keep_video: bool, overwrite: bool = False) -> dict[str, Any]:
    video_id = str(row["video_id"])
    split = str(row["fold"])
    scene_out = root / EXTRACTED_DIRNAME / split / video_id
    done_path = scene_out / ".done"
    manifest_path = scene_out / "manifest.json"
    if exists_with_retry(done_path) and exists_with_retry(manifest_path) and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"status": "skipped", "video_id": video_id, "split": split, "selected_frames": len(manifest.get("frames", []))}

    started = time.time()
    assets = download_scene_assets(root, split, video_id)
    traj_records = parse_traj(assets["traj"])
    traj_times = [float(item["timestamp"]) for item in traj_records]
    pincams = sorted(assets["intr_dir"].rglob("*.pincam"), key=lambda p: float(p.stem.split("_")[-1]))
    if not pincams:
        raise RuntimeError(f"missing_pincams:{assets['intr_dir']}")
    intr_times = [float(path.stem.split("_")[-1]) for path in pincams]
    official_start = intr_times[0]
    official_end = intr_times[-1]
    official_span = official_end - official_start
    if official_span <= 0:
        raise RuntimeError(f"bad_intrinsic_times:{video_id}")

    tmp_out = root / EXTRACTED_DIRNAME / split / f".{video_id}.tmp"
    if exists_with_retry(tmp_out):
        shutil.rmtree(tmp_out)
    ensure_dir(tmp_out / "color")
    ensure_dir(tmp_out / "pose")
    ensure_dir(tmp_out / "intrinsic")

    cap = cv2.VideoCapture(str(assets["mov"]))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_mov:{assets['mov']}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        raise RuntimeError(f"bad_frame_count:{assets['mov']}")

    selected_indices = set(range(0, frame_count, stride))
    selected_indices.add(frame_count - 1)
    frames = []
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx in selected_indices:
                ts = official_start + frame_idx * official_span / max(frame_count - 1, 1)
                pose_idx = nearest_index(traj_times, ts)
                intr_idx = nearest_index(intr_times, ts)
                pose = arkit_pose_matrix(traj_records[pose_idx]["raw_values"])
                intr = parse_pincam(pincams[intr_idx])

                color_rel = Path("color") / f"{frame_idx:06d}.jpg"
                pose_rel = Path("pose") / f"{frame_idx:06d}.txt"
                intr_rel = Path("intrinsic") / f"{frame_idx:06d}.txt"
                cv2.imwrite(str(tmp_out / color_rel), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                np.savetxt(tmp_out / pose_rel, pose, fmt="%.9g")
                np.savetxt(tmp_out / intr_rel, intr["intrinsic_4x4"], fmt="%.9g")
                frames.append(
                    {
                        "frame_index": int(frame_idx),
                        "video_timestamp_est": float(ts),
                        "pose_index": int(traj_records[pose_idx]["pose_index"]),
                        "pose_timestamp": float(traj_records[pose_idx]["timestamp"]),
                        "intrinsics_timestamp": float(intr_times[intr_idx]),
                        "color_path": str(color_rel),
                        "pose_path": str(pose_rel),
                        "intrinsic_path": str(intr_rel),
                    }
                )
            frame_idx += 1
    finally:
        cap.release()

    manifest = {
        "dataset": "arkitscenes",
        "video_id": video_id,
        "split": split,
        "stride": int(stride),
        "frame_count": int(frame_count),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "selected_frames": len(frames),
        "raw_assets": {key: str(value) for key, value in assets.items()},
        "pose_convention": "ARKitScenes lowres_wide.traj axis-angle+translation matrix as originally stored; previous REAL scripts treated it directly as w2c",
        "frames": frames,
    }
    ensure_dir(scene_out.parent)
    write_text_with_retry(tmp_out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text_with_retry(tmp_out / ".done", time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    if exists_with_retry(scene_out):
        shutil.rmtree(scene_out)
    retry_eagain(lambda: tmp_out.rename(scene_out))
    if not keep_video:
        assets["mov"].unlink(missing_ok=True)
    return {
        "status": "done",
        "video_id": video_id,
        "split": split,
        "selected_frames": len(frames),
        "frame_count": int(frame_count),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download minimal ARKitScenes raw assets and extract RGB+pose+intrinsics.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--skip-scenes", type=int, default=0)
    parser.add_argument("--keep-video", action="store_true", default=True)
    parser.add_argument("--delete-video-after-extract", dest="keep_video", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    ensure_dir(args.output_root)
    metadata_path = download_metadata(args.output_root)
    rows = load_metadata(metadata_path, args.max_scenes, args.skip_scenes)
    log_path = args.output_root / "progress.jsonl"
    err_path = args.output_root / "errors.jsonl"
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "stride": args.stride,
                "workers": args.workers,
                "scenes": len(rows),
                "skip_scenes": args.skip_scenes,
                "keep_video": args.keep_video,
                "proxy_disabled": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    done = skipped = failed = total_selected = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_scene, args.output_root, row, args.stride, args.keep_video, args.overwrite) for row in rows]
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                if result["status"] == "done":
                    done += 1
                else:
                    skipped += 1
                total_selected += int(result.get("selected_frames") or 0)
                append_jsonl_with_retry(log_path, result)
                print(
                    f"[{idx}/{len(futures)}] {result['status']} {result['split']}/{result['video_id']} "
                    f"selected={result.get('selected_frames')} total_selected={total_selected}",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                payload = {"status": "failed", "error": repr(exc)}
                append_jsonl_with_retry(err_path, payload)
                print(f"[{idx}/{len(futures)}] failed {repr(exc)}", flush=True)

    summary = {
        "output_root": str(args.output_root),
        "stride": args.stride,
        "workers": args.workers,
        "skip_scenes": args.skip_scenes,
        "total_scenes": len(rows),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "total_selected_frames": total_selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_text_with_retry(args.output_root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
