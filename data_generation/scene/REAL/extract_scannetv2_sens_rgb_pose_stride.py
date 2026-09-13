#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


SENS_ROOT = Path("/path/to/workspace/Scannet-V2数据集/split_scans/sens")
OUTPUT_ROOT = Path("/path/to/workspace/Scannet-V2数据集/extracted_rgb_pose_stride10")

COLOR_COMPRESSION = {
    -1: "unknown",
    0: "raw",
    1: "png",
    2: "jpeg",
}
DEPTH_COMPRESSION = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


def read_exact(handle, nbytes: int) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"expected {nbytes} bytes, got {len(data)}")
    return data


def read_u32(handle) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_i32(handle) -> int:
    return struct.unpack("<i", read_exact(handle, 4))[0]


def read_u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_f32(handle) -> float:
    return struct.unpack("<f", read_exact(handle, 4))[0]


def read_mat4(handle) -> np.ndarray:
    values = struct.unpack("<16f", read_exact(handle, 16 * 4))
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def write_mat(path: Path, mat: np.ndarray) -> None:
    np.savetxt(path, mat, fmt="%.9g")


def sensor_header(handle) -> dict[str, Any]:
    version = read_u32(handle)
    name_len = read_u64(handle)
    sensor_name = read_exact(handle, name_len).decode("utf-8", errors="replace")
    intrinsic_color = read_mat4(handle)
    extrinsic_color = read_mat4(handle)
    intrinsic_depth = read_mat4(handle)
    extrinsic_depth = read_mat4(handle)
    color_compression_type = read_i32(handle)
    depth_compression_type = read_i32(handle)
    color_width = read_u32(handle)
    color_height = read_u32(handle)
    depth_width = read_u32(handle)
    depth_height = read_u32(handle)
    depth_shift = read_f32(handle)
    num_frames = read_u64(handle)
    return {
        "version": int(version),
        "sensor_name": sensor_name,
        "intrinsic_color": intrinsic_color,
        "extrinsic_color": extrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "extrinsic_depth": extrinsic_depth,
        "color_compression_type": int(color_compression_type),
        "color_compression": COLOR_COMPRESSION.get(int(color_compression_type), f"unknown_{color_compression_type}"),
        "depth_compression_type": int(depth_compression_type),
        "depth_compression": DEPTH_COMPRESSION.get(int(depth_compression_type), f"unknown_{depth_compression_type}"),
        "color_width": int(color_width),
        "color_height": int(color_height),
        "depth_width": int(depth_width),
        "depth_height": int(depth_height),
        "depth_shift": float(depth_shift),
        "num_frames": int(num_frames),
    }


def image_suffix(color_compression_type: int) -> str:
    if color_compression_type == 2:
        return ".jpg"
    if color_compression_type == 1:
        return ".png"
    return ".bin"


def extract_one(sens_path: Path, output_root: Path, stride: int, overwrite: bool = False) -> dict[str, Any]:
    scene = sens_path.stem
    final_dir = output_root / scene
    done_path = final_dir / ".done"
    manifest_path = final_dir / "manifest.json"
    if done_path.exists() and manifest_path.exists() and not overwrite:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "scene": scene,
            "status": "skipped",
            "selected_frames": len(payload.get("frames", [])),
            "num_frames": payload.get("num_frames"),
            "output_dir": str(final_dir),
        }

    tmp_dir = output_root / f".{scene}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "color").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "pose").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "intrinsic").mkdir(parents=True, exist_ok=True)

    started = time.time()
    frames: list[dict[str, Any]] = []
    with sens_path.open("rb") as handle:
        header = sensor_header(handle)
        suffix = image_suffix(header["color_compression_type"])
        write_mat(tmp_dir / "intrinsic" / "intrinsic_color.txt", header["intrinsic_color"])
        write_mat(tmp_dir / "intrinsic" / "extrinsic_color.txt", header["extrinsic_color"])
        write_mat(tmp_dir / "intrinsic" / "intrinsic_depth.txt", header["intrinsic_depth"])
        write_mat(tmp_dir / "intrinsic" / "extrinsic_depth.txt", header["extrinsic_depth"])

        for frame_idx in range(header["num_frames"]):
            camera_to_world = read_mat4(handle)
            timestamp_color = read_u64(handle)
            timestamp_depth = read_u64(handle)
            color_size = read_u64(handle)
            depth_size = read_u64(handle)
            selected = frame_idx % stride == 0 or frame_idx == header["num_frames"] - 1
            if selected:
                color_data = read_exact(handle, color_size)
                color_name = f"{frame_idx:06d}{suffix}"
                pose_name = f"{frame_idx:06d}.txt"
                color_path = tmp_dir / "color" / color_name
                pose_path = tmp_dir / "pose" / pose_name
                color_path.write_bytes(color_data)
                write_mat(pose_path, camera_to_world)
                frames.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp_color": int(timestamp_color),
                        "timestamp_depth": int(timestamp_depth),
                        "color_path": str(color_path.relative_to(tmp_dir)),
                        "pose_path": str(pose_path.relative_to(tmp_dir)),
                    }
                )
                handle.seek(depth_size, 1)
            else:
                handle.seek(color_size + depth_size, 1)

    manifest = {
        "scene": scene,
        "sens_path": str(sens_path),
        "stride": int(stride),
        "num_frames": int(header["num_frames"]),
        "selected_frames": len(frames),
        "version": header["version"],
        "sensor_name": header["sensor_name"],
        "color_compression": header["color_compression"],
        "depth_compression": header["depth_compression"],
        "color_width": header["color_width"],
        "color_height": header["color_height"],
        "depth_width": header["depth_width"],
        "depth_height": header["depth_height"],
        "depth_shift": header["depth_shift"],
        "intrinsic_color_path": "intrinsic/intrinsic_color.txt",
        "extrinsic_color_path": "intrinsic/extrinsic_color.txt",
        "intrinsic_depth_path": "intrinsic/intrinsic_depth.txt",
        "extrinsic_depth_path": "intrinsic/extrinsic_depth.txt",
        "pose_convention": "ScanNet official camera_to_world pose from .sens",
        "frames": frames,
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (tmp_dir / ".done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    elapsed = time.time() - started
    return {
        "scene": scene,
        "status": "done",
        "selected_frames": len(frames),
        "num_frames": header["num_frames"],
        "elapsed_seconds": round(elapsed, 3),
        "output_dir": str(final_dir),
    }


def list_sens(root: Path) -> list[Path]:
    return sorted(root.glob("scene*/scene*.sens"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast ScanNet V2 .sens RGB+pose extractor with frame stride.")
    parser.add_argument("--sens-root", type=Path, default=SENS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "progress.jsonl"
    error_path = args.output_root / "errors.jsonl"

    sens_paths = list_sens(args.sens_root)
    if args.max_scenes > 0:
        sens_paths = sens_paths[: args.max_scenes]
    print(
        json.dumps(
            {
                "sens_root": str(args.sens_root),
                "output_root": str(args.output_root),
                "stride": args.stride,
                "workers": args.workers,
                "scenes": len(sens_paths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    done = skipped = failed = total_selected = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, path, args.output_root, args.stride, args.overwrite) for path in sens_paths]
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                if result["status"] == "done":
                    done += 1
                else:
                    skipped += 1
                total_selected += int(result.get("selected_frames") or 0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"[{idx}/{len(futures)}] {result['status']} {result['scene']} "
                    f"selected={result.get('selected_frames')} total_selected={total_selected}",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                payload = {"status": "failed", "error": repr(exc)}
                with error_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(f"[{idx}/{len(futures)}] failed {repr(exc)}", flush=True)

    summary = {
        "sens_root": str(args.sens_root),
        "output_root": str(args.output_root),
        "stride": args.stride,
        "workers": args.workers,
        "total_scenes": len(sens_paths),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "total_selected_frames": total_selected,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
