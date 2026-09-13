#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_3rscan_three_tasks as threed  # noqa: E402
import generate_multiscan_pose_rule_preview as multiscan  # noqa: E402
import generate_scannetpp_three_tasks as scannetpp  # noqa: E402


PILOT_ROOT = Path("/path/to/workspace/DATA")
OUT_ROOT = Path("/path/to/workspace/DATA")
SCANNETV2_ROOT = Path("/path/to/workspace/Scannet-V2数据集/extracted_rgb_pose_stride10")
ARKIT_ROOT = Path("/path/to/workspace/ARKitScenes_QA_stride10/extracted_rgb_pose_stride10/Training")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_indices(values: list[int], frame_count: int) -> list[int]:
    if len(values) <= frame_count:
        return values
    positions = np.linspace(0, len(values) - 1, frame_count)
    out = []
    seen = set()
    for pos in positions:
        value = values[int(round(float(pos)))]
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def segment_sample_count(segment: dict) -> int:
    return 5 if segment.get("segment_type") == "长" else 3


class FrameReader:
    def __init__(self) -> None:
        self.video_caps: dict[Path, cv2.VideoCapture] = {}
        self.threed_cache: dict[str, object] = {}

    def close(self) -> None:
        for cap in self.video_caps.values():
            cap.release()

    def video_frame(self, path: Path, frame_index: int) -> np.ndarray | None:
        cap = self.video_caps.get(path)
        if cap is None:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            self.video_caps[path] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        return frame if ok else None

    def read(self, dataset: str, scene: str, frame_index: int) -> np.ndarray | None:
        if dataset == "scannetv2":
            color_dir = SCANNETV2_ROOT / scene / "color"
            path = color_dir / f"{int(frame_index)}.jpg"
            if not path.exists():
                path = color_dir / f"{int(frame_index):06d}.jpg"
            return cv2.imread(str(path))
        if dataset == "arkit":
            return cv2.imread(str(ARKIT_ROOT / scene / "color" / f"{int(frame_index):06d}.jpg"))
        if dataset == "scannetpp":
            return self.video_frame(scannetpp.DATA_ROOT / scene / "iphone" / "rgb.mkv", int(frame_index))
        if dataset == "multiscan":
            zip_path = multiscan.SCAN_ROOT / f"{scene}.zip"
            mp4 = multiscan.extract_mp4(zip_path, Path("/path/to/workspace/MultiScan_preview_cache"))
            if mp4 is None:
                return None
            return self.video_frame(mp4, int(frame_index))
        if dataset == "3rscan":
            cached = self.threed_cache.get(scene)
            if cached is None:
                scene_dir = threed.EXTRACTED_SAMPLE_ROOT / scene
                if scene_dir.exists():
                    cached = threed.load_extracted_scene(scene_dir)
                if cached is None:
                    cached = threed.load_zipped_scene(threed.SEQUENCE_ROOT / scene)
                if cached is None:
                    return None
                self.threed_cache[scene] = cached
            return threed.read_frame_from_scene(cached, int(frame_index))
        return None


def put_text(img: np.ndarray, text: str, y: int, scale: float = 0.55) -> None:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def make_segment_sheet(segment: dict, frames: list[np.ndarray], frame_ids: list[int], out_path: Path) -> bool:
    if not frames:
        return False
    tile_w, tile_h = 320, 240
    tiles = []
    for frame, frame_id in zip(frames, frame_ids):
        tile = cv2.resize(frame, (tile_w, tile_h))
        cv2.rectangle(tile, (0, 0), (110, 28), (255, 255, 255), -1)
        cv2.putText(tile, f"f={frame_id}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
        tiles.append(tile)

    cols = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    blank = np.zeros_like(tiles[0])
    grid_rows = []
    for row in range(rows):
        row_tiles = tiles[row * cols : (row + 1) * cols]
        while len(row_tiles) < cols:
            row_tiles.append(blank.copy())
        grid_rows.append(np.concatenate(row_tiles, axis=1))
    grid = np.concatenate(grid_rows, axis=0)
    header_h = 86
    header = np.full((header_h, grid.shape[1], 3), 245, dtype=np.uint8)
    label = segment.get("action_label") or segment.get("label") or segment.get("candidate_label")
    value = segment.get("cumulative_value")
    unit = segment.get("unit")
    value_text = "n/a" if value is None else f"{float(value):.2f}{unit}"
    title = f"{segment['segment_id']} | {segment['segment_type']} | {label} | {value_text}"
    metrics = f"scene={segment['scene']} frames={segment['start_frame_index']}->{segment['end_frame_index']} purity={segment['purity']:.3f} mono={segment['monotonicity']:.3f}"
    cv2.putText(header, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 180), 2, cv2.LINE_AA)
    cv2.putText(header, metrics, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 20, 20), 1, cv2.LINE_AA)
    out = np.concatenate([header, grid], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), out)


def make_overview(dataset: str, rows: list[dict], out_dir: Path) -> None:
    panels = []
    for row in rows:
        img = cv2.imread(row["visual_path"])
        if img is None:
            continue
        thumb = cv2.resize(img, (480, 300))
        panels.append(thumb)
    if not panels:
        return
    cols = 2
    blank = np.zeros_like(panels[0])
    lines = []
    for i in range(0, len(panels), cols):
        row = panels[i : i + cols]
        while len(row) < cols:
            row.append(blank.copy())
        lines.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(out_dir / f"{dataset}_pure_segments_overview.png"), np.concatenate(lines, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize pure 0.33s motion segments as per-segment contact sheets.")
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    reader = FrameReader()
    summary: dict[str, dict] = {}
    try:
        for dataset in ["scannetv2", "scannetpp", "multiscan", "3rscan", "arkit"]:
            segments = [row for row in read_jsonl(args.pilot_root / dataset / "segments.jsonl") if row.get("is_pure")]
            out_dir = args.output_root / dataset / "pure_segment_visuals"
            made = []
            for segment in segments:
                frame_ids = sample_indices(
                    [int(x) for x in segment.get("resampled_frame_indices") or segment.get("frame_indices") or []],
                    segment_sample_count(segment),
                )
                frames = []
                kept_ids = []
                for frame_id in frame_ids:
                    frame = reader.read(dataset, segment["scene"], frame_id)
                    if frame is None:
                        continue
                    frames.append(frame)
                    kept_ids.append(frame_id)
                out_path = out_dir / f"{segment['segment_id']}.png"
                if make_segment_sheet(segment, frames, kept_ids, out_path):
                    made.append(
                        {
                            "segment_id": segment["segment_id"],
                            "scene": segment["scene"],
                            "segment_type": segment["segment_type"],
                            "action_label": segment.get("action_label"),
                            "cumulative_value": segment.get("cumulative_value"),
                            "unit": segment.get("unit"),
                            "purity": segment.get("purity"),
                            "monotonicity": segment.get("monotonicity"),
                            "visual_path": str(out_path),
                        }
                    )
            make_overview(dataset, made, out_dir)
            write_json(out_dir / "visual_summary.json", made)
            summary[dataset] = {
                "pure_segments": len(segments),
                "visualized": len(made),
                "overview": str(out_dir / f"{dataset}_pure_segments_overview.png") if made else None,
                "visual_summary": str(out_dir / "visual_summary.json"),
            }
            print(f"[visual] {dataset}: {len(made)}/{len(segments)}", flush=True)
    finally:
        reader.close()
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
