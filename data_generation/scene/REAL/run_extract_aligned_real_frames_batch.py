#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


EXTRACT_SCRIPT = Path("/path/to/workspace/SCENE/REAL/extract_aligned_real_frames.py")
SCANNET_MANIFEST = Path("/path/to/workspace/DATA/REAL_OFFICIAL/manifests/scannet_scene_ids.txt")
ARKIT_CSV = Path("/path/to/workspace/DATA/REAL_OFFICIAL/manifests/arkitscenes_vsi_subset.csv")
ARKIT_RAW_ROOT = Path("/path/to/workspace/DATA/REAL_OFFICIAL/arkitscenes/raw")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_scannet_scene_ids() -> list[str]:
    return [line.strip() for line in SCANNET_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_arkit_scene_ids(only_available: bool) -> list[str]:
    available_ids: set[str] = set()
    if only_available:
        available_ids = {path.parent.name for path in ARKIT_RAW_ROOT.glob("**/lowres_wide.traj")}

    scene_ids: list[str] = []
    with ARKIT_CSV.open("r", encoding="utf-8") as handle:
        header = next(handle, None)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            scene_id = line.split(",", 1)[0]
            if only_available and scene_id not in available_ids:
                continue
            scene_ids.append(scene_id)
    return scene_ids


def run_one(
    *,
    dataset: str,
    scene_id: str,
    output_root: Path,
    sample_every: int,
    max_items: int,
    max_residual_seconds: float,
) -> dict:
    scene_output_dir = output_root / dataset / scene_id
    metadata_path = scene_output_dir / "metadata.json"
    if metadata_path.exists():
        return {"dataset": dataset, "scene_id": scene_id, "status": "skipped_existing", "metadata_path": str(metadata_path)}

    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--dataset",
        dataset,
        "--scene-id",
        scene_id,
        "--output-dir",
        str(output_root),
        "--sample-every",
        str(sample_every),
        "--max-items",
        str(max_items),
    ]
    if dataset == "arkitscenes":
        cmd.extend(["--max-residual-seconds", str(max_residual_seconds)])

    started_at = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started_at
    if proc.returncode == 0:
        return {
            "dataset": dataset,
            "scene_id": scene_id,
            "status": "ok",
            "elapsed_seconds": elapsed,
            "metadata_path": proc.stdout.strip(),
        }

    return {
        "dataset": dataset,
        "scene_id": scene_id,
        "status": "error",
        "elapsed_seconds": elapsed,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract aligned real-video frames with official pose/intrinsics.")
    parser.add_argument("--output-root", type=Path, default=Path("/path/to/workspace/SCENEOUTPUT/REAL/aligned_frames"))
    parser.add_argument("--datasets", type=str, default="scannet,arkitscenes")
    parser.add_argument("--sample-every-scannet", type=int, default=100)
    parser.add_argument("--sample-every-arkit", type=int, default=50)
    parser.add_argument("--max-items-scannet", type=int, default=8)
    parser.add_argument("--max-items-arkit", type=int, default=8)
    parser.add_argument("--max-residual-seconds", type=float, default=0.05)
    parser.add_argument("--only-available-arkit", action="store_true", default=True)
    parser.add_argument("--log-dir", type=Path, default=Path("/path/to/workspace/SCENEOUTPUT/REAL/aligned_frames_logs"))
    args = parser.parse_args()

    selected = {item.strip() for item in args.datasets.split(",") if item.strip()}
    run_log = args.log_dir / "batch_results.jsonl"
    summary_path = args.log_dir / "summary.json"
    ensure_dir(args.log_dir)

    totals = {"ok": 0, "error": 0, "skipped_existing": 0}
    dataset_plan: list[tuple[str, list[str], int, int]] = []

    if "scannet" in selected:
        dataset_plan.append(("scannet", load_scannet_scene_ids(), args.sample_every_scannet, args.max_items_scannet))
    if "arkitscenes" in selected:
        dataset_plan.append(
            ("arkitscenes", load_arkit_scene_ids(args.only_available_arkit), args.sample_every_arkit, args.max_items_arkit)
        )

    started_at = time.time()
    for dataset, scene_ids, sample_every, max_items in dataset_plan:
        print(f"[batch] dataset={dataset} scenes={len(scene_ids)} sample_every={sample_every} max_items={max_items}", flush=True)
        for idx, scene_id in enumerate(scene_ids, start=1):
            result = run_one(
                dataset=dataset,
                scene_id=scene_id,
                output_root=args.output_root,
                sample_every=sample_every,
                max_items=max_items,
                max_residual_seconds=args.max_residual_seconds,
            )
            totals[result["status"]] += 1
            result["dataset_index"] = idx
            result["dataset_total"] = len(scene_ids)
            append_jsonl(run_log, result)
            print(
                f"[batch] {dataset} {idx}/{len(scene_ids)} scene={scene_id} status={result['status']}",
                flush=True,
            )

    summary = {
        "started_at_epoch": started_at,
        "finished_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "totals": totals,
        "output_root": str(args.output_root),
        "log_path": str(run_log),
        "datasets": [item[0] for item in dataset_plan],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
