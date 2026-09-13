#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

from scene_catalog import all_scene_names, normalize_scene_tokens
from scene_worker import process_scene


DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/PROC/output")


def _worker_entry(payload: Dict[str, object]) -> Dict[str, object]:
    args = SimpleNamespace(**payload["args"])
    return process_scene(args, str(payload["scene"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel ProcTHOR scene-exhaustive precomputation runner.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--probe-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--yaws", type=str, default="0,30,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--horizons", type=str, default="0")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-states-per-position", type=int, default=2)
    parser.add_argument("--max-total-states", type=int, default=40)
    parser.add_argument("--max-position-groups", type=int, default=20)
    parser.add_argument("--max-proposals-per-subcat", type=int, default=5)
    parser.add_argument("--min-position-separation", type=float, default=0.75)
    parser.add_argument("--global-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-min-position-separation", type=float, default=0.35)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--disable-state-dedup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = all_scene_names() if args.scenes is None else normalize_scene_tokens(args.scenes.split(","))
    args.output_root.mkdir(parents=True, exist_ok=True)

    payload_args = {
        "output_root": args.output_root,
        "width": args.width,
        "height": args.height,
        "grid_size": args.grid_size,
        "field_of_view": args.field_of_view,
        "visibility_distance": args.visibility_distance,
        "probe_yaws": args.probe_yaws,
        "yaws": args.yaws,
        "horizons": args.horizons,
        "max_positions": args.max_positions,
        "max_states_per_position": args.max_states_per_position,
        "max_total_states": args.max_total_states,
        "max_position_groups": args.max_position_groups,
        "max_proposals_per_subcat": args.max_proposals_per_subcat,
        "min_position_separation": args.min_position_separation,
        "global_position_groups": args.global_position_groups,
        "hotspot_position_groups": args.hotspot_position_groups,
        "hotspot_min_position_separation": args.hotspot_min_position_separation,
        "use_cloud_rendering": args.use_cloud_rendering,
        "disable_state_dedup": args.disable_state_dedup,
    }

    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as pool:
        future_to_scene = {
            pool.submit(_worker_entry, {"args": payload_args, "scene": scene}): scene
            for scene in scenes
        }
        for future in as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "scene": scene,
                    "status": "failed",
                    "error": str(exc),
                    "kept_states": 0,
                    "proposal_count": 0,
                }
            results.append(result)
            status = str(result.get("status", "ok"))
            print(
                f"[{status.upper()}] {result['scene']} states={result.get('kept_states', 0)} "
                f"proposals={result.get('proposal_count', 0)}",
                flush=True,
            )

    results.sort(key=lambda item: str(item["scene"]))
    overview = {
        "scenes": len(results),
        "finished_scenes": [item["scene"] for item in results if item.get("status", "ok") == "ok"],
        "failed_scenes": [item["scene"] for item in results if item.get("status", "ok") != "ok"],
        "total_kept_states": sum(int(item["kept_states"]) for item in results),
        "total_proposals": sum(int(item["proposal_count"]) for item in results),
        "scene_summaries": results,
    }
    overview_path = args.output_root / "parallel_overview.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
