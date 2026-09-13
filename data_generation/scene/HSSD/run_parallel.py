#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from scene_catalog import all_scene_names, normalize_scene_tokens


DEFAULT_OUTPUT_ROOT = Path("/path/to/workspace/SCENEOUTPUT/HSSD/output")
SCENE_ROOT = Path("/path/to/workspace/SCENE/HSSD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel HSSD scene-exhaustive precomputation runner.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=str, default="/path/to/workspace/habitat_data")
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--sensor-height", type=float, default=1.6)
    parser.add_argument("--agent-radius", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefilter-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--probe-yaws", type=str, default="0,90,180,270")
    parser.add_argument("--yaws", type=str, default="0,30,60,90,120,150,180,210,240,270,300,330")
    parser.add_argument("--horizons", type=str, default="0")
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-states-per-position", type=int, default=2)
    parser.add_argument("--max-total-states", type=int, default=40)
    parser.add_argument("--max-position-groups", type=int, default=20)
    parser.add_argument("--max-probe-positions", type=int, default=160)
    parser.add_argument("--prefilter-candidate-positions", type=int, default=120)
    parser.add_argument("--prefilter-top-positions", type=int, default=50)
    parser.add_argument("--prefilter-candidate-separation", type=float, default=1.0)
    parser.add_argument("--prefilter-top-separation", type=float, default=0.75)
    parser.add_argument("--max-proposals-per-subcat", type=int, default=5)
    parser.add_argument("--min-position-separation", type=float, default=0.75)
    parser.add_argument("--probe-position-separation", type=float, default=0.5)
    parser.add_argument("--global-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-position-groups", type=int, default=10)
    parser.add_argument("--hotspot-min-position-separation", type=float, default=0.35)
    parser.add_argument("--prefilter-restart-interval", type=int, default=20)
    parser.add_argument("--probe-restart-interval", type=int, default=10)
    parser.add_argument("--full-restart-interval", type=int, default=2)
    parser.add_argument("--scene-retries", type=int, default=2)
    parser.add_argument("--disable-position-prefilter", action="store_true")
    parser.add_argument("--disable-state-dedup", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-limit:]


def _scene_is_complete(scene_dir: Path) -> bool:
    summary_path = scene_dir / "scene_pipeline_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(summary.get("status", "")) == "ok"


def _run_one_scene(args: argparse.Namespace, scene: str) -> Dict[str, object]:
    scene_dir = args.output_root / scene
    stdout_log = scene_dir / "scene_worker.stdout.log"
    stderr_log = scene_dir / "scene_worker.stderr.log"

    if args.reset_output and scene_dir.exists():
        subprocess.run(["rm", "-rf", str(scene_dir)], check=False)
    scene_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and _scene_is_complete(scene_dir):
        result = json.loads((scene_dir / "scene_pipeline_summary.json").read_text(encoding="utf-8"))
        result["status"] = "skipped_existing"
        return result

    cmd = [
        str(SCENE_ROOT / "run_scene_worker.sh"),
        "--output-root",
        str(args.output_root),
        "--dataset-root",
        str(args.dataset_root),
        "--scenes",
        scene,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--field-of-view",
        str(args.field_of_view),
        "--sensor-height",
        str(args.sensor_height),
        "--agent-radius",
        str(args.agent_radius),
        "--seed",
        str(args.seed),
        "--prefilter-yaws",
        str(args.prefilter_yaws),
        "--probe-yaws",
        str(args.probe_yaws),
        "--yaws",
        str(args.yaws),
        "--horizons",
        str(args.horizons),
        "--max-states-per-position",
        str(args.max_states_per_position),
        "--max-total-states",
        str(args.max_total_states),
        "--max-position-groups",
        str(args.max_position_groups),
        "--max-probe-positions",
        str(args.max_probe_positions),
        "--prefilter-candidate-positions",
        str(args.prefilter_candidate_positions),
        "--prefilter-top-positions",
        str(args.prefilter_top_positions),
        "--prefilter-candidate-separation",
        str(args.prefilter_candidate_separation),
        "--prefilter-top-separation",
        str(args.prefilter_top_separation),
        "--prefilter-restart-interval",
        str(args.prefilter_restart_interval),
        "--max-proposals-per-subcat",
        str(args.max_proposals_per_subcat),
        "--min-position-separation",
        str(args.min_position_separation),
        "--probe-position-separation",
        str(args.probe_position_separation),
        "--global-position-groups",
        str(args.global_position_groups),
        "--hotspot-position-groups",
        str(args.hotspot_position_groups),
        "--hotspot-min-position-separation",
        str(args.hotspot_min_position_separation),
        "--probe-restart-interval",
        str(args.probe_restart_interval),
        "--full-restart-interval",
        str(args.full_restart_interval),
    ]
    if args.max_positions is not None:
        cmd.extend(["--max-positions", str(args.max_positions)])
    if args.disable_position_prefilter:
        cmd.append("--disable-position-prefilter")
    if args.disable_state_dedup:
        cmd.append("--disable-state-dedup")

    attempts = max(1, int(args.scene_retries))
    last_result: Dict[str, object] | None = None
    for attempt_idx in range(1, attempts + 1):
        with stdout_log.open("w", encoding="utf-8") as fout, stderr_log.open("w", encoding="utf-8") as ferr:
            print(
                f"[run_parallel] attempt={attempt_idx}/{attempts} scene={scene}",
                file=fout,
                flush=True,
            )
            print(
                " ".join(cmd),
                file=fout,
                flush=True,
            )
            cp = subprocess.run(cmd, stdout=fout, stderr=ferr, text=True)

        summary_path = scene_dir / "scene_pipeline_summary.json"
        if summary_path.is_file():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            result = {
                "scene": scene,
                "status": "failed",
                "error": f"scene_worker_exit_{cp.returncode}",
                "kept_states": 0,
                "proposal_count": 0,
            }
        result["attempt"] = attempt_idx
        result["worker_returncode"] = cp.returncode
        result["stdout_tail"] = _tail_text(stdout_log)
        result["stderr_tail"] = _tail_text(stderr_log)
        last_result = result
        if str(result.get("status", "")) == "ok":
            return result
    assert last_result is not None
    return last_result


def main() -> int:
    args = parse_args()
    scenes = all_scene_names() if args.scenes is None else normalize_scene_tokens(args.scenes.split(","))
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    max_workers = max(1, min(int(args.num_workers), len(scenes) if scenes else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_scene = {pool.submit(_run_one_scene, args, scene): scene for scene in scenes}
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
        "num_workers": max_workers,
        "finished_scenes": [
            item["scene"] for item in results if item.get("status", "ok") in {"ok", "skipped_existing"}
        ],
        "failed_scenes": [item["scene"] for item in results if item.get("status", "ok") == "failed"],
        "total_kept_states": sum(int(item["kept_states"]) for item in results),
        "total_proposals": sum(int(item["proposal_count"]) for item in results),
        "scene_summaries": results,
    }
    overview_path = args.output_root / "parallel_overview.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(overview_path)
    return 0 if not overview["failed_scenes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
