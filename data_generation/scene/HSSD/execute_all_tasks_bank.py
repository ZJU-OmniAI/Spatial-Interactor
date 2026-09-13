#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CLASS_SCRIPTS = {
    1: "execute_class01_bank.py",
    2: "execute_class02_bank.py",
    3: "execute_class03_bank.py",
    4: "execute_class04_bank.py",
    5: "execute_class05_bank.py",
    6: "execute_class06_bank.py",
    7: "execute_class07_bank.py",
    8: "execute_class08_bank.py",
    9: "execute_class09_bank.py",
    10: "execute_class10_bank.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified execution runner for HSSD class banks.")
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classes", type=str, default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--subcats", type=str, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--target-per-subcat", type=int, default=5)
    parser.add_argument("--max-episode-retry", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--field-of-view", type=int, default=90)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--use-cloud-rendering", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    classes = [int(part.strip()) for part in args.classes.split(",") if part.strip()]
    for class_id in classes:
        script_name = CLASS_SCRIPTS.get(class_id)
        if script_name is None:
            summaries.append({"class_id": class_id, "status": "missing_script"})
            continue
        script_path = Path(__file__).resolve().parent / script_name
        class_output = args.output_root / str(class_id)
        cmd = [
            sys.executable,
            str(script_path),
            "--state-bank",
            str(args.state_bank),
            "--proposals",
            str(args.proposals),
            "--output-root",
            str(class_output),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--grid-size",
            str(args.grid_size),
            "--field-of-view",
            str(args.field_of_view),
            "--visibility-distance",
            str(args.visibility_distance),
            "--target-per-subcat",
            str(args.target_per_subcat),
        ]
        if args.subcats:
            cmd.extend(["--subcats", args.subcats])
        if args.max_proposals is not None:
            cmd.extend(["--max-proposals", str(args.max_proposals)])
        if args.max_episode_retry is not None:
            cmd.extend(["--max-episode-retry", str(args.max_episode_retry)])
        if args.use_cloud_rendering:
            cmd.append("--use-cloud-rendering")
        if args.reset_output:
            cmd.append("--reset-output")
        result = subprocess.run(cmd, capture_output=True, text=True)
        summaries.append(
            {
                "class_id": class_id,
                "script": str(script_path),
                "output_root": str(class_output),
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        )
    summary_path = args.output_root / "execution_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0 if all(item.get("returncode", 0) == 0 for item in summaries if "returncode" in item) else 1


if __name__ == "__main__":
    raise SystemExit(main())
