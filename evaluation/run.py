#!/usr/bin/env python3
"""Launch upstream benchmark implementations without replacing their scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


VLMEVAL_REVISION = "c44bc601dcf4698e3cf2fc851a6a0b00b13a2d06"
BENCHMARKS = {
    "vsi": ("VsiBench", "VSI-Bench"),
    "mindcube": ("MindCubeBench", "MindCubeBench_tiny_raw_qa"),
    "spbench": ("SPBench", "SPBench-MV"),
    "mmsi": ("MMSIBench", "MMSIBench_wo_circular"),
    "viewspatial": ("ViewSpatialBench", "ViewSpatialBench"),
    "vsti": (None, "vstibench"),
    "sat-real": (None, "sat_real"),
    "sat-syn": (None, "sat_syn"),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", choices=BENCHMARKS, required=True)
    p.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    p.add_argument("--family", choices=("qwen25vl", "qwen3vl"), required=True)
    p.add_argument("--toolkit", type=Path, required=True, help="Upstream checkout")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, help="VLMEvalKit LMUData directory")
    p.add_argument("--include-path", type=Path, help="Author-provided lmms-eval task definitions")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, help="Default: 64 for VLMEvalKit; task YAML for lmms-eval")
    p.add_argument("--min-pixels", type=int, default=200704)
    p.add_argument("--max-pixels", type=int, default=12845056)
    p.add_argument("--python", default=sys.executable, help="Python in the evaluation environment")
    p.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print configuration; do not import models or write files")
    return p


def plan(args):
    toolkit = args.toolkit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model = str(Path(args.model).expanduser().resolve()) if Path(args.model).expanduser().exists() else args.model
    dataset_class, task = BENCHMARKS[args.bench]
    if args.frames <= 0 or (args.max_new_tokens is not None and args.max_new_tokens <= 0):
        raise ValueError("Frame and token budgets must be positive.")
    if not 0 < args.min_pixels <= args.max_pixels:
        raise ValueError("Expected 0 < min-pixels <= max-pixels.")
    env = {}
    if dataset_class:
        if args.include_path:
            raise ValueError("--include-path applies only to lmms-eval tasks.")
        model_config = {
            "class": "Qwen2VLChat" if args.family == "qwen25vl" else "Qwen3VLChat",
            "model_path": model, "use_custom_prompt": False,
            "use_vllm": False, "temperature": 0, "do_sample": False,
            "max_new_tokens": args.max_new_tokens or 64,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        }
        dataset_config = {"class": dataset_class, "dataset": task}
        if args.bench == "vsi":
            dataset_config.update(nframe=args.frames, fps=-1)
            model_config.update(nframe=args.frames, fps=None)
        config = {"model": {"spatial-interactor": model_config}, "data": {task: dataset_config}}
        command = [args.python, str(toolkit / "run.py"), "--config", str(output / "config.json"),
                   "--work-dir", str(output / "results"), "--mode", args.mode,
                   "--judge", "exact_matching"]
        if args.resume:
            command.append("--reuse")
        if args.data_root:
            env["LMUData"] = str(args.data_root.expanduser().resolve())
        env["PRED_FORMAT"] = "tsv"
        required = toolkit / "run.py"
    else:
        if args.mode != "all" or args.resume:
            raise ValueError("lmms-eval runs use --mode all without --resume; use a fresh output directory.")
        if args.data_root:
            raise ValueError("Set media paths in the author task YAML or HF_HOME for lmms-eval, not --data-root.")
        if any(c in model for c in ",\n"):
            raise ValueError("lmms-eval model paths cannot contain commas or newlines.")
        adapter = "qwen2_5_vl" if args.family == "qwen25vl" else "qwen3_vl"
        model_args = f"pretrained={model},min_pixels={args.min_pixels},max_pixels={args.max_pixels}"
        if args.bench == "vsti":
            model_args += f",max_num_frames={args.frames}"
        config = {"task": task, "model": adapter, "model_args": model_args,
                  "max_new_tokens": args.max_new_tokens}
        command = [args.python, "-m", "lmms_eval", "--model", adapter, "--model_args", model_args,
                   "--tasks", task, "--batch_size", "1", "--output_path", str(output / "results"),
                   "--log_samples"]
        if args.include_path:
            include = args.include_path.expanduser().resolve()
            command += ["--include_path", str(include)]
            config["include_path"] = str(include)
        if args.max_new_tokens:
            command += ["--gen_kwargs", f"max_new_tokens={args.max_new_tokens},temperature=0,do_sample=False"]
        required = toolkit / "lmms_eval" / "__main__.py"
    return {"command": command, "cwd": str(toolkit), "env": env, "config": config,
            "required": str(required), "benchmark": args.bench}


def source_state(toolkit, include_path=None):
    def git(*args):
        result = subprocess.run(["git", "-C", str(toolkit), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    state = {"commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--porcelain"),
             "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--").encode()).hexdigest()}
    if include_path:
        state["task_files"] = {
            p.relative_to(include_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(include_path.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}
        }
    return state


def main():
    args = parser().parse_args()
    try:
        execution = plan(args)
        if args.dry_run:
            print(json.dumps(execution, indent=2))
            return
        if not Path(execution["required"]).is_file():
            raise ValueError(f"Missing upstream entrypoint: {execution['required']}")
        if args.include_path and not args.include_path.is_dir():
            raise ValueError("Task --include-path is not a directory.")
        state = source_state(Path(execution["cwd"]), args.include_path)
        manifest = {"benchmark": args.bench, "config": execution["config"],
                    "env": execution["env"], "upstream": state}
        output = args.output.expanduser().resolve()
        manifest_file = output / "run_manifest.json"
        if output.exists() and any(output.iterdir()):
            if not args.resume or not manifest_file.is_file():
                raise ValueError("Output is not empty; choose a new directory or use --resume with the same configuration.")
            if json.loads(manifest_file.read_text()) != manifest:
                raise ValueError("Run configuration or upstream code changed; use a new output directory.")
        output.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "config.json").write_text(json.dumps(execution["config"], indent=2) + "\n")
        env = os.environ.copy()
        env.update(execution["env"])
        env["PYTHONPATH"] = execution["cwd"] + os.pathsep + env.get("PYTHONPATH", "")
        print(shlex.join(execution["command"]), flush=True)
        subprocess.run(execution["command"], cwd=execution["cwd"], env=env, check=True)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
