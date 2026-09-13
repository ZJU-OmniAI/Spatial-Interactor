#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SRC_DATA = Path("/path/to/workspace/DATA")
DEFAULT_OUT = Path("/path/to/workspace/SFT_QA")

SIM_DATASETS = {
    "AI2THOR": {
        "src": SRC_DATA / "AI2THOR",
        "qa": "ai2thor_qa_standardized_data_only.json",
        "summary": "ai2thor_package_summary.json",
    },
    "PROC": {
        "src": SRC_DATA / "PROC",
        "qa": "proc_qa_standardized_data_only.json",
        "summary": "proc_package_summary.json",
    },
    "HSSD": {
        "src": SRC_DATA / "HSSD",
        "qa": "hssd_qa_standardized_data_only.json",
        "summary": "hssd_package_summary.json",
    },
    "REP": {
        "src": SRC_DATA / "REP",
        "qa": "rep_qa_standardized_data_only.json",
        "summary": "rep_package_summary.json",
    },
}

REAL_TASKS = ["action_inference", "movement_degree_comparison", "movement_sequence_sorting"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree_files(src: Path, dst: Path) -> int:
    count = 0
    if not src.exists():
        return 0
    for path in src.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src)
            copy_file(path, dst / rel)
            count += 1
    return count


def scene_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("scene")) for row in rows if row.get("scene")})


def package_simulator(out_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for name, cfg in SIM_DATASETS.items():
        src = cfg["src"]
        dst = out_root / "simulator" / name
        ensure_dir(dst)
        qa_rows = load_json(src / cfg["qa"])
        write_json(dst / "qa_data.json", qa_rows)
        if (src / cfg["summary"]).exists():
            copy_file(src / cfg["summary"], dst / "package_summary.json")
        copied = copy_tree_files(src / "images", dst / "images")
        summaries.append(
            {
                "domain": "simulator",
                "dataset": name,
                "qa_count": len(qa_rows),
                "scene_count": scene_count(qa_rows),
                "image_files_seen": copied,
                "qa_path": str(dst / "qa_data.json"),
                "image_root": str(dst / "images"),
            }
        )
        print(f"[sim] {name}: qa={len(qa_rows)} scenes={scene_count(qa_rows)} image_files_seen={copied}", flush=True)
    return summaries


def rewrite_input_paths(value: Any, src_sample_dir: Path, rel_sample_dir: str, real_src_root: Path) -> Any:
    if isinstance(value, str):
        raw = Path(value)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(real_src_root / value)
        candidates.append(src_sample_dir / raw.name)
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.parent == src_sample_dir:
                    return f"{rel_sample_dir}/{candidate.name}"
            except OSError:
                pass
        return value
    if isinstance(value, list):
        return [rewrite_input_paths(item, src_sample_dir, rel_sample_dir, real_src_root) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_input_paths(item, src_sample_dir, rel_sample_dir, real_src_root) for key, item in value.items()}
    return value


def package_real(out_root: Path, src_root: Path) -> list[dict[str, Any]]:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_copy_counts: Counter = Counter()

    for task in REAL_TASKS:
        path = src_root / task / "qa_data.jsonl"
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = str(row["dataset"])
                qa_id = str(row["qa_id"])
                src_sample_dir = src_root / task / qa_id
                dst_sample_dir = out_root / "real" / dataset / "images" / qa_id
                copied = copy_tree_files(src_sample_dir, dst_sample_dir)
                image_copy_counts[dataset] += copied
                rel_sample_dir = f"images/{qa_id}"
                row = dict(row)
                row["input"] = rewrite_input_paths(row.get("input", {}), src_sample_dir, rel_sample_dir, src_root)
                rows_by_dataset[dataset].append(row)

    summaries = []
    for dataset in sorted(rows_by_dataset):
        dst = out_root / "real" / dataset
        rows = rows_by_dataset[dataset]
        rows.sort(key=lambda row: (row.get("task_type", ""), row.get("qa_id", "")))
        jsonl_path = dst / "qa_data.jsonl"
        json_path = dst / "qa_data.json"
        if jsonl_path.exists():
            jsonl_path.unlink()
        for row in rows:
            append_jsonl(jsonl_path, row)
        write_json(json_path, rows)
        task_counts = Counter(str(row.get("task_type")) for row in rows)
        scenes = {str(row.get("scene")) for row in rows if row.get("scene")}
        summary = {
            "domain": "real",
            "dataset": dataset,
            "qa_count": len(rows),
            "scene_count": len(scenes),
            "task_counts": dict(task_counts),
            "image_files_seen": int(image_copy_counts[dataset]),
            "qa_jsonl_path": str(jsonl_path),
            "qa_json_path": str(json_path),
            "image_root": str(dst / "images"),
        }
        write_json(dst / "package_summary.json", summary)
        summaries.append(summary)
        print(f"[real] {dataset}: qa={len(rows)} scenes={len(scenes)} image_files_seen={image_copy_counts[dataset]}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SFT-ready simulator and real QA data under /path/to/workspace.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--real-src-root", type=Path, default=SRC_DATA / "MOTION_QA_FROM_SEGMENTS")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_root)
    summaries = []
    if not args.skip_sim:
        summaries.extend(package_simulator(args.output_root))
    if not args.skip_real:
        summaries.extend(package_real(args.output_root, args.real_src_root))

    manifest = {
        "output_root": str(args.output_root),
        "total_qa": sum(item["qa_count"] for item in summaries),
        "total_scene_count_by_source_sum": sum(item["scene_count"] for item in summaries),
        "datasets": summaries,
    }
    write_json(args.output_root / "manifest.json", manifest)
    write_json(args.output_root / "distribution_summary.json", summaries)
    readme = [
        "# SFT-ready QA package",
        "",
        "This directory contains packaged QA data with relative image paths.",
        "",
        "- `simulator/<dataset>/qa_data.json`: simulator QA rows.",
        "- `real/<dataset>/qa_data.jsonl`: real-scene QA rows.",
        "- `real/<dataset>/qa_data.json`: same rows as JSON array.",
        "- `images/`: image files referenced by each dataset's QA rows.",
        "",
        "Large raw sources and intermediate motion segments are organized under `/path/to/workspace`.",
        "",
    ]
    for item in summaries:
        readme.append(f"- {item['domain']}/{item['dataset']}: {item['qa_count']} QA, {item['scene_count']} scenes")
    (args.output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
