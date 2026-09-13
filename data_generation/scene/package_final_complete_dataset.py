#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


FINAL_ROOT = Path("/path/to/workspace/FINAL")

ASSETS = {
    "AI2THOR": {
        "all_json": Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/ai2thor_qa_standardized_all.json"),
        "render_root": Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/render_output"),
        "extra_files": [
            Path("/path/to/workspace/SCENEOUTPUT/AI2THOR/ai2thor_qa_standardized_summary.json"),
        ],
    },
    "PROC": {
        "all_json": Path("/path/to/workspace/SCENEOUTPUT/PROC/proc_qa_standardized_all.json"),
        "render_root": Path("/path/to/workspace/SCENEOUTPUT/PROC/render_output"),
        "extra_files": [
            Path("/path/to/workspace/SCENEOUTPUT/PROC/proc_qa_standardized_summary.json"),
        ],
    },
    "HSSD": {
        "all_json": Path("/path/to/workspace/SCENEOUTPUT/HSSD/hssd_qa_standardized_all.json"),
        "render_root": Path("/path/to/workspace/SCENEOUTPUT/HSSD/render_output_prop_20260423"),
        "extra_files": [
            Path("/path/to/workspace/SCENEOUTPUT/HSSD/hssd_qa_standardized_summary.json"),
            Path("/path/to/workspace/SCENEOUTPUT/HSSD/hssd_quality_filter_v2full_summary.json"),
            Path("/path/to/workspace/SCENEOUTPUT/HSSD/hssd_quality_filter_v2full_keep_sample_ids.txt"),
        ],
    },
    "REP": {
        "all_json": Path("/path/to/workspace/SCENEOUTPUT/REP/rep_qa_standardized_all.json"),
        "render_root": Path("/path/to/workspace/SCENEOUTPUT/REP/render_output_prop_20260424"),
        "extra_files": [
            Path("/path/to/workspace/SCENEOUTPUT/REP/rep_qa_standardized_summary.json"),
            Path("/path/to/workspace/SCENEOUTPUT/REP/rep_quality_filter_v2full_summary.json"),
            Path("/path/to/workspace/SCENEOUTPUT/REP/rep_quality_filter_v2full_keep_sample_ids.txt"),
        ],
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def _rewrite_input_paths(value: Any, render_root: Path, portable_root_name: str = "render_output") -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_input_paths(item, render_root, portable_root_name) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_input_paths(item, render_root, portable_root_name) for item in value]
    if isinstance(value, str):
        path = Path(value)
        try:
            rel = path.relative_to(render_root)
        except Exception:
            return value
        return str(Path(portable_root_name) / rel)
    return value


def _package_asset(asset: str, config: dict[str, Any]) -> dict[str, Any]:
    asset_dir = FINAL_ROOT / asset
    render_root = Path(config["render_root"])
    payload = json.loads(Path(config["all_json"]).read_text(encoding="utf-8"))
    rows = payload["data"]

    unique_files: set[str] = set()
    unique_sample_dirs: set[str] = set()
    rewritten_rows: list[dict[str, Any]] = []
    linked = 0
    copied = 0
    reused = 0

    for index, row in enumerate(rows, start=1):
        updated = deepcopy(row)
        updated["input"] = _rewrite_input_paths(updated.get("input") or {}, render_root)
        rewritten_rows.append(updated)

        for frame_path in (row.get("input") or {}).get("frame_paths") or []:
            src = Path(frame_path)
            rel = src.relative_to(render_root)
            dst = asset_dir / "render_output" / rel
            status = _safe_link_or_copy(src, dst)
            unique_files.add(str(rel))
            unique_sample_dirs.add(str(rel.parent))
            if status == "linked":
                linked += 1
            elif status == "copied":
                copied += 1
            else:
                reused += 1

        if index % 1000 == 0:
            print(
                json.dumps(
                    {
                        "asset": asset,
                        "processed_rows": index,
                        "total_rows": len(rows),
                        "unique_files": len(unique_files),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    asset_lower = asset.lower()
    all_payload = deepcopy(payload)
    all_payload["source_root"] = "render_output"
    all_payload["packaged_root"] = str(asset_dir)
    all_payload["portable_paths"] = True
    all_payload["data"] = rewritten_rows

    data_only = rewritten_rows
    prompt_pair = [
        {
            "prompt": row.get("prompt", ""),
            "qa": {"q": row.get("question", ""), "a": row.get("answer", "")},
            "gt": row.get("gt", {}),
        }
        for row in rewritten_rows
    ]

    _write_json(asset_dir / f"{asset_lower}_qa_standardized_all.json", all_payload)
    _write_json(asset_dir / f"{asset_lower}_qa_standardized_data_only.json", data_only)
    _write_json(asset_dir / f"{asset_lower}_prompt_qa_pair_gt_only.json", prompt_pair)

    for extra in config.get("extra_files") or []:
        if Path(extra).exists():
            shutil.copy2(extra, asset_dir / Path(extra).name)

    packaging_summary = {
        "asset": asset,
        "qa_count": len(rewritten_rows),
        "render_root_original": str(render_root),
        "render_root_packaged": str(asset_dir / "render_output"),
        "portable_paths": True,
        "unique_image_files": len(unique_files),
        "unique_sample_dirs": len(unique_sample_dirs),
        "link_stats": {
            "linked": linked,
            "copied": copied,
            "reused_existing": reused,
        },
    }
    _write_json(asset_dir / f"{asset_lower}_package_summary.json", packaging_summary)
    return packaging_summary


def main() -> int:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [_package_asset(asset, config) for asset, config in ASSETS.items()]
    manifest = {
        "root": str(FINAL_ROOT),
        "assets": summaries,
        "total_qa_count": sum(item["qa_count"] for item in summaries),
        "portable_paths": True,
    }
    _write_json(FINAL_ROOT / "package_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
