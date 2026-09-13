#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


FINAL_ROOT = Path("/path/to/workspace/FINAL")
ASSETS = ["AI2THOR", "PROC", "HSSD", "REP"]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rewrite_input_paths(value: Any, mapper: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_input_paths(item, mapper) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_input_paths(item, mapper) for item in value]
    if isinstance(value, str):
        return mapper.get(value, value)
    return value


def simplify_asset(asset: str) -> dict[str, Any]:
    asset_dir = FINAL_ROOT / asset
    asset_lower = asset.lower()
    all_json = asset_dir / f"{asset_lower}_qa_standardized_all.json"
    data_only_json = asset_dir / f"{asset_lower}_qa_standardized_data_only.json"
    package_summary = asset_dir / f"{asset_lower}_package_summary.json"
    render_root = asset_dir / "render_output"
    images_root = asset_dir / "images"

    payload = json.loads(all_json.read_text(encoding="utf-8"))
    rows = payload["data"]
    images_root.mkdir(parents=True, exist_ok=True)

    unique_files = 0
    unique_sample_dirs: set[str] = set()

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        path_map: dict[str, str] = {}
        for old_rel in (row.get("input") or {}).get("frame_paths") or []:
            src = asset_dir / old_rel
            filename = Path(old_rel).name
            new_rel = str(Path("images") / sample_id / filename)
            dst = asset_dir / new_rel
            _safe_link_or_copy(src, dst)
            path_map[str(old_rel)] = new_rel
            unique_sample_dirs.add(sample_id)
            unique_files += 1
        row["input"] = _rewrite_input_paths(row.get("input") or {}, path_map)

    payload["data"] = rows
    payload["source_root"] = "images"
    payload["portable_paths"] = True
    payload["simple_layout"] = True

    _write_json(all_json, payload)
    _write_json(data_only_json, rows)

    summary = json.loads(package_summary.read_text(encoding="utf-8")) if package_summary.exists() else {"asset": asset}
    summary["layout"] = "images/<sample_id>/<frame_file>"
    summary["packaged_root"] = str(images_root)
    summary["portable_paths"] = True
    summary["simple_layout"] = True
    summary["unique_sample_dirs"] = len(unique_sample_dirs)
    summary["unique_image_files"] = len(list(images_root.rglob("*.png")))
    _write_json(package_summary, summary)

    if render_root.exists():
        shutil.rmtree(render_root)

    return {
        "asset": asset,
        "qa_count": len(rows),
        "image_file_count": summary["unique_image_files"],
        "layout": summary["layout"],
    }


def main() -> int:
    manifest_path = FINAL_ROOT / "package_manifest.json"
    results = [simplify_asset(asset) for asset in ASSETS]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["assets"] = results
    manifest["portable_paths"] = True
    manifest["simple_layout"] = True
    manifest["root"] = str(FINAL_ROOT)
    manifest["total_qa_count"] = sum(item["qa_count"] for item in results)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
