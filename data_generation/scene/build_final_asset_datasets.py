#!/usr/bin/env python3

from __future__ import annotations

import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any


ROOT = Path("/path/to/workspace/SCENEOUTPUT")
FINAL_DIR = ROOT / "FINAL_ASSETS"


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_prompt_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row.get("prompt", ""),
        "qa": {
            "q": row.get("question", ""),
            "a": row.get("answer", ""),
        },
        "gt": row.get("gt", {}),
    }


def _standardize_filtered_asset(
    *,
    asset: str,
    kept_json: Path,
    output_dir: Path,
    augment_row,
    source_root: str,
    filter_source: str,
) -> dict[str, Any]:
    payload = _load_json(kept_json)
    rows = payload["data"] if isinstance(payload, dict) else payload
    standardized_rows = [augment_row(row) for row in rows]

    all_payload = {
        "asset": asset,
        "source_root": source_root,
        "filter_source": filter_source,
        "qa_count": len(standardized_rows),
        "data": standardized_rows,
        "filter_stats": {
            "kept_after_quality_filter": len(standardized_rows),
            "quality_filter_path": str(kept_json),
            "filter_enabled": True,
        },
    }
    prompt_pair = [_build_prompt_pair(row) for row in standardized_rows]

    asset_lower = asset.lower()
    all_path = output_dir / f"{asset_lower}_qa_standardized_all.json"
    data_only_path = output_dir / f"{asset_lower}_qa_standardized_data_only.json"
    prompt_pair_path = output_dir / f"{asset_lower}_prompt_qa_pair_gt_only.json"
    summary_path = output_dir / f"{asset_lower}_qa_standardized_summary.json"

    _write_json(all_path, all_payload)
    _write_json(data_only_path, standardized_rows)
    _write_json(prompt_pair_path, prompt_pair)
    _write_json(summary_path, all_payload)

    return {
        "asset": asset,
        "qa_count": len(standardized_rows),
        "all_path": str(all_path),
        "data_only_path": str(data_only_path),
        "prompt_pair_path": str(prompt_pair_path),
        "summary_path": str(summary_path),
    }


def _existing_asset_summary(asset: str, all_path: Path, data_only_path: Path, prompt_pair_path: Path) -> dict[str, Any]:
    payload = _load_json(all_path)
    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    summary_path = all_path.parent / f"{asset.lower()}_qa_standardized_summary.json"
    return {
        "asset": asset,
        "qa_count": len(data),
        "all_path": str(all_path),
        "data_only_path": str(data_only_path),
        "prompt_pair_path": str(prompt_pair_path),
        "summary_path": str(summary_path),
    }


def main() -> int:
    shared = SourceFileLoader("shared_qa_augmentation", "/path/to/workspace/SCENE/shared_qa_augmentation.py").load_module()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []

    summaries.append(
        _existing_asset_summary(
            "AI2THOR",
            ROOT / "AI2THOR" / "ai2thor_qa_standardized_all.json",
            ROOT / "AI2THOR" / "ai2thor_qa_standardized_data_only.json",
            ROOT / "AI2THOR" / "ai2thor_prompt_qa_pair_gt_only.json",
        )
    )
    summaries.append(
        _existing_asset_summary(
            "PROC",
            ROOT / "PROC" / "proc_qa_standardized_all.json",
            ROOT / "PROC" / "proc_qa_standardized_data_only.json",
            ROOT / "PROC" / "proc_prompt_qa_pair_gt_only.json",
        )
    )
    summaries.append(
        _standardize_filtered_asset(
            asset="HSSD",
            kept_json=ROOT / "HSSD" / "hssd_quality_filter_v2full_kept_all.json",
            output_dir=ROOT / "HSSD",
            augment_row=shared.augment_row,
            source_root=str(ROOT / "HSSD" / "render_output_prop_20260423"),
            filter_source=str(ROOT / "HSSD" / "hssd_quality_filter_v2full_summary.json"),
        )
    )
    summaries.append(
        _standardize_filtered_asset(
            asset="REP",
            kept_json=ROOT / "REP" / "rep_quality_filter_v2full_kept_all.json",
            output_dir=ROOT / "REP",
            augment_row=shared.augment_row,
            source_root=str(ROOT / "REP" / "render_output_prop_20260424"),
            filter_source=str(ROOT / "REP" / "rep_quality_filter_v2full_summary.json"),
        )
    )

    manifest = {
        "assets": summaries,
        "total_qa_count": sum(int(item["qa_count"]) for item in summaries),
    }
    _write_json(FINAL_DIR / "final_assets_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
