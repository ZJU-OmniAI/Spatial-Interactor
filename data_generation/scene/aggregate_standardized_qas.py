#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _iter_standardized_json_paths(input_root: Path) -> Iterable[Path]:
    yield from sorted(input_root.glob("*/[0-9]*/meta/qa_data_standardized.json"))


def _load_rows(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_paths(row: Dict) -> Tuple[str, ...]:
    paths = ((row.get("input") or {}).get("frame_paths") or [])
    return tuple(str(item) for item in paths)


def _row_key(row: Dict) -> Tuple:
    return (
        str(row.get("task_type") or ""),
        str(row.get("scene") or ""),
        _frame_paths(row),
        str(row.get("question_core") or row.get("question") or ""),
        str(row.get("answer") or ""),
    )


def _frame_files_exist(row: Dict) -> bool:
    for item in _frame_paths(row):
        if not Path(item).exists():
            return False
    return True


def _build_prompt_pair(row: Dict) -> Dict:
    return {
        "prompt": row.get("prompt", ""),
        "qa": {
            "q": row.get("question", ""),
            "a": row.get("answer", ""),
        },
        "gt": row.get("gt", {}),
    }


def aggregate(input_root: Path, *, asset: str, output_dir: Path, no_filter: bool = False) -> Dict[str, object]:
    rows: List[Dict] = []
    total_files = 0
    skipped_missing_frames = 0
    skipped_duplicates = 0
    seen_keys = set()

    for path in _iter_standardized_json_paths(input_root):
        total_files += 1
        data = _load_rows(path)
        for row in data:
            if not no_filter and not _frame_files_exist(row):
                skipped_missing_frames += 1
                continue
            if not no_filter:
                key = _row_key(row)
                if key in seen_keys:
                    skipped_duplicates += 1
                    continue
                seen_keys.add(key)
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_payload = {
        "asset": asset,
        "source_root": str(input_root),
        "source_file_count": total_files,
        "qa_count": len(rows),
        "data": rows,
        "filter_stats": {
            "missing_frames_skipped": skipped_missing_frames,
            "duplicate_rows_skipped": skipped_duplicates,
            "filter_enabled": not no_filter,
        },
    }
    data_only = rows
    prompt_pair = [_build_prompt_pair(row) for row in rows]

    all_path = output_dir / f"{asset.lower()}_qa_standardized_all.json"
    data_only_path = output_dir / f"{asset.lower()}_qa_standardized_data_only.json"
    prompt_pair_path = output_dir / f"{asset.lower()}_prompt_qa_pair_gt_only.json"
    summary_path = output_dir / f"{asset.lower()}_qa_standardized_summary.json"

    all_path.write_text(json.dumps(all_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    data_only_path.write_text(json.dumps(data_only, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_pair_path.write_text(json.dumps(prompt_pair, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(all_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "all_path": str(all_path),
        "data_only_path": str(data_only_path),
        "prompt_pair_path": str(prompt_pair_path),
        "summary_path": str(summary_path),
        "source_file_count": total_files,
        "qa_count": len(rows),
        "missing_frames_skipped": skipped_missing_frames,
        "duplicate_rows_skipped": skipped_duplicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate standardized QA files into top-level JSON bundles.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()

    result = aggregate(args.input_root, asset=args.asset, output_dir=args.output_dir, no_filter=args.no_filter)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
