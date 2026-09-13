#!/usr/bin/env python3
"""Build one deterministic annotation item per unique video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    yield row
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or a data list")
    for row in rows:
        if isinstance(row, dict):
            yield row


def row_videos(row: dict[str, Any]) -> list[str]:
    values = row.get("videos")
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [str(value) for value in values if value]
    value = row.get("video") or row.get("video_path")
    if value:
        return [str(value)]
    inputs = row.get("input") or {}
    if isinstance(inputs, dict):
        value = inputs.get("video") or inputs.get("video_path") or inputs.get("video_zip")
        if value:
            return [str(value)]
    return []


def task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(
        row.get("task_type")
        or metadata.get("task_type")
        or metadata.get("original_question_type")
        or "unknown"
    )


def stable_id(portable_video_path: str) -> str:
    digest = hashlib.sha1(portable_video_path.encode("utf-8")).hexdigest()[:20]
    return f"trace_{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    unique: dict[str, dict[str, Any]] = {}
    source_rows = 0
    rows_without_video = 0

    for source in args.input:
        for row in read_rows(source):
            source_rows += 1
            videos = row_videos(row)
            if not videos:
                rows_without_video += 1
                continue
            for video in videos:
                path = Path(video)
                absolute = path if path.is_absolute() else data_root / path
                absolute = absolute.resolve()
                try:
                    portable = absolute.relative_to(data_root).as_posix()
                except ValueError as exc:
                    raise ValueError(f"Video is outside --data-root: {absolute}") from exc
                normalized = os.path.normpath(str(absolute))
                if args.require_files and not Path(normalized).is_file():
                    raise FileNotFoundError(f"Missing video: {normalized}")
                item = unique.setdefault(
                    normalized,
                    {
                        "id": stable_id(portable),
                        "video": portable,
                        "task_types": set(),
                        "source_files": set(),
                        "source_row_count": 0,
                    },
                )
                item["task_types"].add(task_type(row))
                item["source_files"].add(source.name)
                item["source_row_count"] += 1

    output_rows = []
    for item in unique.values():
        output_rows.append(
            {
                **item,
                "task_types": sorted(item["task_types"]),
                "source_files": sorted(item["source_files"]),
            }
        )
    output_rows.sort(key=lambda row: (row["video"], row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_rows": source_rows,
        "unique_videos": len(output_rows),
        "rows_without_video": rows_without_video,
        "collapsed_video_references": sum(row["source_row_count"] for row in output_rows),
        "task_counts_by_video_membership": dict(
            Counter(task for row in output_rows for task in row["task_types"]).most_common()
        ),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
