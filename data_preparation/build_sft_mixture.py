#!/usr/bin/env python3
"""Build the paper SFT mixture: LSI levels 1-2 plus Public-80K."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COUNTS = {
    "lsi_l1": 13_109,
    "lsi_l2": 69_487,
    "vsi": 50_000,
    "mindcube": 10_000,
    "vsti": 20_000,
}
PUBLIC_SOURCES = ("vsi", "mindcube", "vsti")
DEFAULT_EXCLUDED_L1_TASKS = ("long_horizon_manipulation_program",)


def read_rows(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                yield path, line_number, row


def normalize_conversations(row: dict[str, Any], location: str) -> list[dict[str, str]]:
    values = row.get("conversations") or row.get("messages")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{location} has no two-turn conversation")
    result = []
    for message in values:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from") or message.get("role") or "").lower()
        text = message.get("value") if "value" in message else message.get("content")
        role = {"user": "human", "assistant": "gpt"}.get(role, role)
        if role in {"human", "gpt", "system"} and isinstance(text, str):
            result.append({"from": role, "value": text})
    if not any(item["from"] == "human" for item in result) or not any(
        item["from"] == "gpt" for item in result
    ):
        raise ValueError(f"{location} has invalid conversation roles")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1", type=Path, nargs="+", required=True)
    parser.add_argument("--l2", type=Path, nargs="+", required=True)
    parser.add_argument("--vsi", type=Path, nargs="+", required=True)
    parser.add_argument("--mindcube", type=Path, nargs="+", required=True)
    parser.add_argument("--vsti", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--excluded-l1-tasks",
        default=",".join(DEFAULT_EXCLUDED_L1_TASKS),
        help=(
            "Comma-separated L1 tasks excluded from the reported SFT mixture. "
            "The full LSI-108K release still retains these records."
        ),
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "lsi_l1": args.l1,
        "lsi_l2": args.l2,
        "vsi": args.vsi,
        "mindcube": args.mindcube,
        "vsti": args.vsti,
    }
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    excluded_l1_tasks = {
        value.strip() for value in args.excluded_l1_tasks.split(",") if value.strip()
    }

    for source_name, paths in sources.items():
        for path, line_number, row in read_rows(paths):
            location = f"{path}:{line_number}"
            metadata = row.get("metadata") or {}
            task = str(row.get("task_type") or metadata.get("task_type") or "")
            if source_name == "lsi_l1" and task in excluded_l1_tasks:
                excluded_counts[task] += 1
                continue
            normalized = dict(row)
            normalized.pop("messages", None)
            normalized["conversations"] = normalize_conversations(row, location)
            row_id = str(row.get("id") or f"{source_name}:{path.name}:{line_number}")
            unique_id = row_id
            if unique_id in seen_ids:
                unique_id = f"{source_name}:{row_id}"
            if unique_id in seen_ids:
                raise ValueError(f"Duplicate sample id after source prefixing: {unique_id}")
            normalized["id"] = unique_id
            metadata = dict(normalized.get("metadata") or {})
            metadata["release_source"] = source_name
            normalized["metadata"] = metadata
            seen_ids.add(unique_id)
            rows.append(normalized)
            source_counts[source_name] += 1

    mismatches = {
        name: {"expected": expected, "actual": source_counts[name]}
        for name, expected in EXPECTED_COUNTS.items()
        if source_counts[name] != expected
    }
    if mismatches and not args.allow_count_mismatch:
        raise RuntimeError(
            "Paper SFT mixture count mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "spatial_interactor_sft.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    dataset_info = {
        "spatial_interactor_sft": {
            "file_name": output.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images", "videos": "videos"},
        }
    }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": "LSI L1 + L2 + Public-80K",
        "total_rows": len(rows),
        "source_counts": dict(source_counts),
        "public_rows": sum(source_counts[name] for name in PUBLIC_SOURCES),
        "lsi_rows": source_counts["lsi_l1"] + source_counts["lsi_l2"],
        "excluded_l1_task_counts": dict(excluded_counts),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "input_sha256": {
            f"{source}:{index}:{path.name}": sha256(path)
            for source, paths in sources.items()
            for index, path in enumerate(paths)
        },
        "output_sha256": sha256(output),
    }
    (args.output_dir / "sft_mixture_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
