#!/usr/bin/env python3
"""Create portable, privacy-checked LSI-108K annotation files."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from assign_curriculum_levels import DEFAULT_TASK_LEVELS, infer_task, read_rows


EXPECTED_COUNTS = {"l1": 15_109, "l2": 69_487, "l3": 22_922}
SFT_EXCLUDED_TASKS = {"long_horizon_manipulation_program"}
PRIVATE_PATTERN = re.compile(
    r"(?:/home(?:2)?/|/Dataset2/|/data/I\d{4,}/|"
    r"(?:sk|ms)-[A-Za-z0-9]{16,}|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY)"
)
SCALAR_METADATA_FIELDS = (
    "source_domain",
    "dataset",
    "source",
    "scene",
    "subcat",
    "answer_text",
    "correct_option",
)
JSON_METADATA_FIELDS = ("gt", "options", "correct_options")
SOURCE_NAMES = {
    "AI2THOR": "AI2-THOR", "PROC": "ProcTHOR", "REP": "ReplicaCAD",
    "arkit": "ARKitScenes", "scannetpp": "ScanNet++", "scannetv2": "ScanNet",
    "3rscan": "3RScan", "multiscan": "MultiScan", "roomtour3d": "RoomTour3D",
    "bridgedata_v2": "BridgeData V2",
}


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    source = str(metadata.get("dataset") or metadata.get("source") or "")
    if source.startswith("camera_pose_qa_") and "arkit" in source:
        source = "ARKitScenes"
    elif source.startswith(("roomtour3d_", "videoqa_extra_natural_candidates_")):
        source = "RoomTour3D"
    return {
        "curriculum_level": metadata["curriculum_level"],
        "task_type": metadata["task_type"],
        "source": SOURCE_NAMES.get(source, source),
        "scene": str(metadata.get("scene") or ""),
        "used_in_reported_sft": metadata["used_in_reported_sft"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-counts", action="store_true")
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    parser.add_argument("--metadata", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--parquet", action="store_true", help="Also write Parquet for Hub configurations.")
    return parser.parse_args()


def normalize_conversations(row: dict[str, Any], location: str) -> list[dict[str, str]]:
    messages = row.get("conversations") or row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{location} has no two-turn conversation")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from") or message.get("role") or "").lower()
        role = {"user": "human", "assistant": "gpt"}.get(role, role)
        content = message.get("value") if "value" in message else message.get("content")
        if role in {"human", "gpt", "system"} and isinstance(content, str):
            normalized.append({"from": role, "value": content})
    if not any(item["from"] == "human" for item in normalized) or not any(
        item["from"] == "gpt" for item in normalized
    ):
        raise ValueError(f"{location} has invalid conversation roles")
    return normalized


def portable_media(row: dict[str, Any], field: str, location: str) -> list[str]:
    values = row.get(field) or []
    if not values and row.get("media_type") == {"images": "image", "videos": "video"}[field]:
        values = row.get("media_paths") or []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError(f"{location} has invalid {field}")
    result = []
    for value in values:
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{location} has non-portable media path: {value}")
        result.append(path.as_posix())
    return result


def assert_private_data_absent(value: Any, location: str) -> None:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    match = PRIVATE_PATTERN.search(serialized)
    if match:
        raise ValueError(f"{location} contains private or machine-specific content: {match.group(0)}")


def remove_private_values(value: Any) -> Any:
    """Drop development-machine strings while preserving geometric metadata."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            cleaned = remove_private_values(item)
            if cleaned is not None:
                result[key] = cleaned
        return result
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := remove_private_values(item)) is not None]
    if isinstance(value, str) and PRIVATE_PATTERN.search(value):
        return None
    return value


def release_row(row: dict[str, Any], location: str, metadata_mode: str = "minimal") -> tuple[str, str, dict[str, Any]]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{location} has invalid metadata")
    task = infer_task(row, DEFAULT_TASK_LEVELS)
    level = DEFAULT_TASK_LEVELS.get(task, "")
    if level not in EXPECTED_COUNTS:
        raise ValueError(f"{location} has unmapped task_type: {task!r}")

    output_metadata = {
        key: str(remove_private_values(metadata.get(key)) or "")
        for key in SCALAR_METADATA_FIELDS
    }
    for key in JSON_METADATA_FIELDS:
        value = metadata.get(key)
        if value is None and metadata.get(f"{key}_json"):
            value = json.loads(metadata[f"{key}_json"])
        output_metadata[f"{key}_json"] = (
            json.dumps(remove_private_values(value), ensure_ascii=False, separators=(",", ":"))
            if value is not None else ""
        )

    output_metadata.update(
        {
            "task_type": task,
            "curriculum_level": level.upper(),
            "used_in_reported_sft": (
                level in {"l1", "l2"} and task not in SFT_EXCLUDED_TASKS
            ),
        }
    )
    if metadata_mode == "minimal":
        output_metadata = compact_metadata(output_metadata)

    images = portable_media(row, "images", location)
    videos = portable_media(row, "videos", location)
    if bool(images) == bool(videos):
        raise ValueError(f"{location} must have exactly one media type")
    output: dict[str, Any] = {
        "id": str(row.get("id") or row.get("qa_id") or ""),
        "conversations": normalize_conversations(row, location),
        "metadata": output_metadata,
        "media_paths": images or videos,
        "media_type": "image" if images else "video",
    }
    if not output["id"]:
        raise ValueError(f"{location} has no id")
    assert_private_data_absent(output, location)
    return level, task, output


@contextmanager
def output_handle(path: Path, compression: str) -> Iterator[TextIO]:
    if compression == "none":
        with path.open("w", encoding="utf-8") as handle:
            yield handle
        return
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                yield handle


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".jsonl.gz" if args.compression == "gzip" else ".jsonl"
    paths = {level: args.output_dir / f"lsi_{level}{suffix}" for level in EXPECTED_COUNTS}
    counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    media_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    with (
        output_handle(paths["l1"], args.compression) as l1_handle,
        output_handle(paths["l2"], args.compression) as l2_handle,
        output_handle(paths["l3"], args.compression) as l3_handle,
    ):
        handles = {"l1": l1_handle, "l2": l2_handle, "l3": l3_handle}
        for path, line_number, row in read_rows(args.input):
            location = f"{path.name}:{line_number}"
            level, task, output = release_row(row, location, args.metadata)
            if output["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {output['id']}")
            seen_ids.add(output["id"])
            handles[level].write(
                json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            counts[level] += 1
            task_counts[f"{level}:{task}"] += 1
            metadata = output["metadata"]
            source = str(metadata.get("dataset") or metadata.get("source") or "unspecified")
            source_counts[source] += 1
            media_counts[f"{output['media_type']}_records"] += 1

    actual = {level: counts[level] for level in EXPECTED_COUNTS}
    if args.paper_counts and actual != EXPECTED_COUNTS:
        raise RuntimeError(f"LSI-108K count mismatch: expected {EXPECTED_COUNTS}, got {actual}")
    if args.parquet:
        from datasets import Dataset

        for level, path in paths.items():
            rows = [row for _, _, row in read_rows([path])]
            if rows:
                Dataset.from_list(rows).to_parquet(args.output_dir / f"lsi_{level}.parquet")
    summary = {
        "dataset": "LSI-108K",
        "total_rows": sum(actual.values()),
        "level_counts": actual,
        "task_counts": dict(sorted(task_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "media_counts": dict(sorted(media_counts.items())),
        "reported_sft_rows": sum(
            count
            for key, count in task_counts.items()
            if key.startswith("l2:")
            or (
                key.startswith("l1:")
                and key != "l1:long_horizon_manipulation_program"
            )
        ),
        "media_included": False,
        "metadata_mode": args.metadata,
    }
    assert_private_data_absent(summary, "dataset summary")
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
