#!/usr/bin/env python3
"""Validate SFT or OPD data before launching distributed training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                yield row


def validate_sft(row: dict[str, Any], location: str) -> None:
    dialog = row.get("conversations")
    if not isinstance(dialog, list) or len(dialog) < 2:
        raise ValueError(f"{location}: missing conversations")
    roles = {message.get("from") for message in dialog if isinstance(message, dict)}
    if not {"human", "gpt"}.issubset(roles):
        raise ValueError(f"{location}: conversations require human and gpt turns")


def validate_opd(row: dict[str, Any], location: str) -> None:
    required = ("prompt", "privileged_prompt", "answer", "videos", "task_type")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"{location}: missing {missing}")
    if "<PRIVILEGED_STATE_TRANSITIONS>" in str(row["prompt"]):
        raise ValueError(f"{location}: plain student prompt contains privileged context")
    if "<PRIVILEGED_STATE_TRANSITIONS>" not in str(row["privileged_prompt"]):
        raise ValueError(f"{location}: teacher prompt has no privileged context")
    truth = json.loads(str(row["answer"]))
    if truth.get("reward_family") not in {"choice", "numeric_mra", "multi_choice"}:
        raise ValueError(f"{location}: unsupported reward family")
    if not isinstance(row["videos"], (list, tuple)) or len(row["videos"]) != 1:
        raise ValueError(f"{location}: expected one video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("sft", "opd"), required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--require-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = Counter()
    seen_ids = set()
    for path in args.input:
        for index, row in enumerate(rows(path), 1):
            location = f"{path}:{index}"
            if args.format == "sft":
                validate_sft(row, location)
            else:
                validate_opd(row, location)
            row_id = str(row.get("id") or row.get("source_id") or location)
            if row_id in seen_ids:
                raise ValueError(f"{location}: duplicate id {row_id}")
            seen_ids.add(row_id)
            for field in ("images", "videos"):
                values = row.get(field) or []
                if isinstance(values, str):
                    values = [values]
                counts[field] += len(values)
                if args.require_media:
                    if not args.media_root:
                        raise ValueError("--require-media also requires --media-root")
                    for value in values:
                        candidate = Path(str(value))
                        candidate = candidate if candidate.is_absolute() else args.media_root / candidate
                        if not candidate.is_file():
                            raise FileNotFoundError(f"{location}: missing media {candidate}")
            counts["rows"] += 1
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()

