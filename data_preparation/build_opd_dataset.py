#!/usr/bin/env python3
"""Join the selected RoomTour-L3 and VSTI records with privileged traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_VSTI_TASKS = ("camera_movement_direction", "camera_displacement")
VSTI_TASK_ALIASES = {
    "camera_movement_direction_interval": "camera_movement_direction",
    "camera_displacement_interval": "camera_displacement",
}
PRIVILEGED_BLOCK = """<PRIVILEGED_STATE_TRANSITIONS>
This training-only trace describes visible local state transitions in chronological order. It is additional visual evidence, not an answer. Use only details relevant to the question and do not mention this trace in the response.

{trace}
</PRIVILEGED_STATE_TRANSITIONS>
"""


def read_jsonl(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    yield path, line_number, row


def messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("conversations") or row.get("messages") or []
    return value if isinstance(value, list) else []


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("value") if "value" in message else message.get("content") or "")


def question(row: dict[str, Any]) -> str:
    dialog = messages(row)
    text = message_text(dialog[0]) if dialog else str(row.get("question") or "")
    text = text.replace("<video>", "")
    text = re.sub(r"\s*Please answer the question using a single word or phrase\.?", "", text, flags=re.I)
    text = re.sub(r"\s*Answer with (?:only )?(?:one |the )?(?:option )?letter(?: only)?\.?", "", text, flags=re.I)
    return text.strip()


def answer_text(row: dict[str, Any]) -> str:
    dialog = messages(row)
    return message_text(dialog[-1]).strip() if dialog else str(row.get("answer") or "").strip()


def task_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(
        row.get("task_type")
        or metadata.get("task_type")
        or metadata.get("original_question_type")
        or "unknown"
    )


def canonical_task_type(row: dict[str, Any]) -> str:
    value = task_type(row)
    return VSTI_TASK_ALIASES.get(value, value)


def videos(row: dict[str, Any]) -> list[str]:
    values = row.get("videos")
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [str(value) for value in values if value]
    value = row.get("video") or (row.get("input") or {}).get("video")
    return [str(value)] if value else []


def path_keys(video: str, data_root: Path) -> set[str]:
    path = Path(video)
    absolute = path if path.is_absolute() else data_root / path
    return {os.path.normpath(str(path)), os.path.normpath(str(absolute.resolve()))}


def canonical_video_key(video: str, data_root: Path) -> str:
    path = Path(video)
    absolute = (path if path.is_absolute() else data_root / path).resolve()
    try:
        return absolute.relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Video is outside --data-root: {absolute}") from exc


def load_traces(paths: list[Path], data_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _, _, row in read_jsonl(paths):
        trace = str(row.get("segmented_video_description") or "").strip()
        if not trace or not (row.get("validation") or {}).get("ok", True):
            continue
        values = [row.get("video"), row.get("video_abs")]
        for value in values:
            if value:
                for key in path_keys(str(value), data_root):
                    result[key] = row
    return result


def choice_letter(text: str) -> str | None:
    stripped = text.strip()
    if re.fullmatch(r"[A-E]", stripped, re.I):
        return stripped.upper()
    matches = re.findall(r"(?:answer|option|choice)\s*(?:is|:|=)\s*([A-E])\b", text, re.I)
    return matches[-1].upper() if matches else None


def reward_spec(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    metadata = row.get("metadata") or {}
    explicit = metadata.get("reward_spec")
    if isinstance(explicit, dict) and explicit.get("reward_family"):
        truth = dict(explicit)
    else:
        fields = metadata.get("correct_options") or (metadata.get("gt") or {}).get("correct_options")
        if isinstance(fields, dict) and fields:
            truth = {
                "reward_family": "multi_choice",
                "fields": {str(key): str(value).upper() for key, value in fields.items()},
            }
        else:
            numeric = None
            for key in ("original_value_m", "value_m", "target_value", "value"):
                if key in metadata:
                    try:
                        numeric = float(metadata[key])
                        break
                    except (TypeError, ValueError):
                        pass
            if numeric is not None:
                truth = {"reward_family": "numeric_mra", "value": numeric, "unit": "m"}
            else:
                letter = choice_letter(answer_text(row))
                if letter is None:
                    raise ValueError("cannot infer a choice, numeric, or multi-field reward target")
                truth = {"reward_family": "choice", "choice_gold": letter}
    truth["task_type"] = task_type(row)
    family = truth["reward_family"]
    if family == "numeric_mra":
        schema = "On the Answer line, output only the non-negative number in meters."
    elif family == "multi_choice":
        fields = ", ".join(sorted((truth.get("fields") or {}).keys()))
        schema = f"On the Answer line, output a compact JSON object with option letters for: {fields}."
    else:
        schema = "On the Answer line, output only the selected option letter."
    return truth, schema


def trace_for(video: str, traces: dict[str, dict[str, Any]], data_root: Path) -> dict[str, Any] | None:
    for key in path_keys(video, data_root):
        if key in traces:
            return traces[key]
    return None


def select_stratified(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < limit:
        raise RuntimeError(f"VSTI subset requests {limit} rows but only {len(rows)} are eligible")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[canonical_task_type(row)].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    total = len(rows)
    allocation = {name: min(len(values), int(limit * len(values) / total)) for name, values in groups.items()}
    remaining = limit - sum(allocation.values())
    while remaining:
        candidates = [name for name, values in groups.items() if allocation[name] < len(values)]
        if not candidates:
            raise RuntimeError("could not complete stratified VSTI allocation")
        candidates.sort(key=lambda name: (-(limit * len(groups[name]) / total - allocation[name]), name))
        for name in candidates:
            if remaining == 0:
                break
            allocation[name] += 1
            remaining -= 1
    selected = [row for name in sorted(groups) for row in groups[name][: allocation[name]]]
    rng.shuffle(selected)
    return selected


def split_name(video_key: str, seed: int, val_fraction: float) -> str:
    digest = hashlib.sha1(f"{seed}:{video_key}".encode()).hexdigest()
    return "val" if int(digest[:12], 16) / float(16**12) < val_fraction else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l3", type=Path, nargs="+", required=True)
    parser.add_argument("--vsti", type=Path, nargs="+", required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vsti-limit", type=int, default=11_000)
    parser.add_argument("--vsti-tasks", default=",".join(DEFAULT_VSTI_TASKS))
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--allow-missing-traces", action="store_true")
    parser.add_argument("--paper-counts", action="store_true")
    parser.add_argument("--jsonl-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    traces = load_traces(args.traces, data_root)
    allowed_vsti_tasks = {value.strip() for value in args.vsti_tasks.split(",") if value.strip()}
    l3_rows = [row for _, _, row in read_jsonl(args.l3)]
    vsti_eligible = [
        row
        for _, _, row in read_jsonl(args.vsti)
        if canonical_task_type(row) in allowed_vsti_tasks
    ]
    vsti_rows = select_stratified(vsti_eligible, args.vsti_limit, args.seed)

    prepared = []
    missing = Counter()
    for source_name, rows in (("lsi_l3", l3_rows), ("vsti_long_horizon", vsti_rows)):
        for source_index, row in enumerate(rows):
            media = videos(row)
            if len(media) != 1:
                raise ValueError(f"{source_name}:{source_index} must reference exactly one video")
            video = media[0]
            trace_row = trace_for(video, traces, data_root)
            if trace_row is None:
                missing[source_name] += 1
                continue
            query = question(row)
            if not query:
                raise ValueError(f"{source_name}:{source_index} has an empty question")
            truth, schema = reward_spec(row)
            prompt = f"<video>\n{query}\n\nOutput requirement: {schema}"
            trace = str(trace_row["segmented_video_description"])
            privileged_prompt = prompt.replace(
                "<video>\n", "<video>\n" + PRIVILEGED_BLOCK.format(trace=trace) + "\n", 1
            )
            video_key = canonical_video_key(video, data_root)
            prepared.append(
                {
                    "prompt": prompt,
                    "privileged_prompt": privileged_prompt,
                    "answer": json.dumps(truth, separators=(",", ":")),
                    "videos": [video_key],
                    "source_id": str(row.get("id") or f"{source_name}:{source_index}"),
                    "source": source_name,
                    "task_type": task_type(row),
                    "reward_family": truth["reward_family"],
                    "trace_id": str(trace_row.get("id") or ""),
                    "_video_key": video_key,
                }
            )
    if missing and not args.allow_missing_traces:
        raise RuntimeError(f"Missing privileged traces: {dict(missing)}")

    if args.paper_counts:
        actual = Counter(row["source"] for row in prepared)
        expected = {"lsi_l3": 10_912, "vsti_long_horizon": 11_000}
        mismatches = {
            name: {"expected": count, "actual": actual[name]}
            for name, count in expected.items()
            if actual[name] != count
        }
        if mismatches:
            raise RuntimeError(
                "Paper OPD mixture count mismatch: " + json.dumps(mismatches, sort_keys=True)
            )

    prepared.sort(key=lambda row: (row["_video_key"], row["source"], row["source_id"]))
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for row in prepared:
        split_rows[split_name(row["_video_key"], args.seed, args.val_fraction)].append(row)
    if not split_rows["val"] and split_rows["train"]:
        fallback = split_rows["train"][-1]["_video_key"]
        split_rows["val"] = [row for row in split_rows["train"] if row["_video_key"] == fallback]
        split_rows["train"] = [row for row in split_rows["train"] if row["_video_key"] != fallback]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        portable = [{key: value for key, value in row.items() if key != "_video_key"} for row in rows]
        jsonl = args.output_dir / f"{split}.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            for row in portable:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        if not args.jsonl_only:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError("pandas and pyarrow are required to write training parquet files") from exc
            pd.DataFrame(portable).to_parquet(args.output_dir / f"{split}.parquet", index=False)

    train_videos = {row["_video_key"] for row in split_rows["train"]}
    val_videos = {row["_video_key"] for row in split_rows["val"]}
    if train_videos & val_videos:
        raise RuntimeError("video-level train/validation leakage detected")
    if args.paper_counts:
        split_counts = {
            split: Counter(row["source"] for row in rows)
            for split, rows in split_rows.items()
        }
        expected_splits = {
            "train": {"lsi_l3": 10_712, "vsti_long_horizon": 10_783},
            "val": {"lsi_l3": 200, "vsti_long_horizon": 217},
        }
        mismatches = {
            split: {
                source: {
                    "expected": count,
                    "actual": split_counts[split][source],
                }
                for source, count in expected.items()
                if split_counts[split][source] != count
            }
            for split, expected in expected_splits.items()
        }
        mismatches = {split: values for split, values in mismatches.items() if values}
        if mismatches:
            raise RuntimeError(
                "Paper OPD split count mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
    summary = {
        "protocol": "selected RoomTour-L3 + VSTI long-horizon records",
        "train_rows": len(split_rows["train"]),
        "val_rows": len(split_rows["val"]),
        "unique_videos": len(train_videos | val_videos),
        "source_counts": dict(Counter(row["source"] for row in prepared)),
        "task_counts": dict(Counter(row["task_type"] for row in prepared)),
        "reward_family_counts": dict(Counter(row["reward_family"] for row in prepared)),
        "vsti_eligible_rows": len(vsti_eligible),
        "vsti_selected_rows": len(vsti_rows),
        "missing_trace_rows": dict(missing),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
