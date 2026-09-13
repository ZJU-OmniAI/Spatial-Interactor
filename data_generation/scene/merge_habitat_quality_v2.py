#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded habitat quality v2 audit outputs.")
    parser.add_argument("--asset", required=True, choices=["HSSD", "REP"])
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="v2full")
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    source_by_id = {str(row.get("sample_id") or ""): row for row in source_rows if row.get("sample_id")}

    records: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard = args.output_dir / f"{args.asset.lower()}_quality_filter_{args.tag}_shard{shard_index:02d}of{args.num_shards:02d}.jsonl"
        if not shard.exists():
            raise FileNotFoundError(f"Missing shard output: {shard}")
        records.extend(load_records(shard))

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(record)

    kept_rows = [source_by_id[item["sample_id"]] for item in deduped if item.get("decision") == "keep" and item["sample_id"] in source_by_id]
    rejected = [item for item in deduped if item.get("decision") != "keep"]

    decision_counts = Counter(str(item.get("decision") or "unknown") for item in deduped)
    issue_counts = Counter()
    class_stats = defaultdict(lambda: {"keep": 0, "review": 0, "delete": 0, "total": 0})
    for item in deduped:
        cls = str(item.get("class_id"))
        dec = str(item.get("decision") or "unknown")
        class_stats[cls]["total"] += 1
        if dec in {"keep", "review", "delete"}:
            class_stats[cls][dec] += 1
        for issue in item.get("issues") or []:
            issue_counts[str(issue)] += 1

    stem = f"{args.asset.lower()}_quality_filter_{args.tag}"
    kept_path = args.output_dir / f"{stem}_kept_all.json"
    reject_path = args.output_dir / f"{stem}_rejected.jsonl"
    keep_ids_path = args.output_dir / f"{stem}_keep_sample_ids.txt"
    reject_ids_path = args.output_dir / f"{stem}_reject_sample_ids.txt"
    summary_path = args.output_dir / f"{stem}_summary.json"

    kept_payload = {
        "asset": args.asset,
        "input_json": str(args.input_json),
        "tag": args.tag,
        "qa_count": len(deduped),
        "kept_count": len(kept_rows),
        "rejected_count": len(rejected),
        "data": kept_rows,
    }
    kept_path.write_text(json.dumps(kept_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    reject_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    keep_ids_path.write_text("\n".join(row["sample_id"] for row in kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")
    reject_ids_path.write_text(
        "\n".join(str(item.get("sample_id") or "") for item in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )

    summary = {
        "asset": args.asset,
        "input_json": str(args.input_json),
        "tag": args.tag,
        "num_shards": args.num_shards,
        "qa_count": len(deduped),
        "kept_count": len(kept_rows),
        "rejected_count": len(rejected),
        "kept_ratio": len(kept_rows) / len(deduped) if deduped else 0.0,
        "decision_counts": dict(decision_counts),
        "issue_counts": dict(issue_counts.most_common()),
        "class_stats": dict(sorted(class_stats.items(), key=lambda item: int(item[0]))),
        "outputs": {
            "kept_all": str(kept_path),
            "rejected_jsonl": str(reject_path),
            "keep_sample_ids": str(keep_ids_path),
            "reject_sample_ids": str(reject_ids_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
