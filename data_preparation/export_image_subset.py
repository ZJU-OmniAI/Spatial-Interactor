#!/usr/bin/env python3
"""Embed original images in bounded Parquet shards for Hugging Face Datasets."""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image, List, Value
from PIL import Image as PILImage

from assign_curriculum_levels import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-mb", type=int, default=192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output_dir.glob("*.parquet"))
    if existing and not args.resume:
        raise SystemExit("Output directory already contains Parquet shards; use a new directory.")
    root = args.media_root.resolve()
    features = Features({
        "id": Value("string"),
        "images": List(Image()),
        "conversations": List({"from": Value("string"), "value": Value("string")}),
        "metadata": {
            "curriculum_level": Value("string"), "task_type": Value("string"),
            "source": Value("string"), "scene": Value("string"),
            "used_in_reported_sft": Value("bool"),
        },
    })
    rows = [row for _, _, row in read_rows(args.input)
            if row["media_type"] == "image" and row["metadata"]["source"] in args.source]
    if not rows:
        raise SystemExit("No matching image records.")

    def embed(row):
        images = []
        for relative in row["media_paths"]:
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image escapes media root: {relative}")
            data = path.read_bytes()
            with PILImage.open(io.BytesIO(data)) as image:
                image.verify()
            images.append({"bytes": data, "path": path.name})
        return {"id": row["id"], "images": images,
                "conversations": row["conversations"], "metadata": row["metadata"]}

    writer = None
    shard_bytes = 0
    shard_number = 0
    total_bytes = 0
    image_count = 0
    completed = 0
    for path in existing:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            previous = batch.to_pylist()
            expected = rows[completed:completed + len(previous)]
            if [r["id"] for r in previous] != [r["id"] for r in expected]:
                raise ValueError("Existing shard does not match the selected input order.")
            for actual, original in zip(previous, expected):
                if actual["conversations"] != original["conversations"] or actual["metadata"] != original["metadata"]:
                    raise ValueError("Existing shard annotations differ from input.")
            completed += len(previous)
            image_count += sum(len(row["images"]) for row in previous)
            total_bytes += sum(len(image["bytes"]) for row in previous for image in row["images"])
        shard_number += 1
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(completed, len(rows), 64):
                batch = list(executor.map(embed, rows[offset:offset + 64]))
                if writer is None:
                    path = args.output_dir / f"train-{shard_number:05d}.parquet"
                    writer = pq.ParquetWriter(path, features.arrow_schema, compression="zstd")
                    shard_number += 1
                    shard_bytes = 0
                writer.write_table(pa.Table.from_pylist(batch, schema=features.arrow_schema))
                size = sum(len(image["bytes"]) for row in batch for image in row["images"])
                image_count += sum(len(row["images"]) for row in batch)
                shard_bytes += size
                total_bytes += size
                if shard_bytes >= args.shard_mb * 1024 * 1024:
                    writer.close()
                    writer = None
                print(f"Embedded {min(offset + 64, len(rows))}/{len(rows)} records; "
                      f"{total_bytes / 2**30:.2f} GiB", flush=True)
    finally:
        if writer is not None:
            writer.close()
    summary = {"rows": len(rows), "image_references": image_count,
               "unique_images": len({p for row in rows for p in row["media_paths"]}),
               "original_image_bytes": total_bytes, "shards": shard_number,
               "sources": args.source}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
