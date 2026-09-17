#!/usr/bin/env python3
"""Package licensed video clips once, preserving annotation-relative paths."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from assign_curriculum_levels import read_rows


def read_video(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe media path: {relative}")
    parent = (root / path.parent).resolve()
    if not parent.is_relative_to(root):
        raise ValueError(f"Media directory escapes root: {relative}")
    with open(parent / path.name, "rb",
              opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as handle:
        data = handle.read()
    # MP4 metadata may follow the frames; ffprobe needs a seekable input.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temporary:
        temporary.write(data)
        temporary.flush()
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", temporary.name],
            capture_output=True, check=True, timeout=30,
        )
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams or min(streams[0].get("width", 0), streams[0].get("height", 0)) <= 0:
        raise ValueError(f"No readable video stream: {relative}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-mb", type=int, default=512)
    args = parser.parse_args()
    if min(args.workers, args.shard_mb) <= 0:
        parser.error("workers and shard size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty; incomplete exports are not reused.")
    rows = [r for _, _, r in read_rows(args.input)
            if r["media_type"] == "video" and r["metadata"]["source"] == args.source]
    paths = sorted({p for r in rows for p in r["media_paths"]})
    if not paths:
        parser.error("No matching videos")
    root = args.media_root.resolve()
    archive = None
    manifest = []
    shard = 0
    size = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(paths), args.workers):
                batch = paths[offset:offset + args.workers]
                for relative, data in zip(batch, pool.map(lambda p: read_video(root, p), batch)):
                    if archive is None or size >= args.shard_mb * 2**20:
                        if archive is not None:
                            archive.close()
                            partial.replace(final)
                        final = args.output_dir / f"clips-{shard:05d}.tar"
                        partial = final.with_suffix(".tar.partial")
                        archive = tarfile.open(partial, "w")
                        shard += 1
                        size = 0
                    entry = tarfile.TarInfo(relative)
                    entry.size = len(data)
                    entry.mode = 0o644
                    archive.addfile(entry, io.BytesIO(data))
                    manifest.append({"path": relative, "archive": final.name,
                                     "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                    size += len(data)
                print(f"Validated and packed {len(manifest)}/{len(paths)} clips", flush=True)
        archive.close()
        archive = None
        partial.replace(final)
    finally:
        if archive is not None:
            archive.close()
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in manifest))
    summary = {"source": args.source, "qa_rows": len(rows), "unique_videos": len(paths),
               "bytes": sum(v["bytes"] for v in manifest), "shards": shard}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
