#!/usr/bin/env python3
"""Materialize robot images into the LLaMAFactory bundle as normal files.

This intentionally copies file contents only. It does not preserve mtime,
permissions, xattrs, or ownership, because the Dataset2 CIFS mount is slow or
unsupported for some metadata operations.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SRC_ROOT = Path("/path/to/workspace/high_quality_new_interaction_robot_full_20260615_work")
FILELIST = Path("/tmp/high_quality_interaction_robot_files_20260616.txt")
DST_ROOT = Path(
    "/path/to/workspace/llamafactory_sft_bundle_20260608/"
    "data/media/high_quality_interaction_linked_20260616"
)


def copy_contents(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".tmp_copy_{os.getpid()}")
    with src.open("rb") as rf, tmp.open("wb") as wf:
        shutil.copyfileobj(rf, wf, length=1024 * 1024)
    os.replace(tmp, dst)


def copy_one(rel: str) -> str:
    src = SRC_ROOT / rel
    dst = DST_ROOT / rel
    if not src.exists():
        return "missing"
    src_size = src.stat().st_size
    if dst.exists():
        try:
            if dst.stat().st_size == src_size:
                return "skipped"
        except OSError:
            pass
    copy_contents(src, dst)
    return "copied"


def main() -> int:
    rels = [line.strip() for line in FILELIST.read_text().splitlines() if line.strip()]
    total = len(rels)
    workers = int(os.environ.get("MATERIALIZE_WORKERS", "32"))
    copied = 0
    skipped = 0
    missing = []
    start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(copy_one, rel): rel for rel in rels}
        for fut in as_completed(futures):
            done += 1
            rel = futures[fut]
            try:
                status = fut.result()
            except Exception as exc:
                status = f"error:{type(exc).__name__}:{exc}"
            if status == "copied":
                copied += 1
            elif status == "skipped":
                skipped += 1
            elif status == "missing":
                missing.append(rel)
            else:
                missing.append(f"{rel} ({status})")
            if done % 1000 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"progress {done}/{total} copied={copied} skipped={skipped} "
                    f"missing_or_error={len(missing)} workers={workers} rate={rate:.1f}/s",
                    flush=True,
                )
    elapsed = time.time() - start
    if done % 1000 != 0:
        rate = done / elapsed if elapsed else 0.0
        print(
            f"progress {done}/{total} copied={copied} skipped={skipped} "
            f"missing_or_error={len(missing)} workers={workers} rate={rate:.1f}/s",
            flush=True,
        )
    report = {
        "total": total,
        "copied": copied,
        "skipped": skipped,
        "missing": len(missing),
        "workers": workers,
        "dst_root": str(DST_ROOT),
        "elapsed_sec": round(elapsed, 3),
    }
    print(report, flush=True)
    if missing:
        print("missing examples:", missing[:20], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
