#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ID = "3dlg-hcvc/MultiScan"
DEFAULT_DATASET_ROOT = Path("/path/to/workspace/MultiScan")


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])


def remote_scan_files(repo_id: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(path for path in files if path.startswith("scans/scene_") and path.endswith(".zip"))


def existing_scan_names(dataset_root: Path) -> set[str]:
    scans_dir = dataset_root / "scans"
    if not scans_dir.exists():
        return set()
    return {path.name for path in scans_dir.glob("scene_*.zip") if path.is_file()}


def copy_missing_from_stage(stage_root: Path, dataset_root: Path, remote_paths: list[str]) -> tuple[int, int]:
    scans_dir = dataset_root / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for remote_path in remote_paths:
        name = Path(remote_path).name
        src = stage_root / remote_path
        dst = scans_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        if not src.exists() or src.stat().st_size == 0:
            print(f"[WARN] missing staged file: {src}")
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        copied += 1
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only missing MultiScan scene zip files into the current directory first, then move them into Dataset2."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage-root", type=Path, default=Path.cwd() / "MultiScan_stage")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_huggingface_hub()

    os.environ["HF_ENDPOINT"] = args.endpoint
    args.stage_root.mkdir(parents=True, exist_ok=True)
    (args.dataset_root / "scans").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] repo: {args.repo_id}")
    print(f"[INFO] endpoint: {args.endpoint}")
    print(f"[INFO] stage root: {args.stage_root}")
    print(f"[INFO] dataset root: {args.dataset_root}")

    remote_paths = remote_scan_files(args.repo_id, args.token)
    existing = existing_scan_names(args.dataset_root)
    missing = [path for path in remote_paths if Path(path).name not in existing]

    print(f"[INFO] remote scan zip files: {len(remote_paths)}")
    print(f"[INFO] existing in Dataset2: {len(existing)}")
    print(f"[INFO] missing to download: {len(missing)}")
    if args.dry_run:
        for path in missing[:50]:
            print(path)
        if len(missing) > 50:
            print(f"[INFO] ... {len(missing) - 50} more")
        return
    if not missing:
        print("[INFO] nothing to download")
        return

    from huggingface_hub import snapshot_download

    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        end = start + len(batch)
        print(f"[INFO] downloading batch {start + 1}-{end}/{len(missing)}")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.stage_root),
            local_dir_use_symlinks=False,
            allow_patterns=batch,
            token=args.token,
            resume_download=True,
        )
        copied, skipped = copy_missing_from_stage(args.stage_root, args.dataset_root, batch)
        print(f"[INFO] copied to Dataset2: {copied}, skipped existing: {skipped}")

    final_existing = existing_scan_names(args.dataset_root)
    still_missing = [path for path in remote_paths if Path(path).name not in final_existing]
    print(f"[INFO] final existing in Dataset2: {len(final_existing)}")
    print(f"[INFO] still missing: {len(still_missing)}")
    if still_missing:
        print("[WARN] first missing examples:")
        for path in still_missing[:20]:
            print(path)


if __name__ == "__main__":
    main()
