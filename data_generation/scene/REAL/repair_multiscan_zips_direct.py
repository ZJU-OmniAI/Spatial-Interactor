#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests


DATASET_ROOT = Path("/path/to/workspace/MultiScan")
STAGE_ROOT = Path("/path/to/workspace/MultiScan_stage")
BASE_URL = "https://hf-mirror.com/datasets/3dlg-hcvc/MultiScan/resolve/main/scans"


def expected_zip_names(dataset_root: Path) -> list[str]:
    scans_txt = dataset_root / "scans" / "scans.txt"
    return [line.strip() + ".zip" for line in scans_txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def zip_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024 * 1024:
        return False
    result = subprocess.run(["zip", "-T", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def download_one(name: str, stage_dir: Path, token: str) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = stage_dir / name
    if zip_ok(out):
        print(f"[stage-ok] reuse {out}", flush=True)
        return out
    tmp = out.with_suffix(".zip.tmp")
    headers = {"Authorization": f"Bearer {token}"}
    resume = tmp.stat().st_size if tmp.exists() else 0
    if resume:
        headers["Range"] = f"bytes={resume}-"

    session = requests.Session()
    session.trust_env = False  # do not use HTTP_PROXY / HTTPS_PROXY / ALL_PROXY
    with session.get(f"{BASE_URL}/{name}", headers=headers, stream=True, timeout=90, allow_redirects=True) as response:
        response.raise_for_status()
        mode = "ab" if resume and response.status_code == 206 else "wb"
        if mode == "wb":
            resume = 0
        total = int(response.headers.get("content-length") or 0) + resume
        done = resume
        last = time.time()
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 10:
                    print(f"[download] {name} {done/1024/1024:.1f}MB/{total/1024/1024:.1f}MB", flush=True)
                    last = now
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing/corrupt MultiScan scene zip files without using proxies.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--names-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("HF token is required via --token or HF_TOKEN")

    scans_dir = args.dataset_root / "scans"
    stage_scans = args.stage_root / "scans"
    if args.names_file:
        names = [line.strip() for line in args.names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = expected_zip_names(args.dataset_root)

    bad = [name for name in names if not zip_ok(scans_dir / name)]
    print(json.dumps({"checked": len(names), "bad": len(bad), "dataset_root": str(args.dataset_root)}, ensure_ascii=False), flush=True)
    for name in bad[:50]:
        print(name, flush=True)
    if len(bad) > 50:
        print(f"... {len(bad) - 50} more", flush=True)
    if args.dry_run or not bad:
        return

    ok_count = 0
    failed: list[dict[str, str]] = []
    for idx, name in enumerate(bad, start=1):
        print(f"[{idx}/{len(bad)}] repairing {name}", flush=True)
        try:
            staged = download_one(name, stage_scans, args.token)
            if not zip_ok(staged):
                raise RuntimeError(f"downloaded file failed zip test: {staged}")
            target = scans_dir / name
            tmp_target = target.with_suffix(".zip.repair")
            shutil.copyfile(staged, tmp_target)
            if not zip_ok(tmp_target):
                tmp_target.unlink(missing_ok=True)
                raise RuntimeError(f"copied file failed zip test: {tmp_target}")
            tmp_target.replace(target)
            ok_count += 1
            print(f"[ok] {name}", flush=True)
        except Exception as exc:
            failed.append({"name": name, "error": repr(exc)})
            print(f"[failed] {name} {repr(exc)}", flush=True)

    summary = {"requested": len(bad), "repaired": ok_count, "failed": failed}
    (args.dataset_root / "repair_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
