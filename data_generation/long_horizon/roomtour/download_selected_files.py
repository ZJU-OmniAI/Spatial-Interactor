#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ENDPOINT = "https://hf-mirror.com"


def strip_proxy_env() -> None:
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def api_json(repo_id: str) -> dict[str, Any]:
    url = f"{ENDPOINT}/api/datasets/{repo_id}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "roomtour3d-downloader"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def selected_files(repo_id: str, mode: str) -> list[dict[str, Any]]:
    data = api_json(repo_id)
    out = []
    for item in data.get("siblings", []):
        fn = item["rfilename"]
        if mode == "video":
            if fn.endswith(".zip"):
                out.append(item)
        elif mode == "pose":
            if (
                fn in {"README.md", "p1_train_colmap_trajectory_tour3d.json", "p1_train_reformat.json"}
                or fn.startswith("trajectories/")
                or fn.startswith("geometry/")
                or fn.startswith("colmap_reconstruction/")
            ):
                out.append(item)
        else:
            raise ValueError(mode)
    return out


def summarize(files: list[dict[str, Any]]) -> dict[str, Any]:
    by_top: dict[str, dict[str, float | int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for item in files:
        fn = item["rfilename"]
        top = fn.split("/")[0] if "/" in fn else "."
        by_top[top]["files"] += 1
        by_top[top]["bytes"] += int(item.get("size") or 0)
    return {
        "files": len(files),
        "bytes": sum(int(i.get("size") or 0) for i in files),
        "by_top": by_top,
    }


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def already_ok(local_dir: Path, item: dict[str, Any]) -> bool:
    path = local_dir / item["rfilename"]
    size = item.get("size")
    return path.exists() and (size is None or path.stat().st_size == int(size))


def download_one(repo_id: str, local_dir: Path, item: dict[str, Any], log_path: Path, retries: int = 8) -> dict[str, Any]:
    fn = item["rfilename"]
    if already_ok(local_dir, item):
        return {"file": fn, "status": "exists"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fn,
                repo_type="dataset",
                local_dir=str(local_dir),
                endpoint=ENDPOINT,
                etag_timeout=60,
            )
            if already_ok(local_dir, item):
                return {"file": fn, "status": "downloaded", "path": path}
            return {"file": fn, "status": "size_mismatch", "path": path}
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            log(log_path, f"RETRY file={fn} attempt={attempt} error={last_error}")
            time.sleep(min(600, 20 * attempt))
    return {"file": fn, "status": "failed", "error": last_error}


def run_group(repo_id: str, mode: str, local_dir: Path, workers: int, root_log: Path) -> None:
    group_log = root_log.parent / f"download_{mode}.log"
    files = selected_files(repo_id, mode)
    summary = summarize(files)
    log(root_log, f"GROUP {mode} repo={repo_id} local_dir={local_dir}")
    log(root_log, f"GROUP {mode} summary={json.dumps(summary, ensure_ascii=False, default=dict)}")
    (root_log.parent / f"manifest_{mode}.json").write_text(
        json.dumps({"repo_id": repo_id, "mode": mode, "summary": summary, "files": files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pending = [item for item in files if not already_ok(local_dir, item)]
    log(root_log, f"GROUP {mode} pending={len(pending)} already_ok={len(files)-len(pending)}")
    failures = []
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(download_one, repo_id, local_dir, item, group_log): item for item in pending}
        for fut in futures.as_completed(future_map):
            res = fut.result()
            completed += 1
            if res["status"] in {"failed", "size_mismatch"}:
                failures.append(res)
            if completed % 20 == 0 or res["status"] in {"failed", "size_mismatch"}:
                log(root_log, f"GROUP {mode} progress={completed}/{len(pending)} last={json.dumps(res, ensure_ascii=False)}")
    if failures:
        fail_path = root_log.parent / f"failures_{mode}.json"
        fail_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"{mode} failures: {len(failures)} see {fail_path}")
    log(root_log, f"GROUP {mode} complete")


def main() -> None:
    strip_proxy_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=["all", "pose", "video"], default="all")
    args = ap.parse_args()
    log_dir = args.root / "logs"
    root_log = log_dir / "run.log"
    data_dir = args.root / "roomtour3d"
    video_dir = args.root / "room_tour_video_3fps"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    log(root_log, f"START root={args.root} endpoint={ENDPOINT}")
    log(root_log, f"PROXY HTTP_PROXY={os.environ.get('HTTP_PROXY','')} HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')}")
    if args.only in {"all", "pose"}:
        run_group("roomtour3d/roomtour3d", "pose", data_dir, args.workers, root_log)
    if args.only in {"all", "video"}:
        run_group("roomtour3d/room_tour_video_3fps", "video", video_dir, args.workers, root_log)
    log(root_log, "DOWNLOADS_COMPLETE")


if __name__ == "__main__":
    main()
