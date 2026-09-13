#!/usr/bin/env python3
import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_URL = "https://hf-mirror.com/datasets/youliangtan/bridge_dataset/resolve/main/1.0.0"
NAME_TEMPLATE = "bridge_dataset-train.tfrecord-{idx:05d}-of-01024"
PROXY_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}


def clean_env() -> dict:
    env = os.environ.copy()
    for key in PROXY_KEYS:
        env.pop(key, None)
    return env


ENV = clean_env()


def run_cmd(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=ENV,
    )


def parse_content_length(headers: str) -> int | None:
    vals = []
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                vals.append(int(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
    return vals[-1] if vals else None


def remote_size(url: str, attempts: int = 4) -> int:
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = run_cmd(
                [
                    "curl",
                    "-sSIL",
                    "-f",
                    "--noproxy",
                    "*",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "120",
                    url,
                ],
                timeout=150,
            )
            if proc.returncode == 0:
                size = parse_content_length(proc.stdout + "\n" + proc.stderr)
                if size and size > 0:
                    return size
                last_err = "no content-length"
            else:
                last_err = proc.stderr.strip()[-400:]
        except Exception as exc:
            last_err = repr(exc)
        time.sleep(min(20, 3 * attempt))
    raise RuntimeError(last_err or "failed to read remote size")


def download_one(idx: int, out_dir: Path) -> dict:
    name = NAME_TEMPLATE.format(idx=idx)
    url = f"{BASE_URL}/{name}"
    dst = out_dir / name
    t0 = time.time()
    expected = remote_size(url)
    before = dst.stat().st_size if dst.exists() else 0

    if before == expected and expected > 0:
        return {
            "idx": idx,
            "name": name,
            "status": "skip_complete",
            "expected": expected,
            "before": before,
            "after": before,
            "seconds": round(time.time() - t0, 2),
        }

    if before > expected:
        bad = dst.with_suffix(dst.suffix + f".oversize.{int(time.time())}.bad")
        dst.rename(bad)
        before = 0

    cmd = [
        "curl",
        "-fL",
        "--noproxy",
        "*",
        "--retry",
        "20",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "180",
        "-C",
        "-",
        "-o",
        str(dst),
        url,
    ]
    proc = run_cmd(cmd, timeout=3600)
    after = dst.stat().st_size if dst.exists() else 0
    if proc.returncode != 0:
        return {
            "idx": idx,
            "name": name,
            "status": "curl_failed",
            "expected": expected,
            "before": before,
            "after": after,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip()[-1200:],
            "seconds": round(time.time() - t0, 2),
        }
    if after != expected:
        return {
            "idx": idx,
            "name": name,
            "status": "size_mismatch",
            "expected": expected,
            "before": before,
            "after": after,
            "stderr": proc.stderr.strip()[-1200:],
            "seconds": round(time.time() - t0, 2),
        }
    return {
        "idx": idx,
        "name": name,
        "status": "downloaded" if before == 0 else "resumed",
        "expected": expected,
        "before": before,
        "after": after,
        "seconds": round(time.time() - t0, 2),
    }


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/path/to/workspace/BridgeDataV2_full_curl/1.0.0"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--log", type=Path, default=Path("/path/to/workspace/bridgedata_v2_full_resumable_download_20260616.jsonl"))
    ap.add_argument("--marker", type=Path, default=Path("/path/to/workspace/bridgedata_v2_full_download_complete_20260616.ok"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.marker.unlink(missing_ok=True)

    pending = list(range(1024))
    final_ok: set[int] = set()
    started_at = time.time()
    print(f"[download] starting BridgeDataV2 full download: {len(pending)} shards, workers={args.workers}", flush=True)

    for round_id in range(1, args.rounds + 1):
        if not pending:
            break
        print(f"[download] round {round_id}/{args.rounds}: pending={len(pending)}", flush=True)
        next_pending: list[int] = []
        completed_this_round = 0
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_to_idx = {ex.submit(download_one, idx, args.out_dir): idx for idx in pending}
            for fut in futures.as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    row = {"idx": idx, "name": NAME_TEMPLATE.format(idx=idx), "status": "exception", "error": repr(exc)}
                row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                row["round"] = round_id
                append_log(args.log, row)
                status = row.get("status")
                if status in {"skip_complete", "downloaded", "resumed"}:
                    final_ok.add(idx)
                    completed_this_round += 1
                else:
                    next_pending.append(idx)
                if len(final_ok) % 25 == 0 or status not in {"skip_complete", "downloaded", "resumed"}:
                    print(
                        f"[download] ok={len(final_ok)}/1024 last={idx:05d} status={status} pending_next={len(next_pending)}",
                        flush=True,
                    )
        pending = sorted(set(next_pending) - final_ok)
        print(f"[download] round {round_id} done: ok={len(final_ok)}/1024, retry_pending={len(pending)}", flush=True)
        if pending:
            time.sleep(20)

    if len(final_ok) != 1024:
        print(f"[download] incomplete: ok={len(final_ok)}/1024, failed={len(pending)}", flush=True)
        return 2

    total_size = sum((args.out_dir / NAME_TEMPLATE.format(idx=i)).stat().st_size for i in range(1024))
    marker = {
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started_at, 2),
        "out_dir": str(args.out_dir),
        "shards": 1024,
        "total_size": total_size,
        "total_size_gib": round(total_size / (1024**3), 3),
        "log": str(args.log),
    }
    args.marker.write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[download] complete: marker={args.marker} size={marker['total_size_gib']} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
