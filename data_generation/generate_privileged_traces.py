#!/usr/bin/env python3
"""Generate question-free four-segment state-transition traces from 32 frames."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a visual-spatial annotator for egocentric indoor video. You receive "
    "32 uniformly sampled frames in chronological order, with no question, answer, "
    "label, or hidden target. Describe concrete visible changes and infer only coarse "
    "camera motion supported by those changes."
)

USER_PROMPT = """Treat the 32 frames as one continuous first-person video. Use exactly these intervals:

[SEGMENT_1] Frames 01-08
[SEGMENT_2] Frames 09-16
[SEGMENT_3] Frames 17-24
[SEGMENT_4] Frames 25-32

For each interval, write one concise natural English paragraph, normally 40-60 words. Establish the visible environment and useful spatial anchors, describe diagnostic visual changes, and end with a cautious coarse camera-motion inference. Focus on concrete visual content rather than decorative wording. Do not invent distances, angles, coordinates, or unseen events. Never mention a dataset, question, answer, label, ground truth, prompt, or metadata.

Use exactly:
[SEGMENT_1] Frames 01-08: paragraph
[SEGMENT_2] Frames 09-16: paragraph
[SEGMENT_3] Frames 17-24: paragraph
[SEGMENT_4] Frames 25-32: paragraph
"""

SEGMENT_RE = re.compile(
    r"\[SEGMENT_(\d+)\]\s+Frames\s+(\d{1,2})\s*[-\u2013]\s*(\d{1,2})\s*:",
    re.IGNORECASE,
)
EXPECTED_SPANS = [(1, 1, 8), (2, 9, 16), (3, 17, 24), (4, 25, 32)]
WRITE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def probe_frame_count(video: Path) -> int | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = (payload.get("streams") or [{}])[0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value and str(value).isdigit() and int(value) > 0:
            return int(value)
    return None


def probe_duration(video: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    return float(subprocess.check_output(command, text=True).strip())


def uniform_indices(length: int, count: int) -> list[int]:
    if count < 2:
        return [max(0, length // 2)]
    return [round(index * max(0, length - 1) / (count - 1)) for index in range(count)]


def extract_frames(video: Path, cache_root: Path, item_id: str, width: int) -> list[Path]:
    output_dir = cache_root / item_id
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"frame_{index:02d}.jpg" for index in range(1, 33)]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return expected

    frame_count = probe_frame_count(video)
    if frame_count and frame_count >= 32:
        indices = uniform_indices(frame_count, 32)
        expression = "+".join(f"eq(n\\,{index})" for index in indices)
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select={expression},scale={width}:-2",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            str(output_dir / "frame_%02d.jpg"),
        ]
        subprocess.run(command, check=True)
    else:
        duration = max(probe_duration(video), 0.001)
        for index, timestamp in enumerate(
            [value * duration / 32.0 for value in range(32)], 1
        ):
            command = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:-2",
                "-q:v",
                "2",
                str(output_dir / f"frame_{index:02d}.jpg"),
            ]
            subprocess.run(command, check=True)

    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        raise RuntimeError(f"Could not extract exactly 32 frames from {video}")
    return expected


def image_part(path: Path, detail: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": detail},
    }


def endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def call_api(
    base_url: str,
    api_key: str,
    model: str,
    frames: list[Path],
    detail: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[str, dict[str, Any], str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": USER_PROMPT}]
    for index, frame in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {index:02d}:"})
        content.append(image_part(frame, detail))
    session_id = uuid.uuid4().hex
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "session_id": session_id,
    }
    request = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_preview = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {body_preview}") from exc
    text = str(payload["choices"][0]["message"]["content"]).strip()
    return text, payload, session_id


def validate_trace(text: str) -> dict[str, Any]:
    matches = list(SEGMENT_RE.finditer(text))
    spans = [(int(a), int(b), int(c)) for a, b, c in SEGMENT_RE.findall(text)]
    word_counts = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        paragraph = text[match.end() : stop].strip()
        word_counts.append(len(paragraph.split()))
    return {
        "ok": spans == EXPECTED_SPANS and len(word_counts) == 4 and all(25 <= n <= 85 for n in word_counts),
        "spans": spans,
        "paragraph_word_counts": word_counts,
        "word_count": len(text.split()),
    }


def done_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    return {
        str(row["id"])
        for row in read_jsonl(output)
        if row.get("id") and row.get("segmented_video_description")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("VLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("VLM_MODEL", ""))
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--frame-width", type=int, default=768)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe are required")
    api_key = os.getenv(args.api_key_env, "")
    if not api_key or not args.base_url or not args.model:
        raise SystemExit(
            f"Set {args.api_key_env}, VLM_BASE_URL, and VLM_MODEL (or pass --base-url/--model)"
        )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    data_root = args.data_root.resolve()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.frame_cache.mkdir(parents=True, exist_ok=True)
    errors = args.output.with_suffix(args.output.suffix + ".errors.jsonl")
    raw_dir = args.output.parent / "raw_responses"
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    completed = done_ids(args.output)
    rows = [
        row
        for index, row in enumerate(read_jsonl(args.manifest))
        if index % args.num_shards == args.shard_index and str(row.get("id")) not in completed
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    def process(row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        video = (data_root / str(row["video"])).resolve()
        if not video.is_relative_to(data_root):
            raise ValueError(f"Video escapes --data-root: {row['video']}")
        if not video.is_file():
            raise FileNotFoundError(video)
        frames = extract_frames(video, args.frame_cache, str(row["id"]), args.frame_width)
        last_error: Exception | None = None
        for retry in range(args.retries + 1):
            try:
                text, raw, session_id = call_api(
                    args.base_url,
                    api_key,
                    args.model,
                    frames,
                    args.image_detail,
                    args.max_tokens,
                    args.temperature,
                    args.timeout,
                )
                validation = validate_trace(text)
                if not validation["ok"]:
                    raise ValueError(f"invalid trace format: {validation}")
                if args.save_raw:
                    (raw_dir / f"{row['id']}.json").write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return {
                    **row,
                    "model": args.model,
                    "session_id_hash": hashlib.sha1(session_id.encode()).hexdigest()[:12],
                    "segmented_video_description": text,
                    "validation": validation,
                    "elapsed_sec": round(time.time() - started, 3),
                }
            except Exception as exc:
                last_error = exc
                if retry < args.retries:
                    time.sleep(min(60.0, 2**retry + random.random()))
        raise RuntimeError(str(last_error))

    total = len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        finished = 0
        for future in concurrent.futures.as_completed(futures):
            finished += 1
            row = futures[future]
            try:
                result = future.result()
                append_jsonl(args.output, result)
                print(f"[{finished}/{total}] ok {row['id']}", flush=True)
            except Exception as exc:
                append_jsonl(errors, {"id": row.get("id"), "video": row.get("video"), "error": repr(exc)})
                print(f"[{finished}/{total}] error {row.get('id')}: {exc}", flush=True)


if __name__ == "__main__":
    main()
