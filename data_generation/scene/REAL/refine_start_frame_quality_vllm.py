#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from filter_pose_gt_samples_quality import floor_bias_score, start_frame_path  # noqa: E402


ROOT = Path("/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples")
BASE_DIR = Path("/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples_quality_filtered_v1_loose")
MODEL = Path("/path/to/workspace/models/Qwen3.5-9B")
TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
    "motion_family_discrimination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine REAL kept samples by deleting floor/poster/bad start frames with VLM.")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--bias-threshold", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [replace_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, path_map) for key, item in value.items()}
    return value


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def build_prompt(image_path: str) -> dict[str, Any]:
    prompt = (
        "<|im_start|>system\n"
        "You are a strict first-frame quality filter for embodied visual QA. "
        "Do not output reasoning, markdown, or <think>. Output only one compact JSON object."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Judge only this single first frame.\n"
        "Delete if the camera mainly points at floor/ground, or if a poster/sign/paper/magazine/label/object lying on the floor dominates the beginning view, "
        "or if this is an uninformative too-close start frame that mostly shows a partial object/surface rather than a normal room view.\n"
        "Keep normal room-view starts.\n"
        "Return JSON with keys: decision, tags, reason.\n"
        "decision: keep or delete\n"
        "tags: array chosen from [\"floor_dominant_start\", \"poster_or_sign_on_floor\", \"uninformative_closeup_start\", \"normal_start\"]\n"
        "reason: <= 18 words.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": [image_path]}}


def parse_output(text: str) -> tuple[str, list[str], str]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            decision = str(payload.get("decision") or "keep").strip().lower()
            tags = [str(item) for item in payload.get("tags") or [] if item]
            reason = str(payload.get("reason") or "").strip()
            if decision not in {"keep", "delete"}:
                decision = "keep"
            return decision, tags, reason
        except Exception:
            pass
    lowered = text.lower()
    return ("delete" if "delete" in lowered else "keep"), [], text[:120]


def candidate_worker(item: tuple[str, str, str]) -> dict[str, Any]:
    task, qa_id, frame_path = item
    bias, parts = floor_bias_score(frame_path)
    return {
        "task": task,
        "qa_id": qa_id,
        "start_frame": frame_path,
        "floor_bias": float(bias),
        "top_ratio": float(parts["top_ratio"]),
        "bottom_ratio": float(parts["bottom_ratio"]),
    }


def collect_candidates(source_root: Path, keep_ids: set[str], workers: int, bias_threshold: float) -> list[dict[str, Any]]:
    items: list[tuple[str, str, str]] = []
    for task in TASKS:
        rows = read_json(source_root / task / "qa_data.json")
        for row in rows:
            qa_id = str(row["qa_id"])
            if qa_id in keep_ids:
                items.append((task, qa_id, start_frame_path(row)))
    candidates: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(candidate_worker, items, chunksize=64):
            if result["floor_bias"] >= bias_threshold:
                candidates.append(result)
    candidates.sort(key=lambda item: item["floor_bias"], reverse=True)
    return candidates


def build_output_dataset(
    source_root: Path,
    keep_ids: set[str],
    output_dir: Path,
    copy_mode: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in TASKS:
        out_task_dir = output_dir / task
        out_task_dir.mkdir(parents=True, exist_ok=True)
        rows = read_json(source_root / task / "qa_data.json")
        rewritten: list[dict[str, Any]] = []
        for row in rows:
            qa_id = str(row["qa_id"])
            if qa_id not in keep_ids:
                continue
            sample_dir = out_task_dir / qa_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            path_map: dict[str, str] = {}
            for old_path in row["input"]["frame_paths"]:
                src = Path(old_path)
                dst = sample_dir / src.name
                copy_or_link(src, dst, copy_mode)
                path_map[str(src)] = str(dst)
            new_row = deepcopy(row)
            new_row["input"] = replace_paths(new_row["input"], path_map)
            rewritten.append(new_row)
        write_json(out_task_dir / "qa_data.json", rewritten)
        counts[task] = len(rewritten)
    return counts


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_keep_path = args.base_dir / "pose_gt_samples_quality_filter_v1_loose_keep_qa_ids.txt"
    base_keep_ids = {line.strip() for line in base_keep_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    candidates = collect_candidates(args.source_root, base_keep_ids, args.workers, args.bias_threshold)
    candidate_path = args.output_dir / "start_frame_vlm_candidates.json"
    write_json(candidate_path, candidates)

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [build_prompt(item["start_frame"]) for item in candidates]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)

    audit_records: list[dict[str, Any]] = []
    delete_ids: set[str] = set()
    for item, output in zip(candidates, outputs):
        text = output.outputs[0].text if output.outputs else ""
        decision, tags, reason = parse_output(text)
        record = {
            **item,
            "decision": decision,
            "tags": tags,
            "reason": reason,
            "raw_text": text,
        }
        audit_records.append(record)
        if decision == "delete":
            delete_ids.add(item["qa_id"])

    refined_keep_ids = base_keep_ids - delete_ids
    copied_counts = build_output_dataset(args.source_root, refined_keep_ids, args.output_dir, args.copy_mode)

    keep_id_path = args.output_dir / "pose_gt_samples_quality_filter_v1_loose_refined_keep_qa_ids.txt"
    reject_id_path = args.output_dir / "pose_gt_samples_quality_filter_v1_loose_refined_reject_qa_ids.txt"
    audit_jsonl = args.output_dir / "start_frame_vlm_audit.jsonl"
    kept_json = args.output_dir / "pose_gt_samples_quality_filter_v1_loose_refined_kept_all.json"
    summary_json = args.output_dir / "pose_gt_samples_quality_filter_v1_loose_refined_summary.json"

    keep_id_path.write_text("\n".join(sorted(refined_keep_ids)) + ("\n" if refined_keep_ids else ""), encoding="utf-8")
    reject_id_path.write_text("\n".join(sorted(delete_ids)) + ("\n" if delete_ids else ""), encoding="utf-8")
    write_jsonl(audit_jsonl, audit_records)

    kept_rows: list[dict[str, Any]] = []
    for task in TASKS:
        kept_rows.extend(read_json(args.output_dir / task / "qa_data.json"))
    write_json(
        kept_json,
        {
            "base_dir": str(args.base_dir),
            "output_dir": str(args.output_dir),
            "qa_count": len(base_keep_ids),
            "candidate_count": len(candidates),
            "deleted_by_start_audit": len(delete_ids),
            "kept_count": len(kept_rows),
            "data": kept_rows,
        },
    )

    by_tag: dict[str, int] = {}
    for record in audit_records:
        for tag in record["tags"]:
            by_tag[tag] = by_tag.get(tag, 0) + 1

    summary = {
        "base_keep_count": len(base_keep_ids),
        "candidate_count": len(candidates),
        "deleted_by_start_audit": len(delete_ids),
        "refined_keep_count": len(refined_keep_ids),
        "refined_keep_ratio_vs_original_keep": round(len(refined_keep_ids) / max(len(base_keep_ids), 1), 6),
        "bias_threshold": args.bias_threshold,
        "copied_counts": copied_counts,
        "tag_counts": by_tag,
        "outputs": {
            "candidate_json": str(candidate_path),
            "audit_jsonl": str(audit_jsonl),
            "keep_ids": str(keep_id_path),
            "reject_ids": str(reject_id_path),
            "kept_all_json": str(kept_json),
            "summary_json": str(summary_json),
        },
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
