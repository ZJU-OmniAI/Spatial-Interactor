#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
    "motion_family_discrimination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-pass strict answerability refinement for REAL QA."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/path/to/workspace/models/Qwen3.5-9B"))
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-task", choices=TASKS, default=None)
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


def copy_sample_dir(row: dict[str, Any], out_task_dir: Path) -> tuple[dict[str, Any], Path]:
    qa_id = str(row["qa_id"])
    src_paths = [Path(p) for p in row["input"]["frame_paths"]]
    src_sample_dir = src_paths[0].parent
    dst_sample_dir = out_task_dir / qa_id
    if dst_sample_dir.exists():
        shutil.rmtree(dst_sample_dir)
    shutil.copytree(src_sample_dir, dst_sample_dir)
    path_map = {str(src): str(dst_sample_dir / src.name) for src in src_paths}
    new_row = deepcopy(row)
    new_row["input"] = replace_paths(new_row["input"], path_map)
    return new_row, dst_sample_dir


def task_instruction(task: str) -> str:
    if task == "action_inference":
        return (
            "Keep only if the exact motion direction or action family is visually obvious from the two frames. "
            "Delete if the answer depends on guessing, if shared anchors are too weak, or if the view is mostly floor, wall, ceiling, or textureless surfaces."
        )
    if task == "movement_sequence_sorting":
        return (
            "Keep only if the temporal order can be recovered from a clear monotonic visual progression with stable landmarks. "
            "Delete if the frames are too similar, too discontinuous, or if the order would be mostly a guess."
        )
    if task == "movement_degree_comparison":
        return (
            "Keep only if a strong vision-language model can confidently tell which candidate shows larger visible motion. "
            "Delete if the comparison relies on discontinuous jumps, missing overlap, or visually weak evidence."
        )
    if task == "motion_family_discrimination":
        return (
            "Keep only if the model can confidently judge whether the motion is mainly translation or mainly rotation from image evidence alone. "
            "Delete if the pair has too little shared structure, too much floor or wall, or if the dominant family is not strongly grounded."
        )
    return "Keep only if the question is clearly answerable from the frames."


def issue_set(task: str) -> list[str]:
    shared = [
        "unanswerable_due_to_large_gap",
        "unrelated_or_discontinuous_views",
        "insufficient_shared_anchors",
        "dominant_floor_wall_ceiling",
        "too_subtle_or_visually_weak",
        "question_not_grounded_in_frames",
        "hard_to_answer_confidently",
    ]
    if task == "action_inference":
        return shared + ["ambiguous_action_direction"]
    if task == "movement_sequence_sorting":
        return shared + ["ambiguous_sequence_order", "frames_too_similar"]
    if task == "movement_degree_comparison":
        return shared + ["ambiguous_magnitude_comparison"]
    if task == "motion_family_discrimination":
        return shared + ["ambiguous_motion_family"]
    return shared


def frame_labels(row: dict[str, Any]) -> list[str]:
    data = row["input"]
    if "first_frame" in data:
        labels = ["Start"]
        for key in ["A", "B", "C"]:
            if key in (data.get("candidate_frames") or {}):
                labels.append(key)
        return labels
    if "start_frame" in data:
        labels = ["Start"]
        if "frame_A" in data:
            labels.append("A")
        if "frame_B" in data:
            labels.append("B")
        if "candidate_frames" in data:
            for key in ["A", "B", "C"]:
                if key in data["candidate_frames"]:
                    labels.append(key)
        return labels
    if "frame_A" in data and "frame_B" in data:
        return ["A", "B"]
    return [f"Frame{i+1}" for i in range(len(data.get("frame_paths") or []))]


def build_prompt(task: str, row: dict[str, Any]) -> dict[str, Any]:
    frame_paths = list((row.get("input") or {}).get("frame_paths") or [])
    frame_markers = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in frame_paths)
    labels = ", ".join(frame_labels(row))
    issues = issue_set(task)
    prompt = (
        "<|im_start|>system\n"
        "You are a strict multimodal dataset auditor for embodied-motion QA. "
        "Judge only answerability quality. Do not output reasoning, markdown, or <think>. "
        "Output exactly one compact JSON object.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{frame_markers}\n"
        "Decide whether this sample should remain in a benchmark that wants only questions a strong VLM can answer from the images with high confidence.\n"
        "The sample should be deleted if the visible content is not enough, if the question feels meaningless because the frames jump too far, "
        "if the overlap is too small, or if the answer would depend on guessing.\n"
        f"Task: {task}\n"
        f"Frame labels in order: {labels}\n"
        f"Question: {row.get('question', '')}\n"
        f"Ground-truth answer string: {row.get('answer', '')}\n"
        f"Policy: {task_instruction(task)}\n"
        f"Allowed issue names: {issues}\n"
        "Return JSON with keys: decision, confidence, issues, reason.\n"
        "decision must be keep or delete.\n"
        "confidence must be an integer from 1 to 5, where 5 means highly answerable.\n"
        "issues must be a JSON array using only the allowed issue names.\n"
        "reason must be one short sentence under 18 words.\n"
        "Use keep only if confidence is 4 or 5.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": frame_paths}}


def parse_output(text: str) -> tuple[str, int, list[str], str]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            decision = str(payload.get("decision") or "delete").strip().lower()
            confidence = int(payload.get("confidence") or 1)
            issues = [str(item) for item in payload.get("issues") or [] if item]
            reason = str(payload.get("reason") or "").strip()
            if decision not in {"keep", "delete"}:
                decision = "delete"
            confidence = max(1, min(confidence, 5))
            if decision == "keep" and confidence < 4:
                decision = "delete"
            return decision, confidence, issues, reason
        except Exception:
            pass
    lowered = text.lower()
    decision = "keep" if "keep" in lowered and "delete" not in lowered else "delete"
    return decision, 1, [], text[:120]


def load_rows(input_root: Path, tasks: list[str]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for task in tasks:
        rows = read_json(input_root / task / "qa_data.json")
        for row in rows:
            items.append((task, row))
    return items


def main() -> int:
    args = parse_args()
    tasks = [args.only_task] if args.only_task else list(TASKS)

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    items = load_rows(args.input_root, tasks)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [build_prompt(task, row) for task, row in items]
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)

    audit_records: list[dict[str, Any]] = []
    keep_ids: set[str] = set()
    reject_ids: set[str] = set()
    issue_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    task_counts: dict[str, dict[str, int]] = {task: {"total": 0, "keep": 0, "reject": 0} for task in tasks}

    for (task, row), output in zip(items, outputs):
        qa_id = str(row["qa_id"])
        text = output.outputs[0].text if output.outputs else ""
        decision, confidence, issues, reason = parse_output(text)
        task_counts[task]["total"] += 1
        confidence_counts[str(confidence)] = confidence_counts.get(str(confidence), 0) + 1
        if decision == "keep":
            task_counts[task]["keep"] += 1
            keep_ids.add(qa_id)
        else:
            task_counts[task]["reject"] += 1
            reject_ids.add(qa_id)
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        audit_records.append(
            {
                "qa_id": qa_id,
                "task": task,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "decision": decision,
                "confidence": confidence,
                "issues": issues,
                "reason": reason,
                "raw_text": text,
            }
        )

    copied_counts: dict[str, int] = {}
    kept_all: list[dict[str, Any]] = []
    for task in tasks:
        source_rows = read_json(args.input_root / task / "qa_data.json")
        out_task_dir = args.output_root / task
        out_task_dir.mkdir(parents=True, exist_ok=True)
        kept_rows: list[dict[str, Any]] = []
        for row in source_rows:
            qa_id = str(row["qa_id"])
            if qa_id not in keep_ids:
                continue
            new_row, _ = copy_sample_dir(row, out_task_dir)
            kept_rows.append(new_row)
            kept_all.append(new_row)
        copied_counts[task] = len(kept_rows)
        write_json(out_task_dir / "qa_data.json", kept_rows)

    keep_id_path = args.output_root / "answerability_keep_qa_ids.txt"
    reject_id_path = args.output_root / "answerability_reject_qa_ids.txt"
    audit_path = args.output_root / "answerability_audit.jsonl"
    kept_json_path = args.output_root / "answerability_kept_all.json"
    summary_path = args.output_root / "answerability_summary.json"

    keep_id_path.write_text("\n".join(sorted(keep_ids)) + ("\n" if keep_ids else ""), encoding="utf-8")
    reject_id_path.write_text("\n".join(sorted(reject_ids)) + ("\n" if reject_ids else ""), encoding="utf-8")
    write_jsonl(audit_path, audit_records)
    write_json(
        kept_json_path,
        {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "qa_count": len(items),
            "kept_count": len(kept_all),
            "rejected_count": len(reject_ids),
            "data": kept_all,
        },
    )
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "qa_count": len(items),
        "kept_count": len(kept_all),
        "rejected_count": len(reject_ids),
        "kept_ratio": round(len(kept_all) / max(len(items), 1), 6),
        "task_counts": task_counts,
        "confidence_counts": dict(sorted(confidence_counts.items(), key=lambda item: int(item[0]))),
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)),
        "copied_counts": copied_counts,
        "outputs": {
            "keep_ids": str(keep_id_path),
            "reject_ids": str(reject_id_path),
            "audit_jsonl": str(audit_path),
            "kept_all_json": str(kept_json_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
