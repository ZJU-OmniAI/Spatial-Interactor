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
    parser = argparse.ArgumentParser(description="Filter REAL QA by pure multimodal answerability.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/path/to/workspace/models/Qwen3.5-9B"))
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--limit-mm-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
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


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def task_instruction(task: str) -> str:
    if task == "action_inference":
        return (
            "Keep only if the main action direction can be inferred from the two frames without guessing. "
            "Delete if the gap is so large that the motion is meaningless, the views feel discontinuous, "
            "or the direction/family cannot be judged confidently."
        )
    if task == "movement_sequence_sorting":
        return (
            "Keep only if the temporal order of the candidate frames can be recovered from visible progression. "
            "Delete if the frames jump too far, feel unrelated, or the order cannot be judged confidently."
        )
    if task == "movement_degree_comparison":
        return (
            "Keep only if a vision-language model can judge which candidate has larger motion magnitude from the visible changes. "
            "Delete if the comparison is ambiguous, visually unsupported, or based on an excessive meaningless gap."
        )
    if task == "motion_family_discrimination":
        return (
            "Keep only if the dominant motion family can be judged from the two frames. "
            "Delete if the frame gap is too large, the motion family is meaningless to ask, "
            "or the answer would mostly be a guess."
        )
    return "Keep only if the question is clearly answerable from the frames."


def issue_set(task: str) -> list[str]:
    shared = [
        "unanswerable_due_to_large_gap",
        "unrelated_or_discontinuous_views",
        "too_subtle_or_visually_weak",
        "question_not_grounded_in_frames",
        "hard_to_answer_confidently",
    ]
    if task == "action_inference":
        return shared + ["ambiguous_action_direction"]
    if task == "movement_sequence_sorting":
        return shared + ["ambiguous_sequence_order"]
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
        "You are a strict multimodal dataset quality judge for embodied motion QA. "
        "Do not output reasoning, markdown, or <think>. Output only one compact JSON object."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{frame_markers}\n"
        "Judge only whether this question is meaningfully answerable from the provided frames for a strong vision-language model.\n"
        "Delete samples that are visually meaningless, rely on excessive frame jumps, feel discontinuous or unrelated, "
        "or cannot be answered confidently from image evidence.\n"
        f"Task: {task}\n"
        f"Frame labels in order: {labels}\n"
        f"Question: {row.get('question', '')}\n"
        f"Policy: {task_instruction(task)}\n"
        f"Allowed issue names: {issues}\n"
        "Return JSON with keys: decision, issues, reason.\n"
        "decision must be keep or delete.\n"
        "issues must be a JSON array using only the allowed issue names.\n"
        "reason must be one short sentence under 20 words.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"image": frame_paths}}


def parse_output(text: str) -> tuple[str, list[str], str]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            decision = str(payload.get("decision") or "keep").strip().lower()
            issues = [str(item) for item in payload.get("issues") or [] if item]
            reason = str(payload.get("reason") or "").strip()
            if decision not in {"keep", "delete"}:
                decision = "keep"
            return decision, issues, reason
        except Exception:
            pass
    lowered = text.lower()
    return ("delete" if "delete" in lowered else "keep"), [], text[:120]


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
    task_counts: dict[str, dict[str, int]] = {task: {"total": 0, "keep": 0, "reject": 0} for task in tasks}

    for (task, row), output in zip(items, outputs):
        qa_id = str(row["qa_id"])
        text = output.outputs[0].text if output.outputs else ""
        decision, issues, reason = parse_output(text)
        task_counts[task]["total"] += 1
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
                "decision": decision,
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
            sample_dir = out_task_dir / qa_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            path_map: dict[str, str] = {}
            for old_path in row["input"]["frame_paths"]:
                src = Path(old_path)
                dst = sample_dir / src.name
                copy_or_link(src, dst, args.copy_mode)
                path_map[str(src)] = str(dst)
            new_row = deepcopy(row)
            new_row["input"] = replace_paths(new_row["input"], path_map)
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
