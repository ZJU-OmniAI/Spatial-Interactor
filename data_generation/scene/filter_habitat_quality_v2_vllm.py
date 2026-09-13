#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


DEFAULT_MODEL = Path("/path/to/workspace/models/Qwen3.5-9B")
CLASS_RE = re.compile(r"__class(\d{2})__")
QUESTION_NAME_RE = re.compile(r"[【\[]([^】\]]+)[】\]]")
IMAGE_SIZE = (96, 96)

CLASS_NAMES = {
    1: "action_inference",
    2: "overlap_localization",
    3: "parallax_depth",
    4: "movement_sequence_sorting",
    5: "movement_degree_comparison",
    6: "object_state_attribute_changes",
    7: "object_position_swapping",
    8: "dynamic_movement_occlusion",
    9: "imagined_perspective_taking",
    10: "imagined_movement_consequence",
}

ALLOWED_ISSUES = {
    "image_missing",
    "image_corrupt",
    "image_uninformative",
    "duplicate_or_near_duplicate_frames",
    "low_visual_change",
    "target_not_identifiable",
    "multi_instance_confusion",
    "question_ambiguous",
    "answer_not_visually_supported",
    "floating_or_clipping",
    "unphysical_object_support",
    "severe_render_artifact",
    "occlusion_or_crop_problem",
    "other",
}

SIMULATOR_NAME_ALIASES = {
    "house plant": "plant",
    "potted plant": "plant",
    "poulsen ph": "lamp",
    "all up": "ceiling lamp",
    "part": "object part",
}


@dataclass
class Sample:
    sample_id: str
    scene: str
    class_id: int
    task_type: str
    question: str
    answer: str
    gt: dict[str, Any]
    frame_paths: list[str]
    source_row: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass multimodal quality filter for habitat QA renders.")
    parser.add_argument("--asset", required=True, choices=["HSSD", "REP"])
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tag", type=str, default="v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_name(text: object) -> str:
    value = normalize_space(text).strip('"').strip("'").lower()
    return SIMULATOR_NAME_ALIASES.get(value, value)


def parse_class_id(sample_id: str) -> int | None:
    match = CLASS_RE.search(sample_id or "")
    return int(match.group(1)) if match else None


def extract_frame_paths(row: dict[str, Any]) -> list[str]:
    input_block = row.get("input") or {}
    paths: list[str] = []
    if isinstance(input_block.get("frame_paths"), list):
        paths.extend(str(item) for item in input_block["frame_paths"] if item)
    labeled = input_block.get("labeled_frames")
    if isinstance(labeled, dict):
        for key in sorted(labeled):
            value = labeled[key]
            if value:
                paths.append(str(value))
    for key, value in input_block.items():
        if key.startswith("frame_") and isinstance(value, str):
            paths.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def iter_samples(input_json: Path) -> Iterable[Sample]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        class_id = parse_class_id(sample_id)
        if class_id is None:
            continue
        yield Sample(
            sample_id=sample_id,
            scene=str(row.get("scene") or sample_id.split("__", 1)[0]),
            class_id=class_id,
            task_type=str(row.get("task_type") or CLASS_NAMES.get(class_id, "")),
            question=str(row.get("question") or ""),
            answer=str(row.get("answer") or ""),
            gt=dict(row.get("gt") or {}),
            frame_paths=extract_frame_paths(row),
            source_row=row,
        )


def chunked(items: list[Sample], size: int) -> Iterable[list[Sample]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(IMAGE_SIZE), dtype=np.float32)


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)) / 255.0)


def std(a: np.ndarray) -> float:
    return float(np.std(a) / 255.0)


def image_metrics(frame_paths: list[str]) -> dict[str, Any]:
    arrays = [load_gray(path) for path in frame_paths]
    stds = [std(arr) for arr in arrays]
    diffs = [mad(arrays[idx], arrays[idx + 1]) for idx in range(len(arrays) - 1)]
    pair_diffs: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            pair_diffs.append(mad(arrays[i], arrays[j]))
    return {
        "stds": stds,
        "adj_diffs": diffs,
        "min_adj_diff": min(diffs) if diffs else None,
        "min_pair_diff": min(pair_diffs) if pair_diffs else None,
        "mean_pair_diff": (sum(pair_diffs) / len(pair_diffs)) if pair_diffs else None,
    }


def collect_question_names(question: str) -> list[str]:
    names = [normalize_name(item) for item in QUESTION_NAME_RE.findall(question or "")]
    return [item for item in names if item]


def collect_object_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"name", "display_name", "answer_name"} and item:
                names.append(normalize_name(item))
            else:
                names.extend(collect_object_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(collect_object_names(item))
    return names


def object_name_hints(sample: Sample) -> list[str]:
    names = []
    names.extend(collect_question_names(sample.question))
    names.extend(collect_object_names(sample.gt))
    answer = normalize_name(sample.answer)
    if answer and len(answer.split()) <= 4:
        names.append(answer)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped[:16]


def heuristic_flags(sample: Sample, metrics: dict[str, Any]) -> list[str]:
    min_adj = metrics.get("min_adj_diff")
    min_pair = metrics.get("min_pair_diff")
    flags: list[str] = []

    if any(value < 0.008 for value in metrics.get("stds") or []):
        flags.append("image_uninformative")

    if min_pair is not None and min_pair < 0.0012:
        flags.append("duplicate_or_near_duplicate_frames")

    if sample.class_id in {1, 2, 3, 4, 10} and min_adj is not None and min_adj < 0.0035:
        flags.append("low_visual_change")
    if sample.class_id == 5 and min_pair is not None and min_pair < 0.012:
        flags.append("low_visual_change")
    if sample.class_id == 6 and min_adj is not None and min_adj < 0.0012:
        flags.append("low_visual_change")
    if sample.class_id == 7 and min_adj is not None and min_adj < 0.0015:
        flags.append("low_visual_change")
    if sample.class_id == 8 and min_adj is not None and min_adj < 0.0012:
        flags.append("low_visual_change")
    if sample.class_id == 9 and min_pair is not None and min_pair < 0.01:
        flags.append("image_uninformative")

    return sorted(set(flags))


def image_status(frame_paths: list[str]) -> tuple[list[Image.Image], list[str], list[dict[str, Any]]]:
    images: list[Image.Image] = []
    issues: list[str] = []
    stats: list[dict[str, Any]] = []
    for path_str in frame_paths:
        path = Path(path_str)
        if not path.is_file():
            issues.append("image_missing")
            stats.append({"path": path_str, "exists": False})
            continue
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images.append(rgb.copy())
                stats.append({"path": path_str, "exists": True, "width": rgb.width, "height": rgb.height})
        except (UnidentifiedImageError, OSError):
            issues.append("image_corrupt")
            stats.append({"path": path_str, "exists": True, "readable": False})
    return images, sorted(set(issues)), stats


def build_prompt(sample: Sample, auto_flags: list[str], image_issues: list[str], text_only: bool) -> dict[str, Any]:
    name_hints = object_name_hints(sample)
    vision = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in sample.frame_paths)
    system_text = (
        "You are a strict quality filter for Habitat embodied visual QA. "
        "Do not output any reasoning, explanation, markdown, or <think> tags. "
        "Output only a compact JSON object. "
        "Delete any sample with floating objects, clipping/interpenetration, impossible support, severe crop/occlusion, "
        "unidentifiable target objects, ambiguous multi-instance confusion, or changes that are too subtle to judge. "
        "Be conservative: do not flag ceiling lamps, wall-mounted items, curtains, mirrors, or hanging decor as floating if a plausible support fixture exists."
    )
    user_text = f"""Judge whether this QA sample should remain in the final benchmark.

Keep only if the frames are intact, the question is answerable from visible evidence, the named target object(s) are identifiable, the key change / motion / spatial relation is clearly perceivable, and there is no obvious simulator or render defect.

Only use floating_or_clipping or unphysical_object_support when the artifact is clearly visible.

Decision policy:
- keep: clearly high quality and visually grounded
- review: borderline or genuinely uncertain
- delete: clearly flawed or unusable

Use a very short reason: one sentence, at most 25 words.

Allowed issue names:
{sorted(ALLOWED_ISSUES)}

Metadata:
- sample_id: {sample.sample_id}
- class_id: {sample.class_id}
- class_name: {CLASS_NAMES.get(sample.class_id, "unknown")}
- task_type: {sample.task_type}
- image_count: {len(sample.frame_paths)}
- target_name_hints: {name_hints if name_hints else "none"}
- auto_flags: {auto_flags if auto_flags else "none"}
- precheck_image_issues: {image_issues if image_issues else "none"}

Question:
{sample.question}

Answer:
{sample.answer}
"""
    prompt = (
        "<|im_start|>system\n"
        f"{system_text}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{vision}{user_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if text_only:
        return {"prompt": prompt}
    return {"prompt": prompt, "multi_modal_data": {"image": sample.frame_paths}}


def build_structured_outputs() -> StructuredOutputsParams:
    return StructuredOutputsParams(
        json={
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["keep", "review", "delete"]},
                "overall_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "issues": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(ALLOWED_ISSUES)},
                    "maxItems": 4,
                },
                "reason": {"type": "string", "maxLength": 220},
            },
            "required": ["decision", "overall_score", "issues", "reason"],
            "additionalProperties": False,
        },
        disable_additional_properties=True,
    )


def extract_json_block(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def clamp_score(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except Exception:
        return 3
    return max(1, min(5, numeric))


def normalize_issues(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item).strip()
        if not token:
            continue
        if token not in ALLOWED_ISSUES:
            token = "other"
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def normalize_result(
    sample: Sample,
    raw_text: str,
    image_issues: list[str],
    auto_flags: list[str],
    image_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    parsed = extract_json_block(raw_text)
    if parsed is None:
        issues = sorted(set(image_issues + auto_flags + ["other"]))
        return {
            "sample_id": sample.sample_id,
            "scene": sample.scene,
            "class_id": sample.class_id,
            "class_name": CLASS_NAMES.get(sample.class_id, "unknown"),
            "task_type": sample.task_type,
            "question": sample.question,
            "answer": sample.answer,
            "frame_paths": sample.frame_paths,
            "image_stats": image_stats,
            "decision": "review",
            "overall_score": 2,
            "issues": issues,
            "reason": "Model output was not valid JSON.",
            "raw_model_output": raw_text,
        }

    decision = str(parsed.get("decision") or "review").strip().lower()
    if decision not in {"keep", "review", "delete"}:
        decision = "review"
    issues = normalize_issues(parsed.get("issues"))
    for item in image_issues + auto_flags:
        if item not in issues:
            issues.append(item)
    return {
        "sample_id": sample.sample_id,
        "scene": sample.scene,
        "class_id": sample.class_id,
        "class_name": CLASS_NAMES.get(sample.class_id, "unknown"),
        "task_type": sample.task_type,
        "question": sample.question,
        "answer": sample.answer,
        "frame_paths": sample.frame_paths,
        "image_stats": image_stats,
        "decision": decision,
        "overall_score": clamp_score(parsed.get("overall_score", 3)),
        "issues": issues,
        "reason": normalize_space(parsed.get("reason") or "No reason provided.")[:400],
        "raw_model_output": raw_text,
    }


def auto_record(sample: Sample, issues: list[str], image_stats: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "scene": sample.scene,
        "class_id": sample.class_id,
        "class_name": CLASS_NAMES.get(sample.class_id, "unknown"),
        "task_type": sample.task_type,
        "question": sample.question,
        "answer": sample.answer,
        "frame_paths": sample.frame_paths,
        "image_stats": image_stats,
        "decision": "delete",
        "overall_score": 1,
        "issues": sorted(set(issues)),
        "reason": reason,
        "raw_model_output": "",
    }


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize(records: list[dict[str, Any]], args: argparse.Namespace, total_input: int) -> dict[str, Any]:
    decisions = Counter(record["decision"] for record in records)
    issues = Counter()
    by_class = defaultdict(lambda: {"keep": 0, "review": 0, "delete": 0, "total": 0})
    for record in records:
        cls = int(record["class_id"])
        by_class[cls]["total"] += 1
        by_class[cls][record["decision"]] += 1
        for issue in record.get("issues", []):
            issues[issue] += 1

    keep_ids = [record["sample_id"] for record in records if record["decision"] == "keep"]
    reject_ids = [record["sample_id"] for record in records if record["decision"] != "keep"]
    return {
        "asset": args.asset,
        "tag": args.tag,
        "model": str(args.model),
        "input_json": str(args.input_json),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_input_samples": total_input,
        "processed_samples": len(records),
        "decision_counts": dict(decisions),
        "issue_counts": dict(issues.most_common()),
        "class_stats": {
            str(class_id): {**stats, "class_name": CLASS_NAMES.get(class_id, "unknown")}
            for class_id, stats in sorted(by_class.items())
        },
        "keep_sample_ids": keep_ids,
        "reject_sample_ids": reject_ids,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = list(iter_samples(args.input_json))
    all_samples.sort(key=lambda sample: sample.sample_id)
    if args.num_shards > 1:
        all_samples = [sample for idx, sample in enumerate(all_samples) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        all_samples = all_samples[: args.limit]

    stem = f"{args.asset.lower()}_quality_filter_{args.tag}_shard{args.shard_index:02d}of{args.num_shards:02d}"
    out_json = args.output_dir / f"{stem}.json"
    out_jsonl = args.output_dir / f"{stem}.jsonl"
    existing_records = load_existing_records(out_jsonl) if args.resume else []
    done_ids = {str(record.get("sample_id") or "") for record in existing_records}
    pending = [sample for sample in all_samples if sample.sample_id not in done_ids]

    print(
        json.dumps(
            {
                "asset": args.asset,
                "input_json": str(args.input_json),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "loaded": len(all_samples),
                "already_processed": len(existing_records),
                "pending": len(pending),
            },
            ensure_ascii=False,
        )
    )

    llm = None
    sampling_params = None
    if pending:
        llm = LLM(
            model=str(args.model),
            tokenizer=str(args.model),
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": 6},
            max_num_seqs=max(1, args.batch_size),
            disable_log_stats=True,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
        )

    with out_jsonl.open("a", encoding="utf-8") as fout:
        processed = len(existing_records)
        for batch in chunked(pending, max(1, args.batch_size)):
            auto_records: list[dict[str, Any]] = []
            model_batch: list[tuple[Sample, list[str], list[dict[str, Any]], dict[str, Any]]] = []
            for sample in batch:
                images, image_issues, image_stats = image_status(sample.frame_paths)
                if image_issues:
                    auto_records.append(auto_record(sample, image_issues, image_stats, "Missing or corrupt image."))
                    continue
                if not sample.question.strip() or not sample.answer.strip():
                    auto_records.append(auto_record(sample, ["other"], image_stats, "Question or answer is empty."))
                    continue
                metrics = image_metrics(sample.frame_paths)
                auto_flags = heuristic_flags(sample, metrics)
                if "duplicate_or_near_duplicate_frames" in auto_flags and sample.class_id in {5, 6, 7, 8}:
                    auto_records.append(
                        auto_record(
                            sample,
                            auto_flags,
                            image_stats,
                            "Frames are near-duplicate for a class that requires a visible change.",
                        )
                    )
                    continue
                prompt = build_prompt(sample, auto_flags, image_issues, args.text_only)
                model_batch.append((sample, auto_flags, image_stats, prompt))

            for record in auto_records:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

            if model_batch:
                outputs = llm.generate(
                    [entry[3] for entry in model_batch],
                    sampling_params=[
                        SamplingParams(
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=args.max_tokens,
                            structured_outputs=build_structured_outputs(),
                        )
                        for _ in model_batch
                    ],
                    use_tqdm=False,
                )
                for (sample, auto_flags, image_stats, _prompt), output in zip(model_batch, outputs):
                    raw_text = output.outputs[0].text if output.outputs else ""
                    record = normalize_result(sample, raw_text, [], auto_flags, image_stats)
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed += 1

            fout.flush()
            print(
                json.dumps(
                    {
                        "asset": args.asset,
                        "shard": f"{args.shard_index}/{args.num_shards}",
                        "processed": processed,
                        "total": len(all_samples),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    records = load_existing_records(out_jsonl)
    summary = summarize(records, args, len(all_samples))
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
