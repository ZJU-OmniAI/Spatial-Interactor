#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
]

ACTION_LABELS = [
    ("A", "Move forward"),
    ("B", "Move backward"),
    ("C", "Move left"),
    ("D", "Move right"),
    ("E", "Rotate left"),
    ("F", "Rotate right"),
]
ACTION_CN_TO_EN = {
    "向前移动": "Move forward",
    "向后移动": "Move backward",
    "向左移动": "Move left",
    "向右移动": "Move right",
    "向左转动": "Rotate left",
    "向右转动": "Rotate right",
}
MAX_ACTION_FRAME_GAP = 240
MAX_SEQUENCE_STEP_GAP = 240
MAX_DEG_MAGNITUDE = 180.0
MAX_METER_MAGNITUDE = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standardize 3RScan three-task dataset.")
    parser.add_argument("--root", type=Path, default=Path("/path/to/workspace/DATA/3RSCAN"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt_m(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f} m"


def fmt_deg_int(value: float | None) -> str:
    return "unknown" if value is None else f"{int(round(value))} deg"


def stable_wrong_sequence_options(correct: str, qa_id: str) -> list[str]:
    tokens = sorted({token.strip() for token in correct.split("->")})
    perms = [" -> ".join(order) for order in itertools.permutations(tokens)]
    wrong = [item for item in perms if item != correct]
    wrong.sort(key=lambda item: hashlib.md5(f"{qa_id}|{item}".encode("utf-8")).hexdigest())
    return wrong[:3]


def stable_shuffle_texts(texts: list[str], qa_id: str, salt: str) -> list[str]:
    decorated = [
        (hashlib.md5(f"{qa_id}|{salt}|{idx}|{text}".encode("utf-8")).hexdigest(), text)
        for idx, text in enumerate(texts)
    ]
    decorated.sort(key=lambda item: item[0])
    return [text for _, text in decorated]


def infer_action_option(answer_cn: str, qa_id: str) -> tuple[list[dict[str, str]], str, str]:
    correct_text = ACTION_CN_TO_EN[answer_cn]
    texts = stable_shuffle_texts([text for _, text in ACTION_LABELS], qa_id, "action_options")
    keys = [chr(ord("A") + idx) for idx in range(len(texts))]
    options = [{"key": key, "text": text} for key, text in zip(keys, texts)]
    correct_option = next(item["key"] for item in options if item["text"] == correct_text)
    return options, correct_option, correct_text


def build_sequence_options(correct: str, qa_id: str) -> tuple[list[dict[str, str]], str]:
    texts = [correct, *stable_wrong_sequence_options(correct, qa_id)]
    texts = stable_shuffle_texts(texts, qa_id, "sequence_options")
    keys = ["A", "B", "C", "D"]
    options = [{"key": key, "text": text} for key, text in zip(keys, texts)]
    correct_key = next(option["key"] for option in options if option["text"] == correct)
    return options, correct_key


def standardize_action(row: dict[str, Any]) -> dict[str, Any]:
    options, correct_option, correct_text = infer_action_option(str(row["answer"]), str(row["qa_id"]))
    prompt = "\n".join(
        [
            "Image order: image 1 is the start view and image 2 is the end view.",
            "Question: What is the dominant camera motion from image 1 to image 2?",
            "Options:",
            *[f"{item['key']}. {item['text']}" for item in options],
            "Example output: B",
            "Reply with only the option letter.",
        ]
    )
    augmented = dict(row)
    augmented.update(
        {
            "sample_id": str(row["qa_id"]),
            "subcat": "action_inference",
            "question_original": row["question"],
            "answer_original": row["answer"],
            "question_language": "en",
            "answer_language": "en",
            "question": "What is the dominant camera motion from image 1 to image 2?",
            "answer": correct_text,
            "answer_style": "mcq",
            "answer_format": "multiple_choice",
            "options": options,
            "correct_option": correct_option,
            "prompt": prompt,
            "prompt_cot": prompt + "\nReason from visible scene changes only.\nThen output the final option letter only.",
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {"type": "exact_option", "strict_fields": ["option_letter"]},
        }
    )
    return augmented


def standardize_sequence(row: dict[str, Any]) -> dict[str, Any]:
    correct_text = str(row["answer"]).replace("A", "2").replace("B", "3").replace("C", "4")
    options, correct_option = build_sequence_options(correct_text, str(row["qa_id"]))
    prompt = "\n".join(
        [
            "Image order: image 1 is the start frame. Images 2, 3, and 4 are three candidate future frames in shuffled order.",
            "Question: Which option gives the correct chronological order of images 2, 3, and 4 after image 1?",
            "Options:",
            *[f"{item['key']}. {item['text']}" for item in options],
            "Example output: C",
            "Reply with only the option letter.",
        ]
    )
    augmented = dict(row)
    augmented.update(
        {
            "sample_id": str(row["qa_id"]),
            "subcat": "movement_sequence_sorting",
            "question_original": row["question"],
            "answer_original": row["answer"],
            "question_language": "en",
            "answer_language": "en",
            "question": "Which option gives the correct chronological order of images 2, 3, and 4 after image 1?",
            "answer": correct_text,
            "answer_style": "mcq",
            "answer_format": "multiple_choice",
            "options": options,
            "correct_option": correct_option,
            "prompt": prompt,
            "prompt_cot": prompt + "\nReason from gradual visual progression only.\nThen output the final option letter only.",
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {"type": "exact_option", "strict_fields": ["option_letter"]},
        }
    )
    return augmented


def standardize_degree(row: dict[str, Any]) -> dict[str, Any]:
    gt = row["gt"]
    frame_a = gt["frame_A"]
    frame_b = gt["frame_B"]
    unit_cn = str(frame_a.get("unit") or frame_b.get("unit") or "米")
    unit = "m" if unit_cn == "米" else "deg"
    value_a = sanitize_number(frame_a.get("magnitude"))
    value_b = sanitize_number(frame_b.get("magnitude"))
    larger = str(gt["larger_clip"])
    answer = (
        f"larger=image_{'2' if larger == 'A' else '3'} | image_2={(fmt_m(value_a) if unit == 'm' else fmt_deg_int(value_a))} | "
        f"image_3={(fmt_m(value_b) if unit == 'm' else fmt_deg_int(value_b))}"
    )
    fmt_rule = (
        "Use exactly two decimals for both numeric values because the unit is meters."
        if unit == "m"
        else "Use integer degree values with no decimals for both numeric values."
    )
    prompt = "\n".join(
        [
            "Image order: image 1 is the start frame, image 2 is the first candidate, and image 3 is the second candidate.",
            "Question: Relative to image 1, which candidate has the larger camera-motion magnitude, and what is the magnitude of image 2 and image 3?",
            f"Formatting rule: {fmt_rule}",
            "Reply with only this format:",
            f"larger=<image_2 or image_3> | image_2=<value> {unit} | image_3=<value> {unit}",
            f"Example output: larger=image_2 | image_2={( '1.25 m' if unit == 'm' else '25 deg')} | image_3={( '0.80 m' if unit == 'm' else '10 deg')}",
        ]
    )
    augmented = dict(row)
    augmented.update(
        {
            "sample_id": str(row["qa_id"]),
            "subcat": "movement_degree_comparison",
            "question_original": row["question"],
            "answer_original": row["answer"],
            "question_language": "en",
            "answer_language": "en",
            "question": "Relative to image 1, which candidate has the larger camera-motion magnitude, and what is the magnitude of image 2 and image 3?",
            "answer": answer,
            "answer_style": "structured_text",
            "answer_format": "structured_text",
            "options": None,
            "correct_option": None,
            "prompt": prompt,
            "prompt_cot": prompt + "\nReason from visible scene displacement only.\nThen output the final formatted answer only.",
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {
                "type": "structured_motion_magnitude",
                "strict_fields": ["larger", "image_2", "image_3"],
                "unit": unit,
            },
        }
    )
    return augmented


def row_is_valid(row: dict[str, Any]) -> bool:
    task = str(row.get("task_type"))
    gt = row.get("gt") or {}
    if task == "action_inference":
        motion = gt.get("motion") or {}
        return int(motion.get("frame_gap", 0)) <= MAX_ACTION_FRAME_GAP
    if task == "movement_sequence_sorting":
        motions = gt.get("motions") or []
        if len(motions) != 3:
            return False
        return all(int(motion.get("frame_gap", 0)) <= MAX_SEQUENCE_STEP_GAP for motion in motions)
    if task == "movement_degree_comparison":
        fam = str(gt.get("comparison_family"))
        motion_a = ((gt.get("frame_A") or {}).get("motion")) or {}
        motion_b = ((gt.get("frame_B") or {}).get("motion")) or {}
        if fam == "rotation":
            return (
                sanitize_number((gt.get("frame_A") or {}).get("magnitude")) is not None
                and sanitize_number((gt.get("frame_B") or {}).get("magnitude")) is not None
                and abs(float((gt.get("frame_A") or {}).get("magnitude"))) <= MAX_DEG_MAGNITUDE
                and abs(float((gt.get("frame_B") or {}).get("magnitude"))) <= MAX_DEG_MAGNITUDE
            )
        return (
            sanitize_number((gt.get("frame_A") or {}).get("magnitude")) is not None
            and sanitize_number((gt.get("frame_B") or {}).get("magnitude")) is not None
            and abs(float((gt.get("frame_A") or {}).get("magnitude"))) <= MAX_METER_MAGNITUDE
            and abs(float((gt.get("frame_B") or {}).get("magnitude"))) <= MAX_METER_MAGNITUDE
            and int(motion_a.get("frame_gap", 0)) <= MAX_ACTION_FRAME_GAP
            and int(motion_b.get("frame_gap", 0)) <= MAX_ACTION_FRAME_GAP
        )
    return False


def standardize_row(row: dict[str, Any]) -> dict[str, Any]:
    task = str(row["task_type"])
    if task == "action_inference":
        return standardize_action(row)
    if task == "movement_sequence_sorting":
        return standardize_sequence(row)
    if task == "movement_degree_comparison":
        return standardize_degree(row)
    raise ValueError(f"unsupported task: {task}")


def main() -> None:
    args = parse_args()
    raw_all_path = args.root / "qa_data_all.json"
    raw_payload = read_json(raw_all_path)
    raw_rows = raw_payload["data"] if isinstance(raw_payload, dict) and "data" in raw_payload else raw_payload

    kept_rows = [row for row in raw_rows if str(row.get("task_type")) in TASKS and row_is_valid(row)]
    standardized = [standardize_row(row) for row in kept_rows]

    data_only = [
        {
            "sample_id": row["sample_id"],
            "subcat": row["subcat"],
            "dataset": row.get("dataset"),
            "scene": row.get("scene"),
            "input": row.get("input"),
            "question": row["question"],
            "answer": row["answer"],
            "prompt": row["prompt"],
            "options": row.get("options"),
            "correct_option": row.get("correct_option"),
        }
        for row in standardized
    ]
    prompt_pair_gt = [{"prompt": row["prompt"], "qa": {"q": row["question"], "a": row["answer"]}, "gt": row.get("gt")} for row in standardized]
    eval_rows = [{"sample_id": row["sample_id"], "question": row["question"], "answer": row["answer"], "input": row.get("input"), "subcat": row["subcat"]} for row in standardized]

    task_counts = {task: sum(1 for row in kept_rows if row["task_type"] == task) for task in TASKS}
    summary = {
        "asset": "3RSCAN",
        "tasks": TASKS,
        "task_counts": task_counts,
        "removed_invalid_counts": {
            task: sum(1 for row in raw_rows if row.get("task_type") == task) - task_counts[task]
            for task in TASKS
        },
        "qa_count": len(kept_rows),
        "removed_tasks": [],
        "filters": {
            "max_action_frame_gap": MAX_ACTION_FRAME_GAP,
            "max_sequence_step_gap": MAX_SEQUENCE_STEP_GAP,
            "max_degree_magnitude_deg": MAX_DEG_MAGNITUDE,
            "max_degree_magnitude_m": MAX_METER_MAGNITUDE,
        },
        "outputs": {
            "raw_all": str(args.root / "qa_data_all.json"),
            "standardized_all": str(args.root / "3rscan_qa_standardized_all.json"),
            "standardized_data_only": str(args.root / "3rscan_qa_standardized_data_only.json"),
            "prompt_pair_gt_only": str(args.root / "3rscan_prompt_qa_pair_gt_only.json"),
            "eval_dataset_jsonl": str(args.root / "3rscan_eval_dataset.jsonl"),
        },
    }

    write_json(args.root / "3rscan_qa_standardized_all.json", standardized)
    write_json(args.root / "3rscan_qa_standardized_data_only.json", data_only)
    write_json(args.root / "3rscan_prompt_qa_pair_gt_only.json", prompt_pair_gt)
    write_json(args.root / "3rscan_qa_standardized_summary.json", summary)
    write_jsonl(args.root / "3rscan_eval_dataset.jsonl", eval_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
