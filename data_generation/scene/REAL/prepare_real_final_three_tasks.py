#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any


TASKS = [
    "action_inference",
    "movement_sequence_sorting",
    "movement_degree_comparison",
]
REMOVE_TASKS = ["motion_family_discrimination"]

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
    parser = argparse.ArgumentParser(description="Finalize REAL dataset with 3 tasks only.")
    parser.add_argument("--root", type=Path, default=Path("/path/to/workspace/FINAL/REAL"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def replace_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [replace_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, path_map) for key, item in value.items()}
    return value


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
            "prompt_cot": "\n".join(
                [
                    prompt,
                    "Reason from visible scene changes only.",
                    "Then output the final option letter only.",
                ]
            ),
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {
                "type": "exact_option",
                "strict_fields": ["option_letter"],
            },
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
            "prompt_cot": "\n".join(
                [
                    prompt,
                    "Reason from gradual visual progression only.",
                    "Then output the final option letter only.",
                ]
            ),
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {
                "type": "exact_option",
                "strict_fields": ["option_letter"],
            },
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
            "prompt_cot": "\n".join(
                [
                    prompt,
                    "Reason from visible viewpoint change only.",
                    "Then output the final answer in the required format only.",
                ]
            ),
            "cot_capable": True,
            "cot_steps": [],
            "cot": "",
            "scoring_rubric": {
                "type": "structured",
                "strict_fields": ["larger"],
                "continuous_fields": [
                    {"name": "frame_A", "unit": unit, "grading": "tolerance"},
                    {"name": "frame_B", "unit": unit, "grading": "tolerance"},
                ],
            },
        }
    )
    return augmented


def valid_degree_row(row: dict[str, Any]) -> bool:
    if str(row.get("task_type")) != "movement_degree_comparison":
        return True
    gt = row["gt"]
    value_a = sanitize_number((gt.get("frame_A") or {}).get("magnitude"))
    value_b = sanitize_number((gt.get("frame_B") or {}).get("magnitude"))
    larger = str(gt.get("larger_clip") or "")
    if value_a is None or value_b is None or larger not in {"A", "B"}:
        return False
    unit_cn = str((gt.get("frame_A") or {}).get("unit") or (gt.get("frame_B") or {}).get("unit") or "米")
    max_mag = max(abs(value_a), abs(value_b))
    if unit_cn == "米":
        return max_mag <= MAX_METER_MAGNITUDE
    if unit_cn == "度":
        return max_mag <= MAX_DEG_MAGNITUDE
    return False


def valid_action_row(row: dict[str, Any]) -> bool:
    if str(row.get("task_type")) != "action_inference":
        return True
    motion = (row.get("gt") or {}).get("motion") or {}
    frame_gap = sanitize_number(motion.get("frame_gap"))
    return frame_gap is not None and frame_gap <= MAX_ACTION_FRAME_GAP


def valid_sequence_row(row: dict[str, Any]) -> bool:
    if str(row.get("task_type")) != "movement_sequence_sorting":
        return True
    motions = (row.get("gt") or {}).get("motions") or []
    gaps = [sanitize_number(motion.get("frame_gap")) for motion in motions]
    gaps = [gap for gap in gaps if gap is not None]
    return bool(gaps) and max(gaps) <= MAX_SEQUENCE_STEP_GAP


def valid_row(row: dict[str, Any]) -> bool:
    return valid_action_row(row) and valid_sequence_row(row) and valid_degree_row(row)


def standardize_row(row: dict[str, Any]) -> dict[str, Any]:
    task = str(row["task_type"])
    if task == "action_inference":
        return standardize_action(row)
    if task == "movement_sequence_sorting":
        return standardize_sequence(row)
    if task == "movement_degree_comparison":
        return standardize_degree(row)
    raise ValueError(f"Unsupported task: {task}")


def build_eval_item(row: dict[str, Any], item_id: int) -> dict[str, Any]:
    task = str(row["task_type"])
    item = {
        "id": item_id,
        "benchmark": "real_standardized_three_tasks",
        "asset": "REAL",
        "task_type": task,
        "subcat": str(row["subcat"]),
        "question_type": f"{task}__{row['answer_style']}",
        "answer_style": row["answer_style"],
        "answer_format": row["answer_format"],
        "scene_name": row["scene"],
        "dataset_name": row["dataset"],
        "pose_source": row["pose_source"],
        "frame_paths": list(row["input"]["frame_paths"]),
        "source_qa_id": row["qa_id"],
        "prompt": row["prompt"],
        "ground_truth": {},
        "scoring": {
            "type": f"{task}__{row['answer_style']}",
            "max_score": 1.0,
            "rubric": row["scoring_rubric"],
        },
    }
    if task == "action_inference":
        item["options"] = row["options"]
        item["ground_truth"] = {
            "correct_letter": row["correct_option"],
            "correct_text": row["answer"],
        }
    elif task == "movement_sequence_sorting":
        item["options"] = row["options"]
        item["ground_truth"] = {
            "correct_letter": row["correct_option"],
            "correct_text": row["answer"],
        }
    elif task == "movement_degree_comparison":
        gt = row["gt"]
        frame_a = gt["frame_A"]
        frame_b = gt["frame_B"]
        unit_cn = str(frame_a.get("unit") or frame_b.get("unit") or "米")
        unit = "m" if unit_cn == "米" else "deg"
        item["options"] = None
        item["ground_truth"] = {
            "larger_candidate": str(gt["larger_clip"]),
            "frame_A_value": sanitize_number(frame_a.get("magnitude")),
            "frame_B_value": sanitize_number(frame_b.get("magnitude")),
            "unit": unit,
        }
    return item


def build_prompt_pair(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "qa": {
            "q": row["question"],
            "a": row["answer"],
        },
        "gt": row["gt"],
    }


def main() -> int:
    args = parse_args()
    root = args.root

    raw_rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    removed_invalid_counts: dict[str, int] = {}

    for task in REMOVE_TASKS:
        task_dir = root / task
        if task_dir.exists():
            shutil.rmtree(task_dir)

    for task in TASKS:
        task_dir = root / task
        rows = read_json(task_dir / "qa_data.json")
        fixed_rows: list[dict[str, Any]] = []
        removed_invalid = 0
        for row in rows:
            if not valid_row(row):
                removed_invalid += 1
                continue
            qa_id = str(row["qa_id"])
            sample_dir = task_dir / qa_id
            path_map = {}
            for old_path in list((row.get("input") or {}).get("frame_paths") or []):
                src = Path(old_path)
                path_map[str(src)] = str(sample_dir / src.name)
            fixed = dict(row)
            fixed["input"] = replace_paths(fixed["input"], path_map)
            fixed_rows.append(fixed)
        write_json(task_dir / "qa_data.json", fixed_rows)
        raw_rows.extend(fixed_rows)
        task_counts[task] = len(fixed_rows)
        removed_invalid_counts[task] = removed_invalid

    standardized_rows = [standardize_row(row) for row in raw_rows]
    eval_items = [build_eval_item(row, idx) for idx, row in enumerate(standardized_rows)]
    prompt_pairs = [build_prompt_pair(row) for row in standardized_rows]

    write_json(root / "qa_data_all.json", {"qa_count": len(raw_rows), "data": raw_rows})
    write_json(
        root / "real_qa_standardized_all.json",
        {
            "asset": "REAL",
            "qa_count": len(standardized_rows),
            "tasks": TASKS,
            "data": standardized_rows,
        },
    )
    write_json(root / "real_qa_standardized_data_only.json", standardized_rows)
    write_json(root / "real_prompt_qa_pair_gt_only.json", prompt_pairs)
    (root / "real_eval_dataset.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in eval_items) + ("\n" if eval_items else ""),
        encoding="utf-8",
    )
    write_json(
        root / "real_qa_standardized_summary.json",
        {
            "asset": "REAL",
            "tasks": TASKS,
            "task_counts": task_counts,
            "removed_invalid_counts": removed_invalid_counts,
            "qa_count": len(standardized_rows),
            "removed_tasks": REMOVE_TASKS,
            "filters": {
                "max_action_frame_gap": MAX_ACTION_FRAME_GAP,
                "max_sequence_step_gap": MAX_SEQUENCE_STEP_GAP,
                "max_degree_magnitude_deg": MAX_DEG_MAGNITUDE,
                "max_degree_magnitude_m": MAX_METER_MAGNITUDE,
            },
            "outputs": {
                "raw_all": str(root / "qa_data_all.json"),
                "standardized_all": str(root / "real_qa_standardized_all.json"),
                "standardized_data_only": str(root / "real_qa_standardized_data_only.json"),
                "prompt_pair_gt_only": str(root / "real_prompt_qa_pair_gt_only.json"),
                "eval_dataset_jsonl": str(root / "real_eval_dataset.jsonl"),
            },
        },
    )
    write_json(
        root / "summary.json",
        {
            "root": str(root),
            "task_counts": task_counts,
            "removed_invalid_counts": removed_invalid_counts,
            "removed_tasks": REMOVE_TASKS,
            "kept_total": len(raw_rows),
        },
    )
    print(json.dumps(read_json(root / "real_qa_standardized_summary.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
