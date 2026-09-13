#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


FINAL_ROOT = Path("/path/to/workspace/FINAL")

ASSET_CONFIGS = {
    "HSSD": {
        "source_dir": FINAL_ROOT / "HSSD",
        "output_dir": FINAL_ROOT / "HSSD_CURATED",
        "all_json": FINAL_ROOT / "HSSD" / "hssd_qa_standardized_all.json",
        "asset_lower": "hssd",
    },
    "REP": {
        "source_dir": FINAL_ROOT / "REP",
        "output_dir": FINAL_ROOT / "REP_CURATED",
        "all_json": FINAL_ROOT / "REP" / "rep_qa_standardized_all.json",
        "asset_lower": "rep",
    },
}

ROMAN_SUFFIX_RE = re.compile(r"\s+(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)$", re.IGNORECASE)
QUOTED_NAME_RE = re.compile(r'"([^"]+)"')
SPACE_RE = re.compile(r"\s+")

BAD_NAMES = {"", "object", "part"}
ZH_TO_EN = {
    "更容易被遮挡": "less visible",
    "更不容易被遮挡": "more visible",
    "无变化": "no visibility change",
    "没有变化": "no visibility change",
    "无法判断": "cannot tell",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "existing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def normalize_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    value = name.strip().lower()
    value = ROMAN_SUFFIX_RE.sub("", value)
    value = value.replace("_", " ")
    value = SPACE_RE.sub(" ", value).strip()
    if value == "tv object":
        value = "tv"
    return value


def quoted_names(text: str) -> list[str]:
    return [normalize_name(match) for match in QUOTED_NAME_RE.findall(text or "")]


def replace_quoted_names(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        return f'"{normalize_name(raw)}"'

    return QUOTED_NAME_RE.sub(repl, text or "")


def option(key: str, text: str) -> dict[str, str]:
    return {"key": key, "text": text}


def format_action(step: dict[str, Any], index: int) -> str:
    api = step.get("action_api") or step.get("api") or ""
    params = step.get("action_params") or step.get("action_args") or {}
    if api in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
        value = params.get("moveMagnitude") or params.get("distance") or params.get("meters") or 0.0
        return f"step{index}={api} {float(value):.2f} m"
    if api in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
        value = params.get("degrees") or params.get("degree") or params.get("angle") or 0.0
        return f"step{index}={api} {int(round(float(value)))} deg"
    return f"step{index}={api}"


def build_action_answer(steps: list[dict[str, Any]]) -> str:
    return " | ".join(format_action(step, idx) for idx, step in enumerate(steps, start=1))


def english_visibility_relation(value: str | None) -> str:
    normalized = ZH_TO_EN.get(value or "", value or "")
    if normalized in {"more visible", "less visible", "no visibility change", "cannot tell"}:
        return normalized
    return "cannot tell"


def build_mcq_prompt(question: str, options: list[dict[str, str]], include_names: bool = True) -> str:
    lines = [question]
    names = []
    if include_names:
        seen = set()
        for name in quoted_names(question):
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    if names:
        joined = ", ".join(f'"{name}"' for name in names)
        lines.append(f"Canonical object names that may be referenced: {joined}.")
    lines.append("Choose exactly one option: A, B, C, or D.")
    lines.append("Options:")
    for item in options:
        lines.append(f"{item['key']}. {item['text']}")
    lines.append("Reply with only one letter. Do not copy the option text.")
    return "\n".join(lines)


def build_short_answer_prompt(row: dict[str, Any]) -> str:
    question = row["question"]
    task = row["task_type"]
    gt = row.get("gt") or {}
    if task == "action_inference":
        count = len(gt.get("steps") or [])
        segments = [f"step{i}=<ACTION> <VALUE> <UNIT>" for i in range(1, count + 1)]
        example = build_action_answer(gt.get("steps") or [])
        return "\n".join(
            [
                question,
                "Choose actions only from: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown.",
                "Use unit m for translation and deg for rotation or camera tilt.",
                "Numeric formatting rules: use exactly two decimals for meters (example: 0.60 m) and use integer degrees with no decimals (example: 45 deg).",
                "Reply with only the final answer in the exact format shown below. Do not add explanation.",
                f"Format: {' | '.join(segments)}.",
                f"Example: {example}",
            ]
        )
    if task == "movement_degree_comparison":
        return "\n".join(
            [
                question,
                "Reply with only the final answer in this exact format.",
                "Format: larger=frame <B_or_C> | frame_B=<VALUE> <UNIT> | frame_C=<VALUE> <UNIT>",
                'Example: larger=frame C | frame_B=30 deg | frame_C=60 deg',
            ]
        )
    if task == "object_state_attribute_changes":
        return "\n".join(
            [
                question,
                'Reply with only the final answer in this exact format: object="<name>" | change="<change>".',
                'Example: object="microwave" | change="off->on"',
            ]
        )
    if task == "object_position_swapping":
        return "\n".join(
            [
                question,
                'Reply with only the final answer in this exact format: object_a="<name>" | object_b="<name>".',
                'Example: object_a="apple" | object_b="bowl"',
            ]
        )
    return question


def build_prompt_cot(prompt: str) -> str:
    return "\n".join(
        [
            prompt,
            "Reason step by step from the visible evidence and spatial relations only.",
            "Do not mention hidden metadata, annotations, or structured records.",
            "Then give the final answer in the required format.",
        ]
    )


def build_cot_and_steps(row: dict[str, Any]) -> tuple[list[str], str]:
    task = row["task_type"]
    answer = row["answer"]
    gt = row.get("gt") or {}
    if task == "action_inference":
        steps = [
            "Compare adjacent frames and infer the single camera action between each neighboring pair.",
            f"The ordered action sequence is {answer}.",
            f"So the final answer is: {answer}",
        ]
    elif task == "parallax_depth_inference":
        steps = [
            "Use the relative shift between the two named objects across the two views as the parallax cue.",
            f'The object that remains closer to the camera is "{answer}".',
            f"So the final answer is: {answer}",
        ]
    elif task == "dynamic_movement_occlusion":
        mover_name = normalize_name((((gt.get("mover") or {}).get("before") or {}).get("name")))
        if row.get("subcat") == "distance_change":
            steps = [
                f'Track how the depth of "{mover_name}" changes from frame A to frame B.',
                f'The observed change corresponds to "{answer}".',
                f"So the final answer is: {answer}",
            ]
        else:
            steps = [
                f'Track how much of "{mover_name}" remains visible after the motion.',
                f'The visibility change corresponds to "{answer}".',
                f"So the final answer is: {answer}",
            ]
    elif task == "movement_degree_comparison":
        steps = [
            "Compare frame B and frame C against the same start frame A.",
            f"The larger action and both magnitudes are summarized as {answer}.",
            f"So the final answer is: {answer}",
        ]
    elif task == "object_state_attribute_changes":
        steps = [
            "Find the single named object whose state differs between frame A and frame B.",
            f'The changed object and its state change are "{answer}".',
            f"So the final answer is: {answer}",
        ]
    elif task == "object_position_swapping":
        steps = [
            "Track the named objects that exchange their locations between the two frames.",
            f"The swapped pair is {answer}.",
            f"So the final answer is: {answer}",
        ]
    else:
        steps = [
            "Use only the visible spatial evidence in the provided frames.",
            f"The correct final answer is {answer}.",
            f"So the final answer is: {answer}",
        ]
    return steps, " ".join(steps)


def maybe_drop_row(row: dict[str, Any]) -> tuple[bool, str]:
    task = row["task_type"]
    gt = row.get("gt") or {}
    labels = quoted_names(row.get("question", ""))

    if task == "object_state_attribute_changes" and gt.get("change_category") == "rotate":
        degrees = float(((gt.get("action_params") or {}).get("degrees")) or 0.0)
        if abs(degrees) < 1e-6:
            return True, "class06_rotate0"

    if task == "parallax_depth_inference":
        objects = gt.get("objects") or {}
        name_a = normalize_name(((objects.get("object_a") or {}).get("name")))
        name_b = normalize_name(((objects.get("object_b") or {}).get("name")))
        if name_a in BAD_NAMES or name_b in BAD_NAMES:
            return True, "class03_bad_name"
        if name_a == name_b:
            return True, "class03_duplicate_name"
        return False, ""

    if task == "dynamic_movement_occlusion":
        mover_name = normalize_name((((gt.get("mover") or {}).get("before") or {}).get("name")))
        if mover_name in BAD_NAMES:
            if row.get("subcat") == "occlusion_change":
                return True, "class08_bad_mover"
            return True, "class08_bad_name"
        return False, ""

    for label in labels:
        if label in BAD_NAMES:
            return True, f"bad_label:{task}"
    dedup = [label for label in labels if label]
    if len(dedup) != len(set(dedup)):
        return True, f"duplicate_label:{task}"
    return False, ""


def standardize_parallax(row: dict[str, Any]) -> None:
    objects = row["gt"]["objects"]
    obj_a = objects["object_a"]
    obj_b = objects["object_b"]
    name_a = normalize_name(obj_a["name"])
    name_b = normalize_name(obj_b["name"])
    dist_a = float(obj_a["distance_to_camera_A"])
    dist_b = float(obj_b["distance_to_camera_A"])
    if abs(dist_a - dist_b) < 1e-4:
        answer = "same depth"
        correct = "C"
    elif dist_a < dist_b:
        answer = name_a
        correct = "A"
    else:
        answer = name_b
        correct = "B"
    row["question"] = (
        "Image order: image 1=frame A, image 2=frame B.\n"
        f'Frames A and B show the same scene from nearby viewpoints. Using parallax cues, which object is closer to the camera: "{name_a}" or "{name_b}"?'
    )
    row["answer"] = answer
    row["options"] = [
        option("A", name_a),
        option("B", name_b),
        option("C", "same depth"),
        option("D", "cannot tell"),
    ]
    row["correct_option"] = correct


def standardize_dynamic_movement(row: dict[str, Any]) -> None:
    gt = row["gt"]
    mover_name = normalize_name((((gt.get("mover") or {}).get("before") or {}).get("name")))
    if row.get("subcat") == "distance_change":
        before = gt.get("distance_before")
        after = gt.get("distance_after")
        if before is None or after is None:
            answer = "cannot tell"
        else:
            before_f = float(before)
            after_f = float(after)
            if abs(after_f - before_f) < 1e-4:
                answer = "same distance"
            elif after_f < before_f:
                answer = "closer"
            else:
                answer = "farther"
        row["question"] = (
            "Image order: image 1=frame A, image 2=frame B.\n"
            f'The moving object is "{mover_name}". Compare its depth relative to the observer in frame A and frame B. '
            "Did it move closer, farther, stay at the same distance, or is it impossible to tell?"
        )
        row["options"] = [
            option("A", "closer"),
            option("B", "farther"),
            option("C", "same distance"),
            option("D", "cannot tell"),
        ]
        mapping = {"closer": "A", "farther": "B", "same distance": "C", "cannot tell": "D"}
    else:
        answer = english_visibility_relation(gt.get("relation"))
        row["question"] = (
            "Image order: image 1=frame A, image 2=frame B.\n"
            f'A scene object moved between frame A and frame B. Because of that motion, did "{mover_name}" become more visible, less visible, '
            "show no visibility change, or is it impossible to tell?"
        )
        row["options"] = [
            option("A", "no visibility change"),
            option("B", "less visible"),
            option("C", "more visible"),
            option("D", "cannot tell"),
        ]
        mapping = {
            "no visibility change": "A",
            "less visible": "B",
            "more visible": "C",
            "cannot tell": "D",
        }
    row["answer"] = answer
    row["correct_option"] = mapping[answer]


def standardize_action_inference(row: dict[str, Any]) -> None:
    row["answer"] = build_action_answer((row.get("gt") or {}).get("steps") or [])


def standardize_movement_degree(row: dict[str, Any]) -> None:
    gt = row["gt"]
    larger = gt["larger_image"]
    frame_b = gt["frame_B"]
    frame_c = gt["frame_C"]
    unit_b = "deg" if frame_b.get("unit") == "degrees" else "m"
    unit_c = "deg" if frame_c.get("unit") == "degrees" else "m"
    value_b = int(round(float(frame_b["planned_value"]))) if unit_b == "deg" else float(frame_b["planned_value"])
    value_c = int(round(float(frame_c["planned_value"]))) if unit_c == "deg" else float(frame_c["planned_value"])
    text_b = f"{value_b} {unit_b}" if unit_b == "deg" else f"{value_b:.2f} {unit_b}"
    text_c = f"{value_c} {unit_c}" if unit_c == "deg" else f"{value_c:.2f} {unit_c}"
    row["answer"] = f"larger=frame {larger} | frame_B={text_b} | frame_C={text_c}"


def standardize_state_change(row: dict[str, Any]) -> None:
    gt = row["gt"]
    target_name = normalize_name((gt.get("target") or {}).get("name"))
    category = gt.get("change_category")
    if category == "remove":
        change = "removed"
    elif category == "rotate":
        degrees = int(round(float(((gt.get("action_params") or {}).get("degrees")) or 0.0)))
        change = f"rotated {degrees} deg"
    elif category == "toggle":
        before = (gt.get("target_state_before") or {}).get("is_toggled")
        after = (gt.get("target_state_after") or {}).get("is_toggled")
        change = "off->on" if (before is False and after is True) else "on->off"
    elif category == "open":
        before = (gt.get("target_state_before") or {}).get("is_open")
        after = (gt.get("target_state_after") or {}).get("is_open")
        change = "closed->open" if (before is False and after is True) else "open->closed"
    else:
        change = normalize_name(gt.get("change_text"))
    row["answer"] = f'object="{target_name}" | change="{change}"'


def standardize_swap(row: dict[str, Any]) -> None:
    swap_pair = row["gt"]["swap_pair"]
    names = [normalize_name(item.get("answer_name")) for item in swap_pair]
    row["answer"] = f'object_a="{names[0]}" | object_b="{names[1]}"'


def standardize_generic_text(row: dict[str, Any]) -> None:
    row["question"] = replace_quoted_names(row["question"])
    row["answer"] = replace_quoted_names(row["answer"])
    if row.get("options"):
        for item in row["options"]:
            item["text"] = replace_quoted_names(item["text"])


def standardize_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(row)
    updated["question_original"] = updated.get("question_original") or updated.get("question")
    updated["answer_original"] = updated.get("answer_original") or updated.get("answer")

    task = updated["task_type"]
    if task == "action_inference":
        standardize_action_inference(updated)
    elif task == "parallax_depth_inference":
        standardize_parallax(updated)
    elif task == "dynamic_movement_occlusion":
        standardize_dynamic_movement(updated)
    elif task == "movement_degree_comparison":
        standardize_movement_degree(updated)
    elif task == "object_state_attribute_changes":
        standardize_state_change(updated)
    elif task == "object_position_swapping":
        standardize_swap(updated)

    standardize_generic_text(updated)
    updated["question_core"] = updated["question"]
    updated["question_variants"] = [updated["question"]]
    updated["question_language"] = "en"
    updated["answer_language"] = "en"

    if updated.get("answer_style") == "mcq":
        updated["prompt"] = build_mcq_prompt(updated["question"], updated.get("options") or [])
    else:
        updated["prompt"] = build_short_answer_prompt(updated)
    updated["prompt_cot"] = build_prompt_cot(updated["prompt"])
    updated["cot_steps"], updated["cot"] = build_cot_and_steps(updated)

    augmentation_meta = deepcopy(updated.get("augmentation_meta") or {})
    augmentation_meta["curated_version"] = "habitat_final_v1"
    augmentation_meta["question_standardized"] = True
    augmentation_meta["answer_standardized"] = True
    updated["augmentation_meta"] = augmentation_meta
    return updated


def copy_sample_images(source_images_root: Path, output_images_root: Path, row: dict[str, Any]) -> dict[str, int]:
    stats = {"linked": 0, "copied": 0, "existing": 0}
    for rel_path in (row.get("input") or {}).get("frame_paths") or []:
        src = source_images_root.parent / rel_path
        dst = output_images_root.parent / rel_path
        status = _safe_link_or_copy(src, dst)
        stats[status] += 1
    return stats


def curate_asset(asset: str) -> dict[str, Any]:
    config = ASSET_CONFIGS[asset]
    source_dir = config["source_dir"]
    output_dir = config["output_dir"]
    all_json = config["all_json"]
    asset_lower = config["asset_lower"]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(all_json.read_text(encoding="utf-8"))
    rows = payload["data"]
    curated_rows: list[dict[str, Any]] = []
    drop_reasons: dict[str, int] = {}
    link_stats = {"linked": 0, "copied": 0, "existing": 0}

    source_images_root = source_dir / "images"
    output_images_root = output_dir / "images"

    for row in rows:
        should_drop, reason = maybe_drop_row(row)
        if should_drop:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        updated = standardize_row(row)
        curated_rows.append(updated)
        sample_stats = copy_sample_images(source_images_root, output_images_root, updated)
        for key, value in sample_stats.items():
            link_stats[key] += value

    all_payload = deepcopy(payload)
    all_payload["qa_count"] = len(curated_rows)
    all_payload["data"] = curated_rows
    all_payload["packaged_root"] = str(output_dir)
    all_payload["portable_paths"] = True
    all_payload["simple_layout"] = True
    all_payload["curation_meta"] = {
        "version": "habitat_final_v1",
        "source_dir": str(source_dir),
        "drop_reasons": drop_reasons,
    }
    filter_stats = deepcopy(all_payload.get("filter_stats") or {})
    filter_stats["kept_after_curation"] = len(curated_rows)
    filter_stats["curation_enabled"] = True
    all_payload["filter_stats"] = filter_stats

    prompt_pair = [
        {
            "prompt": row.get("prompt", ""),
            "qa": {"q": row.get("question", ""), "a": row.get("answer", "")},
            "gt": row.get("gt", {}),
        }
        for row in curated_rows
    ]

    _write_json(output_dir / f"{asset_lower}_qa_standardized_all.json", all_payload)
    _write_json(output_dir / f"{asset_lower}_qa_standardized_data_only.json", curated_rows)
    _write_json(output_dir / f"{asset_lower}_prompt_qa_pair_gt_only.json", prompt_pair)

    keep_ids = [row["sample_id"] for row in curated_rows]
    (output_dir / f"{asset_lower}_curated_keep_sample_ids.txt").write_text("\n".join(keep_ids) + "\n", encoding="utf-8")

    summary = {
        "asset": asset,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "original_qa_count": len(rows),
        "curated_qa_count": len(curated_rows),
        "dropped_count": len(rows) - len(curated_rows),
        "drop_reasons": drop_reasons,
        "unique_sample_dirs": len(curated_rows),
        "unique_image_files": sum(1 for _ in output_images_root.rglob("*.png")),
        "link_stats": {
            "linked": link_stats["linked"],
            "copied": link_stats["copied"],
            "reused_existing": link_stats["existing"],
        },
        "layout": "images/<sample_id>/<frame_file>",
        "portable_paths": True,
        "simple_layout": True,
    }
    _write_json(output_dir / f"{asset_lower}_curation_summary.json", summary)
    _write_json(output_dir / f"{asset_lower}_package_summary.json", summary)
    return summary


def main() -> int:
    manifest = {"root": str(FINAL_ROOT), "assets": []}
    for asset in ("HSSD", "REP"):
        manifest["assets"].append(curate_asset(asset))
    manifest["total_qa_count"] = sum(item["curated_qa_count"] for item in manifest["assets"])
    _write_json(FINAL_ROOT / "habitat_curated_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
