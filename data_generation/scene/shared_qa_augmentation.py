#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DIRECTION_EN = {
    "正前方": "front",
    "前方": "front",
    "right front": "front-right",
    "右前方": "front-right",
    "正右方": "right",
    "右侧": "right",
    "正后方": "back",
    "后方": "back",
    "左后方": "back-left",
    "正左方": "left",
    "左侧": "left",
    "左前方": "front-left",
    "右后方": "back-right",
    "front": "front",
    "front right": "front-right",
    "front-left": "front-left",
    "front left": "front-left",
    "right": "right",
    "back-right": "back-right",
    "back right": "back-right",
    "back": "back",
    "back-left": "back-left",
    "back left": "back-left",
    "left": "left",
}

SCREEN_LOC_EN = {
    "左侧": "left",
    "中间": "center",
    "右侧": "right",
    "left": "left",
    "center": "center",
    "right": "right",
}

DEPTH_REL_EN = {
    "更近": "closer",
    "更远": "farther",
    "closer": "closer",
    "nearer": "closer",
    "farther": "farther",
    "further": "farther",
    "same distance": "same distance",
    "无法判断": "cannot tell",
    "cannot tell": "cannot tell",
}

ACTION_API_EN = {
    "RotateLeft": "rotate left",
    "RotateRight": "rotate right",
    "MoveAhead": "move forward",
    "MoveBack": "move backward",
    "MoveLeft": "move left",
    "MoveRight": "move right",
    "LookUp": "look up",
    "LookDown": "look down",
}

MCQ_TASKS = {
    "multi_image_overlap_localization",
    "parallax_depth_inference",
    "movement_sequence_sorting",
    "dynamic_movement_occlusion",
    "imagined_perspective_taking",
    "imagined_movement_consequence",
}

OBJECT_NAME_ALIASES = {
    "house plant": "plant",
    "potted plant": "plant",
    "plant": "plant",
    "soap bottle": "soap dispenser",
    "soap dispenser": "soap dispenser",
    "hand towel": "towel",
    "towel": "towel",
    "alarm clock": "clock",
    "clock": "clock",
    "floor lamp": "lamp",
    "lamp": "lamp",
}

OBJECT_KEYS = {"display_name", "name", "objectType", "type", "id"}
QUESTION_NAME_RE = re.compile(r"[【\[]([^】\]]+)[】\]]")


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _snake_to_words(text: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text or ""))
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\|.*$", "", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return _normalize_spaces(text)


def normalize_object_name(raw: object) -> str:
    if isinstance(raw, dict):
        for key in ("display_name", "name", "objectType", "type", "id"):
            if raw.get(key):
                return normalize_object_name(raw[key])
        return "object"
    text = _snake_to_words(str(raw or "object")).strip("[]【】")
    if not text:
        return "object"
    text = text.lower()
    return OBJECT_NAME_ALIASES.get(text, text)


def _name(obj: object, default: str = "object") -> str:
    text = normalize_object_name(obj)
    return text if text else default


def _quote(name: object, default: str = "object") -> str:
    return f'"{_name(name, default)}"'


def _question_object_names(question: str) -> List[str]:
    return [_name(item) for item in QUESTION_NAME_RE.findall(str(question or "")) if _name(item)]


def _fmt_num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.00"


def _unit_suffix(unit: object) -> str:
    text = _normalize_spaces(str(unit or "")).lower()
    if text in {"degree", "degrees", "deg"}:
        return "deg"
    if text in {"meter", "meters", "metre", "metres", "m"}:
        return "m"
    return text


def _measurement_text(value: object, unit: object) -> str:
    suffix = _unit_suffix(unit)
    digits = 0 if suffix == "deg" else 2
    return f"{_fmt_num(value, digits)} {suffix}".strip()


def _fmt_action_texts(actions: Iterable[Dict]) -> List[str]:
    texts: List[str] = []
    for action in actions or []:
        text = english_action_text(action)
        if text:
            texts.append(text)
    return texts


def _label_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("horizontal", "direction", "direction_word", "relation", "label", "text"):
            if value.get(key):
                return _normalize_spaces(str(value[key]))
        return _normalize_spaces(json.dumps(value, ensure_ascii=False))
    return _normalize_spaces(str(value or ""))


def _stable_pick(options: Sequence[str], sample_id: str, *, salt: str) -> str:
    if not options:
        return ""
    key = f"{sample_id}::{salt}".encode("utf-8")
    idx = int(hashlib.md5(key).hexdigest(), 16) % len(options)
    return options[idx]


def _stable_sort_texts(options: Sequence[str], sample_id: str, *, salt: str) -> List[str]:
    unique = list(dict.fromkeys(_normalize_spaces(item) for item in options if _normalize_spaces(item)))
    return sorted(unique, key=lambda item: hashlib.md5(f"{sample_id}::{salt}::{item}".encode("utf-8")).hexdigest())


def _label_to_english(value: object) -> str:
    text = _label_text(value)
    lowered = text.lower()
    if lowered in DEPTH_REL_EN:
        return DEPTH_REL_EN[lowered]
    if text in DEPTH_REL_EN:
        return DEPTH_REL_EN[text]
    if lowered in DIRECTION_EN:
        return DIRECTION_EN[lowered]
    if text in DIRECTION_EN:
        return DIRECTION_EN[text]
    if lowered in SCREEN_LOC_EN:
        return SCREEN_LOC_EN[lowered]
    if text in SCREEN_LOC_EN:
        return SCREEN_LOC_EN[text]
    return lowered or text


def _collect_object_names(node: object, sink: List[str]) -> None:
    if isinstance(node, dict):
        for key in OBJECT_KEYS:
            if key in node and node.get(key):
                text = normalize_object_name(node.get(key))
                if text and text != "object":
                    sink.append(text)
        for value in node.values():
            _collect_object_names(value, sink)
        return
    if isinstance(node, list):
        for item in node:
            _collect_object_names(item, sink)


def _candidate_object_names(row: Dict) -> List[str]:
    names: List[str] = []
    _collect_object_names(row.get("gt", {}) or {}, names)
    _collect_object_names(row.get("input", {}) or {}, names)
    return sorted(dict.fromkeys(name for name in names if name and name != "object"))


def english_action_text(action: Dict) -> str:
    api = str(action.get("api") or action.get("action_api") or "").strip()
    params = {}
    if isinstance(action.get("action_args"), dict):
        params.update(action["action_args"])
    if isinstance(action.get("action_params"), dict):
        params.update(action["action_params"])
    degrees = action.get("degrees")
    if degrees is None:
        degrees = params.get("degrees")
    magnitude = action.get("moveMagnitude")
    if magnitude is None:
        magnitude = params.get("moveMagnitude")
    label = ACTION_API_EN.get(api, _normalize_spaces(api))
    if api in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
        return f"{label} {_fmt_num(degrees, 0)} deg"
    if api in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
        return f"{label} {_fmt_num(magnitude)} m"
    raw = _normalize_spaces(str(action.get("text") or action.get("action_text") or ""))
    return raw or label


def canonical_action_text(action: Dict) -> str:
    api = str(action.get("api") or action.get("action_api") or "").strip()
    params = {}
    if isinstance(action.get("action_args"), dict):
        params.update(action["action_args"])
    if isinstance(action.get("action_params"), dict):
        params.update(action["action_params"])
    if api in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
        return f"{api} {_fmt_num(params.get('degrees'), 0)} deg"
    if api in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
        return f"{api} {_fmt_num(params.get('moveMagnitude'), 2)} m"
    return api or english_action_text(action)


def _canonical_action_answer(actions: Sequence[Dict]) -> str:
    parts = [canonical_action_text(action) for action in actions or [] if canonical_action_text(action)]
    if not parts:
        return ""
    return " | ".join(f"step{i + 1}={part}" for i, part in enumerate(parts))


def _split_sequence(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"->|,", _normalize_spaces(text)) if part.strip()]


def _gt_sequence_labels(gt: Dict) -> List[str]:
    labels = gt.get("chronological_labels")
    if isinstance(labels, list) and labels:
        return [str(item).strip() for item in labels if str(item).strip()]
    return _split_sequence(gt.get("correct_label_sequence", ""))


def _rotation_delta(before: Dict, after: Dict) -> str:
    try:
        value = abs((float(after.get("rotation_y", 0.0)) - float(before.get("rotation_y", 0.0)) + 180.0) % 360.0 - 180.0)
    except Exception:
        value = 0.0
    return _fmt_num(value, 0)


def _transition_text(gt: Dict) -> str:
    before = gt.get("target_state_before", {}) or {}
    after = gt.get("target_state_after", {}) or {}
    change_category = str(gt.get("change_category") or "")
    if change_category == "rotate":
        return f"rotated { _rotation_delta(before, after) } deg".replace("  ", " ")
    if change_category == "open_close":
        before_state = "open" if before.get("is_open") else "closed"
        after_state = "open" if after.get("is_open") else "closed"
        return f"{before_state}->{after_state}"
    if change_category == "toggle":
        before_state = "on" if before.get("is_toggled") else "off"
        after_state = "on" if after.get("is_toggled") else "off"
        return f"{before_state}->{after_state}"
    if change_category == "remove":
        return "removed"
    return _normalize_spaces(str(gt.get("change_text") or "changed"))


def _direction_option_set(correct: str) -> List[str]:
    wheel = [
        "front",
        "front-right",
        "right",
        "back-right",
        "back",
        "back-left",
        "left",
        "front-left",
    ]
    if correct not in wheel:
        return [correct, "front", "left", "right"]
    idx = wheel.index(correct)
    return [correct, wheel[(idx + 2) % 8], wheel[(idx + 4) % 8], wheel[(idx - 2) % 8]]


def _sequence_distractors(correct_labels: Sequence[str]) -> List[str]:
    labels = list(correct_labels)
    if len(labels) < 4:
        return []
    candidates = [
        list(reversed(labels)),
        [labels[1], labels[0], labels[2], labels[3]],
        [labels[0], labels[2], labels[1], labels[3]],
        [labels[0], labels[1], labels[3], labels[2]],
        [labels[1], labels[2], labels[3], labels[0]],
        [labels[3], labels[0], labels[1], labels[2]],
    ]
    result: List[str] = []
    seen = {" -> ".join(labels)}
    for candidate in candidates:
        text = " -> ".join(candidate)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _build_options(correct_text: str, distractors: Sequence[str], sample_id: str, *, salt: str) -> Tuple[List[Dict[str, str]], str]:
    texts = _stable_sort_texts([correct_text, *distractors], sample_id, salt=salt)[:4]
    if correct_text not in texts:
        texts = texts[:3] + [correct_text]
        texts = _stable_sort_texts(texts, sample_id, salt=f"{salt}_repair")
    options = [{"key": chr(ord("A") + idx), "text": text} for idx, text in enumerate(texts)]
    correct_key = next(option["key"] for option in options if option["text"] == correct_text)
    return options, correct_key


def _answer_style(row: Dict) -> str:
    if str(row.get("task_type") or "") in MCQ_TASKS:
        return "mcq"
    return "short_answer"


def _frame_prefix(row: Dict) -> str:
    task_type = str(row.get("task_type") or "")
    frame_paths = list((row.get("input") or {}).get("frame_paths") or [])
    if task_type == "movement_sequence_sorting":
        return "Image order: image 1 is frame A, image 2 is frame B, image 3 is frame C, image 4 is frame D."
    labels = [chr(ord("A") + idx) for idx in range(len(frame_paths))]
    if not labels:
        return "Use the provided image(s) in order."
    label_text = ", ".join(f"image {idx + 1}=frame {label}" for idx, label in enumerate(labels))
    return f"Image order: {label_text}."


def _short_answer_instruction(row: Dict) -> str:
    task_type = str(row.get("task_type") or "")
    subcat = infer_subcat(row)
    if task_type == "action_inference":
        base = (
            "Choose actions only from: MoveAhead, MoveBack, MoveLeft, MoveRight, "
            "RotateLeft, RotateRight, LookUp, LookDown. "
            "Use unit m for translation and deg for rotation or camera tilt. "
            "Numeric formatting rules: use exactly two decimals for meters (example: 0.60 m) "
            "and use integer degrees with no decimals (example: 45 deg). "
            "Reply with only the final answer in the exact format shown below. Do not add explanation. "
        )
        if subcat == "two_image_single_action":
            return base + "Format: step1=<ACTION> <VALUE> <UNIT>. Example: step1=RotateLeft 45 deg"
        if subcat == "two_image_double_action":
            return (
                base
                + "Format: step1=<ACTION> <VALUE> <UNIT> | step2=<ACTION> <VALUE> <UNIT>. "
                + "Example: step1=RotateLeft 45 deg | step2=MoveAhead 0.60 m"
            )
        return (
            base
            + "Format: step1=<ACTION> <VALUE> <UNIT> | step2=<ACTION> <VALUE> <UNIT> | "
            + "step3=<ACTION> <VALUE> <UNIT>. "
            + "Example: step1=RotateLeft 45 deg | step2=RotateLeft 60 deg | step3=MoveAhead 0.60 m"
        )
    if task_type == "movement_degree_comparison":
        return (
            "Reply with only the final answer in the exact format shown below. Do not add explanation. "
            "Numeric formatting rules: use exactly two decimals for meters and integer degrees with no decimals. "
            "Format: larger=frame B | frame_B=<VALUE> <UNIT> | frame_C=<VALUE> <UNIT> "
            "or larger=frame C | frame_B=<VALUE> <UNIT> | frame_C=<VALUE> <UNIT>. "
            "Use the same unit for frame_B and frame_C. Example: "
            "larger=frame B | frame_B=0.50 m | frame_C=0.25 m"
        )
    if task_type == "object_state_attribute_changes":
        return (
            'Reply with only the final answer in the exact format shown below. Do not add explanation. '
            'Format: object="<NAME>" | change="<CHANGE>". '
            'Allowed changes: open->closed, closed->open, off->on, on->off, removed, rotated <VALUE> deg. '
            'Example: object="microwave" | change="closed->open"'
        )
    if task_type == "object_position_swapping":
        return (
            'Reply with only the final answer in the exact format shown below. Do not add explanation. '
            'Format: object_a="<NAME>" | object_b="<NAME>". Sort the two names alphabetically. '
            'Example: object_a="kettle" | object_b="pan"'
        )
    return "Reply with only the final answer. Do not explain."


def _build_prompt(question: str, row: Dict, answer_style: str, options: Optional[List[Dict[str, str]]]) -> str:
    lines = [_frame_prefix(row), question]
    candidate_objects = _candidate_object_names(row)
    if candidate_objects:
        lines.append(
            "Canonical object names that may be referenced: "
            + ", ".join(f'"{name}"' for name in candidate_objects[:16])
            + "."
        )
    if answer_style == "mcq" and options:
        lines.append("Choose exactly one option: A, B, C, or D.")
        lines.append("Options:")
        lines.extend(f'{item["key"]}. {item["text"]}' for item in options)
        lines.append("Reply with only one letter. Do not copy the option text.")
        return "\n".join(lines)
    lines.append(_short_answer_instruction(row))
    return "\n".join(lines)


def _build_question_package(row: Dict, question: str, answer: str, sample_id: str) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    task_type = str(row.get("task_type") or "")
    gt = row.get("gt", {}) or {}
    subcat = infer_subcat(row)

    if task_type == "multi_image_overlap_localization":
        correct = _label_to_english(gt.get("relation", "") or row.get("answer", ""))
        options, correct_key = _build_options(correct, _direction_option_set(correct)[1:], sample_id, salt="class02")
        return options, correct_key

    if task_type == "parallax_depth_inference":
        objs = gt.get("objects", {}) or {}
        object_a = _name((objs.get("object_a", {}) or {}).get("name"))
        object_b = _name((objs.get("object_b", {}) or {}).get("name"))
        correct = _name((gt.get("nearer_object", {}) or {}).get("name"))
        distractors = [object_a, object_b, "same depth", "cannot tell"]
        distractors = [item for item in distractors if item != correct]
        options, correct_key = _build_options(correct, distractors, sample_id, salt="class03")
        return options, correct_key

    if task_type == "movement_sequence_sorting":
        correct_labels = _gt_sequence_labels(gt)
        correct = " -> ".join(correct_labels)
        options, correct_key = _build_options(correct, _sequence_distractors(correct_labels), sample_id, salt="class04")
        return options, correct_key

    if task_type == "dynamic_movement_occlusion":
        if subcat == "distance_change":
            correct = answer
            options, correct_key = _build_options(correct, ["closer", "farther", "same distance", "cannot tell"], sample_id, salt="class08_distance")
            return options, correct_key
        options, correct_key = _build_options(answer, ["more visible", "less visible", "no visibility change", "cannot tell"], sample_id, salt="class08_occlusion")
        return options, correct_key

    if task_type == "imagined_perspective_taking":
        correct = _label_to_english(gt.get("direction_word", ""))
        options, correct_key = _build_options(correct, _direction_option_set(correct)[1:], sample_id, salt="class09")
        return options, correct_key

    if task_type == "imagined_movement_consequence":
        family = str(gt.get("question_family") or "")
        if family in {"single_direction", "two_frame_direction"}:
            correct = _label_to_english(gt.get("direction_word", "") or gt.get("result_location", "") or row.get("answer", ""))
            options, correct_key = _build_options(correct, _direction_option_set(correct)[1:], sample_id, salt="class10_direction")
            return options, correct_key
        if family == "single_visibility":
            options, correct_key = _build_options(
                answer,
                ["not visible", "visible | left", "visible | center", "visible | right"],
                sample_id,
                salt="class10_visibility",
            )
            return options, correct_key
        if family == "trend_disappear":
            target = _name(gt.get("answer_object", {}))
            other = _name(gt.get("comparison_object", {}))
            options, correct_key = _build_options(answer, [target, other, "both at the same time", "neither"], sample_id, salt="class10_trend")
            return options, correct_key

    return None, None


def infer_subcat(row: Dict) -> str:
    gt = row.get("gt", {}) or {}
    task_type = row.get("task_type", "")
    if task_type == "action_inference":
        return str(row.get("sample_variant") or gt.get("variant") or "default")
    if task_type == "movement_degree_comparison":
        return str(gt.get("comparison_type") or "default")
    if task_type == "object_state_attribute_changes":
        return str(gt.get("change_category") or "default")
    if task_type == "dynamic_movement_occlusion":
        return str(gt.get("question_family") or "default")
    if task_type == "imagined_movement_consequence":
        return str(gt.get("question_family") or "default")
    if task_type == "movement_sequence_sorting":
        mode = str(gt.get("start_visibility_mode") or "").strip()
        if mode in {"start_visible", "start_not_visible"}:
            return mode
    return "default"


def rebuild_answer(row: Dict) -> str:
    task_type = row.get("task_type", "")
    gt = row.get("gt", {}) or {}

    if task_type == "action_inference":
        return _canonical_action_answer(gt.get("actions", [])) or str(row.get("answer", ""))

    if task_type == "multi_image_overlap_localization":
        return _label_to_english(gt.get("relation", "") or row.get("answer", "")) or str(row.get("answer", ""))

    if task_type == "parallax_depth_inference":
        nearer = gt.get("nearer_object", {}) or {}
        return _name(nearer.get("name") or nearer.get("id"))

    if task_type == "movement_sequence_sorting":
        labels = _gt_sequence_labels(gt)
        if labels:
            return " -> ".join(labels)

    if task_type == "movement_degree_comparison":
        larger = str(gt.get("larger_image", row.get("answer", "B"))).strip().replace("图", "").upper()
        frame_b = gt.get("frame_B", {}) or {}
        frame_c = gt.get("frame_C", {}) or {}
        frame_b_text = _measurement_text(frame_b.get("actual_value"), frame_b.get("unit"))
        frame_c_text = _measurement_text(frame_c.get("actual_value"), frame_c.get("unit"))
        return f"larger=frame {larger} | frame_B={frame_b_text} | frame_C={frame_c_text}"

    if task_type == "object_state_attribute_changes":
        target = gt.get("target", {}) or {}
        return f'object="{_name(target)}" | change="{_transition_text(gt)}"'

    if task_type == "object_position_swapping":
        pairs = gt.get("swap_pair", []) or []
        if len(pairs) >= 2:
            names = sorted(
                [
                    _name(pairs[0].get("answer_name") or pairs[0].get("before")),
                    _name(pairs[1].get("answer_name") or pairs[1].get("before")),
                ]
            )
            return f'object_a="{names[0]}" | object_b="{names[1]}"'

    if task_type == "dynamic_movement_occlusion":
        family = str(gt.get("question_family") or "")
        if family == "distance_change":
            relation = _label_to_english(gt.get("relation", ""))
            if relation in {"nearer", "closer"}:
                return "closer"
            if relation in {"farther", "further"}:
                return "farther"
            return relation or str(row.get("answer", ""))
        if family == "occlusion_change":
            change_type = str(gt.get("change_type") or "").strip().lower()
            if change_type == "easier":
                return "more visible"
            if change_type == "harder":
                return "less visible"
            return str(row.get("answer", ""))

    if task_type == "imagined_perspective_taking":
        return _label_to_english(gt.get("direction_word", "")) or str(row.get("answer", ""))

    if task_type == "imagined_movement_consequence":
        family = str(gt.get("question_family") or "")
        if family in {"single_direction", "two_frame_direction"}:
            return _label_to_english(gt.get("direction_word", "") or gt.get("result_location", "") or row.get("answer", "")) or str(row.get("answer", ""))
        if family == "single_visibility":
            visible = bool(gt.get("result_visibility", False))
            if visible:
                loc = _label_to_english(gt.get("result_location", "center")) or "center"
                return f"visible | {loc}"
            return "not visible"
        if family == "trend_disappear":
            target = gt.get("answer_object", {}) or {}
            return _name(target)

    return str(row.get("answer", ""))


def question_variants(row: Dict) -> List[str]:
    task_type = row.get("task_type", "")
    gt = row.get("gt", {}) or {}
    subcat = infer_subcat(row)

    if task_type == "action_inference":
        if subcat == "two_image_single_action":
            return [
                "Exactly one camera action happened from frame A to frame B. Identify that one action.",
            ]
        if subcat == "two_image_double_action":
            return [
                "Exactly two camera actions happened from frame A to frame B, in this order. Identify both actions.",
            ]
        return [
            "Between each adjacent pair of frames, exactly one camera action happened. "
            "That means A->B is one action, B->C is one action, and C->D is one action. "
            "List all actions in order.",
        ]

    if task_type == "multi_image_overlap_localization":
        objs = gt.get("objects", {}) or {}
        target = objs.get("target_object_from_C", {}) or {}
        anchor = objs.get("anchor_object_from_A", {}) or {}
        question_names = _question_object_names(str(row.get("question", "")))
        if not target and question_names:
            target = question_names[0]
        if not anchor and len(question_names) >= 2:
            anchor = question_names[1]
        target_name = _quote(target)
        anchor_name = _quote(anchor)
        return [
            f"Use the shared visible region across frames A, B, and C to align the scene. "
            f"Relative to {anchor_name} in frame A, where is {target_name} in frame C?",
        ]

    if task_type == "parallax_depth_inference":
        objs = gt.get("objects", {}) or {}
        name_a = _quote((objs.get("object_a", {}) or {}).get("name"))
        name_b = _quote((objs.get("object_b", {}) or {}).get("name"))
        return [
            f"Frames A and B show the same scene from nearby viewpoints. "
            f"Using parallax cues, which object is closer to the camera: {name_a} or {name_b}?",
        ]

    if task_type == "movement_sequence_sorting":
        return [
            "These four frames come from one continuous camera motion in a single trajectory. "
            "Sort A, B, C, and D from earliest to latest.",
        ]

    if task_type == "movement_degree_comparison":
        return [
            "Frame A is the start view. Frame B and frame C each apply one action of the same type from frame A, "
            "but with different magnitudes. Decide which magnitude is larger, then estimate the magnitude for "
            "frame B and frame C.",
        ]

    if task_type == "object_state_attribute_changes":
        return [
            "Exactly one named object changed state or attribute from frame A to frame B. "
            "Which object changed, and what was the change?",
        ]

    if task_type == "object_position_swapping":
        return [
            "Exactly two named objects exchanged positions between frame A and frame B. "
            "Which two named objects were swapped?",
        ]

    if task_type == "dynamic_movement_occlusion":
        mover = _quote(((gt.get("mover", {}) or {}).get("before", {}) or {}).get("name"))
        if subcat == "distance_change":
            return [
                f"The moving object is {mover}. Compare its depth relative to the observer in frame A and frame B. "
                f"Did it move closer, farther, stay at the same distance, or is it impossible to tell?",
            ]
        affected = _quote((gt.get("affected_object", {}) or {}).get("before", {}))
        return [
            f"A scene object moved between frame A and frame B. Because of that motion, did {affected} become "
            f"more visible, less visible, show no visibility change, or is it impossible to tell?",
        ]

    if task_type == "imagined_perspective_taking":
        obj_a = _quote((gt.get("object_A", {}) or {}).get("name"))
        obj_b = _quote((gt.get("object_B", {}) or {}).get("name"))
        return [
            f"Use {obj_a} as the observer. Imagine you are standing at that object and facing the same forward "
            f"direction as it. Where is {obj_b} relative to you?",
        ]

    if task_type == "imagined_movement_consequence":
        target = _quote((gt.get("target", {}) or gt.get("answer_object", {}) or {}).get("name"))
        action_text = ", then ".join(_fmt_action_texts(gt.get("action_sequence", []))) or _normalize_spaces(gt.get("action_text", ""))
        if subcat == "single_visibility":
            return [
                f"Start from the viewpoint of frame A. After this motion sequence: {action_text}, "
                f"is {target} still visible? If yes, where does it appear in the image?",
            ]
        if subcat == "two_frame_direction":
            return [
                f"Frame A is the start viewpoint. Frame B is the viewpoint after one motion sequence. "
                f"If you are now at the viewpoint of frame B, where is {target} relative to you?",
            ]
        if subcat == "trend_disappear":
            other = _quote((gt.get("comparison_object", {}) or {}).get("name"))
            return [
                f"Frames A and B show one continuing motion trend. If the same trend continues for one more step, "
                f"which object leaves the view first: {target} or {other}?",
            ]
        return [
            f"Start from the viewpoint of frame A. After this motion sequence: {action_text}, "
            f"where is {target} relative to you?",
        ]

    return [str(row.get("question", ""))]


def scoring_rubric(row: Dict) -> Dict:
    task_type = str(row.get("task_type") or "")
    subcat = infer_subcat(row)

    if task_type == "action_inference":
        return {
            "type": "structured",
            "strict_fields": ["step_order", "action_api"],
            "continuous_fields": [{"name": "action_value", "unit": "deg_or_m", "grading": "tolerance"}],
        }
    if task_type == "movement_degree_comparison":
        return {
            "type": "structured",
            "strict_fields": ["larger"],
            "continuous_fields": [
                {"name": "frame_B", "grading": "tolerance"},
                {"name": "frame_C", "grading": "tolerance"},
            ],
        }
    if task_type == "object_state_attribute_changes":
        transition = "transition"
        if subcat == "rotate":
            return {
                "type": "structured",
                "strict_fields": ["object"],
                "continuous_fields": [{"name": transition, "unit": "deg", "grading": "tolerance"}],
            }
        return {"type": "structured", "strict_fields": ["object", transition], "continuous_fields": []}
    if task_type == "object_position_swapping":
        return {"type": "structured", "strict_fields": ["object_a", "object_b"], "continuous_fields": []}
    if task_type in MCQ_TASKS:
        return {"type": "mcq", "strict_fields": ["option"], "continuous_fields": []}
    return {"type": "short_answer", "strict_fields": ["answer"], "continuous_fields": []}


def generate_cot(row: Dict, normalized_answer: str) -> Tuple[List[str], str]:
    task_type = row.get("task_type", "")
    gt = row.get("gt", {}) or {}

    if task_type == "action_inference":
        texts = _fmt_action_texts(gt.get("actions", []))
        steps = [
            "Compare adjacent frames and infer whether each change comes from translation, rotation, or camera tilt.",
            f"The viewpoint changes are consistent with: {'; '.join(texts)}.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "multi_image_overlap_localization":
        objs = gt.get("objects", {}) or {}
        target = _quote(objs.get("target_object_from_C", {}))
        anchor = _quote(objs.get("anchor_object_from_A", {}))
        relation = _label_to_english(gt.get("relation", ""))
        steps = [
            f"Use frame A as the reference view and align the shared content across the three frames.",
            f"After alignment, {target} falls in the {relation} direction relative to {anchor}.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "parallax_depth_inference":
        objs = gt.get("objects", {}) or {}
        obj_a = objs.get("object_a", {}) or {}
        obj_b = objs.get("object_b", {}) or {}
        dist_a = _fmt_num(obj_a.get("distance_to_camera_A"))
        dist_b = _fmt_num(obj_b.get("distance_to_camera_A"))
        steps = [
            f"{_quote(obj_a)} shows stronger parallax and lies nearer to the camera than {_quote(obj_b)}.",
            f"The relative geometry is consistent with distances of about {dist_a} m versus {dist_b} m.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "movement_sequence_sorting":
        labels = _gt_sequence_labels(gt)
        steps = [
            "Track the viewpoint change and scene progression to recover the temporal order.",
            f"The smooth progression of pose and scene layout matches the order: {', '.join(labels)}.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "movement_degree_comparison":
        frame_b = gt.get("frame_B", {}) or {}
        frame_c = gt.get("frame_C", {}) or {}
        steps = [
            f"Frame B applies {english_action_text(frame_b)} and frame C applies {english_action_text(frame_c)}.",
            f"They use the same action type, but the magnitudes are { _fmt_num(frame_b.get('actual_value')) } and { _fmt_num(frame_c.get('actual_value')) }.".replace("  ", " "),
            f"So the larger-magnitude frame is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "object_state_attribute_changes":
        target = _quote(gt.get("target", {}))
        before = gt.get("target_state_before", {}) or {}
        after = gt.get("target_state_after", {}) or {}
        summary = []
        if "is_open" in before or "is_open" in after:
            summary.append(f"open state {before.get('is_open')} -> {after.get('is_open')}")
        if "is_toggled" in before or "is_toggled" in after:
            summary.append(f"toggle state {before.get('is_toggled')} -> {after.get('is_toggled')}")
        if "rotation_y" in before and "rotation_y" in after:
            summary.append(f"rotation {round(float(before.get('rotation_y', 0.0)), 1)} -> {round(float(after.get('rotation_y', 0.0)), 1)}")
        steps = [
            f"Compare the two frames and inspect state changes for named objects, especially {target}.",
            f"The visual change is consistent with: {'; '.join(summary) if summary else _transition_text(gt)}.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "object_position_swapping":
        pairs = gt.get("swap_pair", []) or []
        if len(pairs) >= 2:
            a = _quote(pairs[0].get("answer_name") or pairs[0].get("before"))
            b = _quote(pairs[1].get("answer_name") or pairs[1].get("before"))
        else:
            a = _quote("object A")
            b = _quote("object B")
        steps = [
            f"Identify the two salient objects that may have changed places: {a} and {b}.",
            "Compare their support surfaces and relative positions across the two frames.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "dynamic_movement_occlusion":
        family = str(gt.get("question_family") or "")
        mover = _quote(((gt.get("mover", {}) or {}).get("before", {}) or {}).get("name"))
        if family == "distance_change":
            before = _fmt_num(gt.get("distance_before"))
            after = _fmt_num(gt.get("distance_after"))
            relation = _label_to_english(gt.get("relation", ""))
            steps = [
                f"The moving object is {mover}.",
                f"It ends up at a different depth, changing from about {before} m to {after} m relative to the observer.",
                f"That means it becomes {relation}.",
                f"So the final answer is: {normalized_answer}",
            ]
            return steps, " ".join(steps)
        affected = _quote(((gt.get("affected_object", {}) or {}).get("before", {}) or {}).get("name"))
        before_overlap = _fmt_num(gt.get("overlap_before"))
        after_overlap = _fmt_num(gt.get("overlap_after"))
        steps = [
            f"The moving object is {mover}, and the affected object is {affected}.",
            f"The motion changes the amount of overlap from {before_overlap} to {after_overlap}, altering how much of {affected} remains visible.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "imagined_perspective_taking":
        obj_a = _quote((gt.get("object_A", {}) or {}).get("name"))
        obj_b = _quote((gt.get("object_B", {}) or {}).get("name"))
        pos_local = gt.get("pos_B_local", []) or [0.0, 0.0, 0.0]
        right = _fmt_num(pos_local[0] if len(pos_local) > 0 else 0.0)
        forward = _fmt_num(pos_local[2] if len(pos_local) > 2 else 0.0)
        steps = [
            f"Use the position and facing direction of {obj_a} as the new local coordinate system.",
            f"From that local viewpoint, {obj_b} lies at about X={right}, Z={forward}, which places it in the {_label_to_english(gt.get('direction_word', ''))} direction.",
            f"So the final answer is: {normalized_answer}",
        ]
        return steps, " ".join(steps)

    if task_type == "imagined_movement_consequence":
        family = str(gt.get("question_family") or "")
        action_text = ", then ".join(_fmt_action_texts(gt.get("action_sequence", []))) or _normalize_spaces(gt.get("action_text", ""))
        if family == "single_visibility":
            steps = [
                f"Mentally execute the motion sequence: {action_text}.",
                f"After that motion, the target is {'still visible' if bool(gt.get('result_visibility', False)) else 'no longer visible'}"
                + (
                    f" and appears near the {_label_to_english(gt.get('result_location', ''))} of the frame."
                    if bool(gt.get('result_visibility', False))
                    else "."
                ),
                f"So the final answer is: {normalized_answer}",
            ]
            return steps, " ".join(steps)
        if family in {"single_direction", "two_frame_direction"}:
            steps = [
                f"Update the observer position and heading according to the sequence: {action_text}.",
                f"From the new viewpoint, the target falls in the {_label_to_english(gt.get('direction_word', ''))} direction.",
                f"So the final answer is: {normalized_answer}",
            ]
            return steps, " ".join(steps)
        if family == "trend_disappear":
            target = _quote(gt.get("answer_object", {}))
            other = _quote(gt.get("comparison_object", {}))
            steps = [
                f"Frames A and B imply a continuing motion trend: {action_text}.",
                f"If the trend continues one more step, compare {target} and {other} relative to the image boundary. {target} reaches the edge and leaves first.",
                f"So the final answer is: {normalized_answer}",
            ]
            return steps, " ".join(steps)

    return [], ""


def augment_row(row: Dict) -> Dict:
    sample_id = str(row.get("sample_id") or hashlib.md5(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest())
    variants = [item for item in question_variants(row) if item]
    chosen_question = variants[0] if variants else str(row.get("question", ""))
    normalized_answer = rebuild_answer(row)
    cot_steps, cot_text = generate_cot(row, normalized_answer)
    answer_style = _answer_style(row)
    options, correct_option = _build_question_package(row, chosen_question, normalized_answer, sample_id)
    prompt = _build_prompt(chosen_question or str(row.get("question", "")), row, answer_style, options)
    augmented = dict(row)
    augmented["question_original"] = row.get("question", "")
    augmented["answer_original"] = row.get("answer", "")
    augmented["question_variants"] = variants
    augmented["question_core"] = chosen_question or str(row.get("question", ""))
    augmented["question"] = f"{_frame_prefix(row)}\n{augmented['question_core']}"
    augmented["answer"] = normalized_answer or str(row.get("answer", ""))
    augmented["subcat"] = infer_subcat(row)
    augmented["question_language"] = "en"
    augmented["answer_language"] = "en"
    augmented["answer_style"] = answer_style
    augmented["answer_format"] = "multiple_choice" if answer_style == "mcq" else "structured_text"
    augmented["options"] = options
    augmented["correct_option"] = correct_option
    augmented["prompt"] = prompt
    augmented["prompt_cot"] = "\n".join(
        [
            prompt,
            "Reason step by step from the visible evidence and spatial relations only.",
            "Do not mention hidden metadata, annotations, or structured records.",
            "Then give the final answer in the required format.",
        ]
    )
    augmented["cot_capable"] = bool(cot_steps)
    augmented["cot_steps"] = cot_steps
    augmented["cot"] = cot_text
    augmented["scoring_rubric"] = scoring_rubric(row)
    augmented["augmentation_meta"] = {
        "version": "v5_explicit_prompting_with_canonical_names",
        "normalized_answer": augmented["answer"] != row.get("answer", ""),
        "diverse_question": False,
        "answer_style_changed": answer_style != "short_answer",
    }
    return augmented


def _iter_meta_jsonl_paths(input_root: Path) -> Iterable[Path]:
    for path in sorted(input_root.glob("*/[0-9]*/meta/qa_data.jsonl")):
        yield path


def augment_dataset_root(
    input_root: Path,
    *,
    output_stem: str = "qa_data_standardized",
    overwrite: bool = True,
    limit_files: Optional[int] = None,
) -> Dict[str, object]:
    input_root.mkdir(parents=True, exist_ok=True)
    processed_files = 0
    processed_rows = 0
    task_counter: Counter[str] = Counter()
    subcat_counter: Counter[str] = Counter()
    cot_counter: Counter[str] = Counter()

    for qa_jsonl_path in _iter_meta_jsonl_paths(input_root):
        if limit_files is not None and processed_files >= limit_files:
            break
        meta_root = qa_jsonl_path.parent
        output_jsonl = meta_root / f"{output_stem}.jsonl"
        output_json = meta_root / f"{output_stem}.json"
        if output_jsonl.exists() and not overwrite:
            continue

        rows: List[Dict] = []
        for line in qa_jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            augmented = augment_row(row)
            rows.append(augmented)
            processed_rows += 1
            task_counter[str(augmented.get("task_type", "unknown"))] += 1
            subcat_counter[str(augmented.get("subcat", "default"))] += 1
            if augmented.get("cot_capable"):
                cot_counter[str(augmented.get("task_type", "unknown"))] += 1

        output_jsonl.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        processed_files += 1

    summary = {
        "input_root": str(input_root),
        "output_stem": output_stem,
        "processed_files": processed_files,
        "processed_rows": processed_rows,
        "task_counts": dict(task_counter),
        "subcat_counts": dict(subcat_counter),
        "cot_counts": dict(cot_counter),
    }
    (input_root / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main_with_default_root(default_root: Path) -> int:
    parser = argparse.ArgumentParser(description="Augment generated QA files with diverse instructions and CoT.")
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-stem", type=str, default="qa_data_standardized")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    summary = augment_dataset_root(
        args.input_root,
        output_stem=args.output_stem,
        overwrite=not args.no_overwrite,
        limit_files=args.limit_files,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
