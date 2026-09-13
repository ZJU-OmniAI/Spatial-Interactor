from __future__ import annotations

import json
import math
import os
import re
from typing import Any


REWARD_NAME = "spatial_video_verifiable_reward"
REWARD_TYPE = "batch"
FORMAT_REWARD_WEIGHT = float(os.getenv("SPATIAL_INTERACTOR_FORMAT_REWARD_WEIGHT", "0.10"))
if not 0.0 <= FORMAT_REWARD_WEIGHT <= 1.0:
    raise ValueError("SPATIAL_INTERACTOR_FORMAT_REWARD_WEIGHT must be between 0 and 1.")

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_COT_RE = re.compile(r"<COT>(.*?)</COT>", re.I | re.S)
_NATIVE_ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+)$", re.I | re.S)
_LETTER_RE = re.compile(r"\b([A-E])\b", re.I)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or ""))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _answer_text(response: str) -> str:
    match = _ANSWER_RE.search(response or "")
    if match:
        return match.group(1).strip()
    match = _NATIVE_ANSWER_RE.search(response or "")
    return match.group(1).strip() if match else ""


def _parse_prediction(response: str) -> dict[str, Any]:
    text = _answer_text(response)
    if not text:
        # The supplied SFT checkpoint can still emit its legacy schema during
        # the first RL steps (for example `A` or `Answer: {"value": 1.4}`).
        # Preserve task reward for those rows while format reward remains zero.
        stripped = _COT_RE.sub("", response or "").strip()
        if re.fullmatch(r"[A-E]", stripped, re.I):
            return {"choice": stripped.upper()}
        legacy_choice = re.search(
            r"(?:answer|option|choice)\s*(?:is|:|=)\s*([A-E])\b",
            stripped,
            re.I,
        )
        if legacy_choice:
            return {"choice": legacy_choice.group(1).upper()}
        legacy_json = re.search(r"(?:answer\s*:\s*)?(\{[^{}]*\})", stripped, re.I | re.S)
        if legacy_json:
            try:
                parsed = json.loads(legacy_json.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        answer_tail = re.search(r"\banswer\s*:\s*(.+)$", stripped, re.I | re.S)
        if answer_tail:
            numbers = _NUMBER_RE.findall(answer_tail.group(1))
            if numbers:
                return {"value": float(numbers[-1])}
        legacy_value = re.search(
            r'"?(?:answer|distance|displacement|value)"?\s*(?:is|:|=)\s*'
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
            stripped,
            re.I,
        )
        if legacy_value:
            return {"value": float(legacy_value.group(1))}
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    if re.fullmatch(r"[A-E]", text, re.I):
        return {"choice": text.upper()}
    if re.fullmatch(_NUMBER_RE, text):
        return {"value": float(text)}

    # Retain semantic reward for an incomplete closing brace while keeping the
    # independent format score at zero.
    recovered: dict[str, Any] = {}
    for field, letter in re.findall(
        r'"?(choice|straight|angle|path)"?\s*:\s*"?([A-E])\b', text, flags=re.I
    ):
        recovered[field.lower()] = letter.upper()
    value_match = re.search(
        r'"?(?:value|value_m|distance_m)"?\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))',
        text,
        flags=re.I,
    )
    if value_match:
        recovered["value"] = float(value_match.group(1))
    return recovered


def _letter(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"[A-E]", text, re.I):
        return text.upper()
    match = _LETTER_RE.search(text)
    return match.group(1).upper() if match else None


def _number(prediction: dict[str, Any]) -> float | None:
    for key in ("value", "value_m", "distance_m", "answer"):
        if key not in prediction:
            continue
        try:
            value = float(prediction[key])
            return value if math.isfinite(value) else None
        except Exception:
            continue
    return None


def _format_scores(response: str, truth: dict[str, Any]) -> tuple[float, float, float]:
    cot_match = _COT_RE.search(response or "")
    native_answer_match = _NATIVE_ANSWER_RE.search(response or "")
    native_reasoning = (response or "")[: native_answer_match.start()].strip() if native_answer_match else ""
    cot_ok = float(bool((cot_match and cot_match.group(1).strip()) or native_reasoning))
    answer_match = _ANSWER_RE.search(response or "")
    answer_text = (
        answer_match.group(1).strip()
        if answer_match
        else native_answer_match.group(1).strip()
        if native_answer_match
        else ""
    )
    answer_format_ok = 0.0
    if answer_text:
        family = str(truth.get("reward_family") or "choice").lower()
        parsed = None
        try:
            parsed = json.loads(answer_text)
        except Exception:
            pass
        if isinstance(parsed, dict):
            if family == "multi_choice":
                fields = truth.get("fields") or {}
                answer_format_ok = float(
                    all(key in parsed for key in fields)
                )
            elif family == "numeric_mra":
                answer_format_ok = float(_number(parsed) is not None)
            else:
                answer_format_ok = float(
                    _letter(parsed.get("choice") or parsed.get("answer")) is not None
                )
        elif family == "numeric_mra":
            answer_format_ok = float(bool(re.fullmatch(_NUMBER_RE, answer_text)))
        else:
            answer_format_ok = float(bool(re.fullmatch(r"[A-E]", answer_text, re.I)))
    return cot_ok * answer_format_ok, cot_ok, answer_format_ok


def _task_score(prediction: dict[str, Any], truth: dict[str, Any]) -> dict[str, float]:
    family = str(truth.get("reward_family") or "choice").lower()
    if family == "numeric_mra":
        prediction_value = _number(prediction)
        try:
            gold = float(truth["value"])
        except Exception:
            return {"task": 0.0, "numeric_mra": 0.0}
        if prediction_value is None:
            score = 0.0
        else:
            score = max(0.0, 1.0 - abs(prediction_value - gold) / max(abs(gold), 1e-6))
        return {"task": score, "numeric_mra": score}

    if family == "multi_choice":
        fields = truth.get("fields") or {}
        field_scores = {
            str(key): float(_letter(prediction.get(key)) == _letter(gold))
            for key, gold in fields.items()
        }
        score = sum(field_scores.values()) / max(len(field_scores), 1)
        return {"task": score, "multi_choice": score, **field_scores}

    gold = _letter(truth.get("choice_gold") or truth.get("value"))
    pred = _letter(prediction.get("choice") or prediction.get("answer"))
    score = float(bool(gold and pred and gold == pred))
    return {"task": score, "accuracy": score}


def compute_score(reward_inputs: list[dict[str, Any]], **_: Any) -> list[dict[str, float]]:
    outputs: list[dict[str, float]] = []
    for item in reward_inputs:
        response = str(item.get("response") or "")
        truth = _loads(item.get("ground_truth"))
        prediction = _parse_prediction(response)
        scores = _task_score(prediction, truth)
        format_score, cot_format, answer_format = _format_scores(response, truth)
        overall = (1.0 - FORMAT_REWARD_WEIGHT) * scores["task"] + FORMAT_REWARD_WEIGHT * format_score
        outputs.append(
            {
                "overall": max(0.0, min(1.0, overall)),
                **scores,
                "format": format_score,
                "cot_format": cot_format,
                "answer_format": answer_format,
            }
        )
    return outputs


if __name__ == "__main__":
    demo = [
        {
            "response": '<COT>Objects grow larger.</COT><answer>{"value_m":3.0}</answer>',
            "ground_truth": json.dumps({"reward_family": "numeric_mra", "value": 2.9}),
        }
    ]
    print(json.dumps(compute_score(demo), indent=2))
