from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

from action_templates import (
    build_single_action_templates,
    random_chain_action_sequences,
    valid_double_action_pairs,
)
from schema import StateRecord, TaskProposal, VisibleObjectRecord
from .base import BaseEnumerator

TARGET_TOTAL = 15
TARGET_BY_SUBCAT = {
    "two_image_single_action": 5,
    "two_image_double_action": 5,
    "multi_image_single_action_chain": 5,
}


class Class01Enumerator(BaseEnumerator):
    class_id = 1
    subcategories = (
        "two_image_single_action",
        "two_image_double_action",
        "multi_image_single_action_chain",
    )

    def enumerate(self, state: StateRecord) -> List[TaskProposal]:
        if not bool(state.suitability.get("class01")):
            return []

        seed = _state_seed(state.state_id)
        context = _state_context(state)

        single_pool = [action for action in build_single_action_templates() if _single_action_allowed(action)]
        single_scored = [
            (_score_single_action(action, context), action)
            for action in single_pool
        ]

        pair_pool = [pair for pair in valid_double_action_pairs() if _double_action_allowed(pair)]
        pair_scored = [
            (_score_double_action(pair, context), pair)
            for pair in pair_pool
        ]

        chain_pool = [
            seq
            for seq in random_chain_action_sequences(
                seed ^ 0x5EED01,
                min_steps=3,
                max_steps=3,
                target_count=240,
            )
            if _chain_action_allowed(seq)
        ]
        chain_scored = [
            (_score_chain_action(seq, context), seq)
            for seq in chain_pool
        ]

        selected_by_subcat = {
            "two_image_single_action": _top_unique(
                single_scored,
                TARGET_BY_SUBCAT["two_image_single_action"],
            ),
            "two_image_double_action": _top_unique(
                pair_scored,
                TARGET_BY_SUBCAT["two_image_double_action"],
            ),
            "multi_image_single_action_chain": _top_unique(
                chain_scored,
                TARGET_BY_SUBCAT["multi_image_single_action_chain"],
            ),
        }

        current_total = sum(len(items) for items in selected_by_subcat.values())
        if current_total < TARGET_TOTAL:
            leftovers = []
            leftovers.extend(
                _leftovers(single_scored, selected_by_subcat["two_image_single_action"])
            )
            leftovers.extend(
                _leftovers(pair_scored, selected_by_subcat["two_image_double_action"])
            )
            leftovers.extend(
                _leftovers(chain_scored, selected_by_subcat["multi_image_single_action_chain"])
            )
            extras = _top_unique(leftovers, TARGET_TOTAL - current_total)
            for score, payload in extras:
                if isinstance(payload, list):
                    if payload and isinstance(payload[0], dict) and "family" in payload[0]:
                        if len(payload) == 1:
                            selected_by_subcat["two_image_single_action"].append((score, payload[0]))
                        elif len(payload) == 2:
                            selected_by_subcat["two_image_double_action"].append((score, payload))
                        else:
                            selected_by_subcat["multi_image_single_action_chain"].append((score, payload))
                else:
                    selected_by_subcat["two_image_single_action"].append((score, payload))

        proposals: List[TaskProposal] = []
        proposals.extend(
            self._proposal(
                state,
                subcat="two_image_single_action",
                score=score,
                payload={
                    "mode": "action_execution_search",
                    "variant": "two_image_single_action",
                    "actions": [action],
                },
            )
            for score, action in _sorted_selected(selected_by_subcat["two_image_single_action"])
        )
        proposals.extend(
            self._proposal(
                state,
                subcat="two_image_double_action",
                score=score,
                payload={
                    "mode": "action_execution_search",
                    "variant": "two_image_double_action",
                    "actions": pair,
                },
            )
            for score, pair in _sorted_selected(selected_by_subcat["two_image_double_action"])
        )
        proposals.extend(
            self._proposal(
                state,
                subcat="multi_image_single_action_chain",
                score=score,
                payload={
                    "mode": "action_execution_search",
                    "variant": "multi_image_single_action_chain",
                    "actions": seq,
                },
            )
            for score, seq in _sorted_selected(selected_by_subcat["multi_image_single_action_chain"])
        )
        return proposals[:TARGET_TOTAL]


def _state_seed(state_id: str) -> int:
    digest = hashlib.md5(state_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _sorted_selected(items: List[Tuple[float, object]]) -> List[Tuple[float, object]]:
    return sorted(items, key=lambda item: item[0], reverse=True)


def _leftovers(
    full_scored: Sequence[Tuple[float, object]],
    selected: Sequence[Tuple[float, object]],
) -> List[Tuple[float, object]]:
    selected_keys = {_item_key(item) for _, item in selected}
    return [(score, item) for score, item in full_scored if _item_key(item) not in selected_keys]


def _item_key(item: object) -> Tuple[object, ...]:
    if isinstance(item, dict):
        return (
            str(item.get("action_api")),
            tuple(sorted(dict(item.get("action_params", {})).items())),
        )
    if isinstance(item, list):
        return tuple(_item_key(part) for part in item)
    raise TypeError(f"Unsupported item type: {type(item)!r}")


def _top_unique(
    scored_items: Sequence[Tuple[float, object]],
    limit: int,
) -> List[Tuple[float, object]]:
    if limit <= 0 or not scored_items:
        return []
    selected: List[Tuple[float, object]] = []
    seen = set()
    for score, item in sorted(scored_items, key=lambda pair: pair[0], reverse=True):
        key = _item_key(item)
        if key in seen:
            continue
        selected.append((score, item))
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _state_context(state: StateRecord) -> Dict[str, float]:
    visible = list(state.visible_objects)
    focus = _focus_object(visible)
    near_count = float(state.metrics.get("near_count", 0))
    far_count = float(state.metrics.get("far_count", 0))
    actionable_count = float(state.metrics.get("actionable_count", 0))
    visible_count = float(state.metrics.get("visible_count", len(visible)))

    focus_distance = focus.distance if focus is not None else 1.2
    focus_x = focus.center_x_ratio if focus is not None and focus.center_x_ratio is not None else 0.5
    focus_y = focus.center_y_ratio if focus is not None and focus.center_y_ratio is not None else 0.5
    focus_area = focus.bbox_area_ratio if focus is not None and focus.bbox_area_ratio is not None else 0.02
    salient_count = sum(
        1 for obj in visible if obj.bbox_area_ratio is not None and obj.bbox_area_ratio >= 0.01
    )

    return {
        "near_count": near_count,
        "far_count": far_count,
        "actionable_count": actionable_count,
        "visible_count": visible_count,
        "focus_distance": focus_distance,
        "focus_x": focus_x,
        "focus_y": focus_y,
        "focus_area": focus_area,
        "salient_count": float(salient_count),
    }


def _focus_object(visible: Sequence[VisibleObjectRecord]) -> VisibleObjectRecord | None:
    ranked = []
    for obj in visible:
        if obj.bbox_area_ratio is None or obj.center_x_ratio is None or obj.center_y_ratio is None:
            continue
        center_bias = abs(obj.center_x_ratio - 0.5) + abs(obj.center_y_ratio - 0.5)
        rank = (
            obj.bbox_area_ratio * 4.0
            - center_bias
            - max(0.0, obj.distance - 1.2) * 0.25
            + (0.6 if (obj.pickupable or obj.moveable or obj.openable or obj.toggleable) else 0.0)
        )
        ranked.append((rank, obj))
    if not ranked:
        return visible[0] if visible else None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _score_single_action(action: Dict[str, object], context: Dict[str, float]) -> float:
    family = str(action["family"])
    action_api = str(action["action_api"])
    score = 1.0

    focus_distance = context["focus_distance"]
    focus_x = context["focus_x"]
    focus_y = context["focus_y"]
    near_count = context["near_count"]
    far_count = context["far_count"]
    salient_count = context["salient_count"]

    if family == "move":
        magnitude = float(action["action_params"].get("moveMagnitude", 0.0))
        score += 0.5 + 0.9 * magnitude
        if action_api == "MoveAhead":
            score += 2.2 if focus_distance >= 1.15 else -0.6
            score -= 1.0 if near_count >= 2 else 0.0
            score += 0.3 if far_count >= 2 else 0.0
        elif action_api == "MoveBack":
            score += 1.2 if near_count >= 2 else 0.1
            score -= 0.7 if focus_distance >= 1.6 else 0.0
        else:
            score += 1.4 if salient_count >= 2 else 0.4
            score += 0.7 if 0.32 <= focus_x <= 0.68 else -0.2
        if magnitude > 0.6:
            score -= 1.4
    elif family == "rotate":
        degrees = float(action["action_params"].get("degrees", 0.0))
        score += 1.3
        score += 1.1 if degrees in {45.0, 60.0} else 0.1
        if action_api == "RotateLeft":
            score += 1.2 if focus_x <= 0.45 else 0.2
        else:
            score += 1.2 if focus_x >= 0.55 else 0.2
        if 0.42 <= focus_x <= 0.58:
            score -= 0.4
    else:
        degrees = float(action["action_params"].get("degrees", 0.0))
        score += 0.4
        score += 0.9 if degrees == 30.0 else -1.2
        if action_api == "LookUp":
            score += 0.7 if focus_y <= 0.42 else -0.6
        else:
            score += 0.8 if focus_y >= 0.6 else -0.4
        if 0.4 <= focus_y <= 0.6:
            score -= 0.6
    return score


def _score_double_action(pair: Sequence[Dict[str, object]], context: Dict[str, float]) -> float:
    first, second = pair
    score = _score_single_action(first, context) + _score_single_action(second, context)
    first_family = str(first["family"])
    second_family = str(second["family"])
    if first_family != second_family:
        score += 1.1
    if {"move", "rotate"} == {first_family, second_family}:
        score += 1.4
    if {"move", "look"} == {first_family, second_family}:
        score += 0.2
    if {"rotate", "look"} == {first_family, second_family}:
        score += 0.1
    if first_family == second_family == "move":
        score -= 1.8
    score += 0.2 * len({str(first["action_api"]), str(second["action_api"])})
    return score


def _score_chain_action(seq: Sequence[Dict[str, object]], context: Dict[str, float]) -> float:
    score = sum(_score_single_action(action, context) for action in seq)
    families = [str(action["family"]) for action in seq]
    score += 0.7 * len(set(families))
    score += 0.35 * len(set(str(action["action_api"]) for action in seq))
    if families.count("move") >= 2:
        score -= 1.2
    if families.count("look") >= 2:
        score -= 1.0
    return score


def _single_action_allowed(action: Dict[str, object]) -> bool:
    family = str(action["family"])
    if family == "move":
        return float(action["action_params"].get("moveMagnitude", 0.0)) <= 0.6
    if family == "rotate":
        return float(action["action_params"].get("degrees", 0.0)) in {45.0, 60.0}
    return float(action["action_params"].get("degrees", 0.0)) == 30.0


def _double_action_allowed(pair: Sequence[Dict[str, object]]) -> bool:
    first, second = pair
    if not _single_action_allowed(first) or not _single_action_allowed(second):
        return False
    families = {str(first["family"]), str(second["family"])}
    if families == {"move"}:
        return False
    if str(first["action_api"]) == str(second["action_api"]):
        return False
    return True


def _chain_action_allowed(seq: Sequence[Dict[str, object]]) -> bool:
    if len(seq) != 3:
        return False
    if any(not _single_action_allowed(action) for action in seq):
        return False
    families = [str(action["family"]) for action in seq]
    if families.count("move") > 1:
        return False
    if families.count("look") > 1:
        return False
    return len(set(str(action["action_api"]) for action in seq)) >= 2
