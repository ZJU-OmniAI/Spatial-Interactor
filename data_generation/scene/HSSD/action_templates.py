from __future__ import annotations

from itertools import product
import random
from typing import Dict, List


MOVE_ACTIONS = [
    ("MoveAhead", "向前移动"),
    ("MoveBack", "向后移动"),
    ("MoveLeft", "向左平移"),
    ("MoveRight", "向右平移"),
]

ROTATE_ACTIONS = [
    ("RotateLeft", "向左旋转"),
    ("RotateRight", "向右旋转"),
]

LOOK_ACTIONS = [
    ("LookUp", "向上抬头"),
    ("LookDown", "向下低头"),
]

MOVE_MAGNITUDES = [0.4, 0.6, 0.8, 1.0]
ROTATE_DEGREES = [30, 45, 60, 75, 90]
LOOK_DEGREES = [30, 45, 60]


def build_single_action_templates() -> List[Dict[str, object]]:
    templates: List[Dict[str, object]] = []
    for action_api, prefix in MOVE_ACTIONS:
        for mag in MOVE_MAGNITUDES:
            templates.append(
                {
                    "family": "move",
                    "action_api": action_api,
                    "action_params": {"moveMagnitude": mag},
                    "action_text": f"{prefix}{mag:.2f}米",
                }
            )
    for action_api, prefix in ROTATE_ACTIONS:
        for deg in ROTATE_DEGREES:
            templates.append(
                {
                    "family": "rotate",
                    "action_api": action_api,
                    "action_params": {"degrees": deg},
                    "action_text": f"{prefix}{int(deg)}度",
                }
            )
    for action_api, prefix in LOOK_ACTIONS:
        for deg in LOOK_DEGREES:
            templates.append(
                {
                    "family": "look",
                    "action_api": action_api,
                    "action_params": {"degrees": deg},
                    "action_text": f"{prefix}{int(deg)}度",
                }
            )
    return templates


def is_direct_inverse(first: Dict[str, object], second: Dict[str, object]) -> bool:
    pairs = {
        ("MoveAhead", "MoveBack"),
        ("MoveBack", "MoveAhead"),
        ("MoveLeft", "MoveRight"),
        ("MoveRight", "MoveLeft"),
        ("RotateLeft", "RotateRight"),
        ("RotateRight", "RotateLeft"),
        ("LookUp", "LookDown"),
        ("LookDown", "LookUp"),
    }
    if (str(first["action_api"]), str(second["action_api"])) not in pairs:
        return False
    first_params = first["action_params"]
    second_params = second["action_params"]
    return first_params == second_params


def valid_double_action_pairs() -> List[List[Dict[str, object]]]:
    singles = build_single_action_templates()
    pairs: List[List[Dict[str, object]]] = []
    for first, second in product(singles, repeat=2):
        if first == second:
            continue
        if is_direct_inverse(first, second):
            continue
        if first["family"] == "move" and second["family"] == "move":
            continue
        if first["family"] in {"rotate", "look"} and second["family"] in {"rotate", "look"}:
            first_deg = float(first["action_params"].get("degrees", 0.0))
            second_deg = float(second["action_params"].get("degrees", 0.0))
            if first_deg == 90.0 and second_deg == 90.0:
                continue
        pairs.append([first, second])
    return pairs


def valid_chain_action_sequences(min_steps: int = 3, max_steps: int = 4) -> List[List[Dict[str, object]]]:
    singles = build_single_action_templates()
    sequences: List[List[Dict[str, object]]] = []
    for step_count in range(min_steps, max_steps + 1):
        pool = singles[:]
        # deterministic but bounded first version: take every 5th template as seed anchors
        for start_idx in range(0, len(pool), 5):
            seq = [pool[start_idx]]
            cursor = start_idx + 1
            while len(seq) < step_count and cursor < len(pool):
                candidate = pool[cursor]
                cursor += 1
                if candidate == seq[-1]:
                    continue
                if is_direct_inverse(seq[-1], candidate):
                    continue
                prev_deg = float(seq[-1]["action_params"].get("degrees", 0.0))
                cur_deg = float(candidate["action_params"].get("degrees", 0.0))
                if (
                    seq[-1]["family"] in {"rotate", "look"}
                    and candidate["family"] in {"rotate", "look"}
                    and prev_deg == 90.0
                    and cur_deg == 90.0
                ):
                    continue
                seq.append(candidate)
            if len(seq) == step_count:
                sequences.append(seq)
    return sequences


def random_chain_action_sequences(
    seed: int,
    *,
    min_steps: int = 3,
    max_steps: int = 4,
    target_count: int = 64,
) -> List[List[Dict[str, object]]]:
    rng = random.Random(seed)
    singles = build_single_action_templates()
    sequences: List[List[Dict[str, object]]] = []
    seen = set()
    max_attempts = max(400, target_count * 40)
    attempts = 0

    while len(sequences) < target_count and attempts < max_attempts:
        attempts += 1
        step_count = rng.randint(min_steps, max_steps)
        seq: List[Dict[str, object]] = []
        cursor_attempts = 0
        while len(seq) < step_count and cursor_attempts < 200:
            cursor_attempts += 1
            candidate = singles[rng.randrange(len(singles))]
            if seq and candidate == seq[-1]:
                continue
            if seq and is_direct_inverse(seq[-1], candidate):
                continue
            if (
                seq
                and seq[-1]["family"] in {"rotate", "look"}
                and candidate["family"] in {"rotate", "look"}
                and float(seq[-1]["action_params"].get("degrees", 0.0)) == 90.0
                and float(candidate["action_params"].get("degrees", 0.0)) == 90.0
            ):
                continue
            if len(seq) >= 2 and candidate == seq[-2]:
                continue
            seq.append(
                {
                    "family": candidate["family"],
                    "action_api": candidate["action_api"],
                    "action_params": dict(candidate["action_params"]),
                    "action_text": candidate["action_text"],
                }
            )
        if len(seq) != step_count:
            continue
        key = tuple(
            (
                str(action["action_api"]),
                tuple(sorted(dict(action["action_params"]).items())),
            )
            for action in seq
        )
        if key in seen:
            continue
        seen.add(key)
        sequences.append(seq)
    return sequences
