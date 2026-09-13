#!/usr/bin/env python3
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SRC = Path("/path/to/workspace/high_quality_new_interaction_filtered_standardcm_20260616_work/qa_data.filtered.standardcm.jsonl")
OUT = Path("/path/to/workspace/high_quality_new_interaction_final_subset_20260616_work")

LEVELS = [5, 10, 15, 20]
LABELS = list("ABCD")
DIRECTIONS = ["forward", "backward", "left", "right", "up", "down"]
TOL_CM = 2.0
MIN_ACTION_CANDIDATE_GAP_CM = 3.0

PURE_TOTAL = 4000
ACTION_TOTAL = 2000
LONG_TOTAL = 2000
SORTING_TOTAL = 2000


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def even_quotas(keys: list[Any], total: int) -> dict[Any, int]:
    base = total // len(keys)
    rem = total % len(keys)
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def capped_balanced_quotas(keys: list[Any], total: int, availability: Counter) -> dict[Any, int]:
    quota = {k: 0 for k in keys}
    remaining = total
    active = set(keys)
    while active and remaining > 0:
        share = max(1, remaining // len(active))
        changed = False
        for k in list(active):
            room = max(0, int(availability.get(k, 0)) - quota[k])
            add = min(room, share, remaining)
            quota[k] += add
            remaining -= add
            changed = changed or add > 0
            if quota[k] >= int(availability.get(k, 0)):
                active.remove(k)
            if remaining <= 0:
                break
        if not changed:
            break
    while remaining > 0:
        candidates = [k for k in keys if quota[k] < int(availability.get(k, 0))]
        if not candidates:
            raise RuntimeError(f"not enough availability for quotas: total={total}, availability={dict(availability)}")
        k = min(candidates, key=lambda x: (quota[x], x))
        quota[k] += 1
        remaining -= 1
    return quota


def episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    gt = row.get("gt", {})
    if row.get("task_type", "").startswith("video_") or row.get("source") in {"roomtour3d", "SIMS-V"}:
        return (row.get("source"), gt.get("path_id") or gt.get("source_qa_id") or row.get("id"))
    return (row.get("source"), gt.get("shard"), gt.get("episode_index"), gt.get("episode_id"))


def shard_key(row: dict[str, Any]) -> tuple[Any, ...]:
    gt = row.get("gt", {})
    return (row.get("source"), gt.get("shard") or gt.get("video_id") or gt.get("path_id"))


def valid_standard_cm(v: float) -> int | None:
    vals = [cm for cm in LEVELS if abs(float(v) - cm) <= TOL_CM]
    if not vals:
        return None
    return min(vals, key=lambda cm: (abs(float(v) - cm), cm))


def mcq_text(opts: list[dict[str, Any]]) -> str:
    return "\n".join(f"{o['label']}. {o.get('text', 'Candidate ' + o['label'])}" for o in opts)


def make_lf_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = row["input"].get("frame_labels") or [str(i + 1) for i in range(len(row["input"]["frame_paths"]))]
    user = "\n".join(f"<image> {lab}" for lab in labels)
    user += "\n" + row["question"] + "\n" + mcq_text(row["options"]) + "\nAnswer with the option letter only."
    return {
        "id": row["id"],
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": row["answer"]}],
        "images": row["input"]["frame_paths"],
        "metadata": {
            "task_type": row["task_type"],
            "subcat": row["subcat"],
            "source": row["source"],
            "gt": row["gt"],
            "answer_text": row["answer_text"],
            "quality_filter": row.get("quality_filter", {}),
            "standard_cm_revision": row.get("standard_cm_revision", {}),
            "final_subset_revision": row.get("final_subset_revision", {}),
        },
    }


def motion_score(direction: str, gt: dict[str, Any]) -> float:
    off = float(gt.get("off_axis_cm", 99.0))
    zoff = float(gt.get("vertical_off_cm", 99.0))
    grip = abs(float(gt.get("gripper_delta", 99.0)))
    dist = float(gt.get("distance_cm", gt.get("target_distance_cm", 0.0)))
    cm = valid_standard_cm(dist)
    err = 10.0 if cm is None else abs(dist - cm)
    return off + (0.0 if direction in {"up", "down"} else zoff) + 45.0 * grip + 0.05 * err


def fixed_cm_options(correct_cm: int, answer_counts: Counter) -> tuple[list[dict[str, str]], str]:
    label = min(LABELS, key=lambda lab: (answer_counts[lab], lab))
    answer_counts[label] += 1
    rest = iter([cm for cm in LEVELS if cm != correct_cm])
    opts = []
    for lab in LABELS:
        val = correct_cm if lab == label else next(rest)
        opts.append({"label": lab, "text": f"about {val} cm"})
    return opts, label


def finalize_pure_row(row: dict[str, Any], answer_counts: Counter) -> dict[str, Any]:
    rr = json.loads(json.dumps(row, ensure_ascii=False))
    cm = int(rr["gt"]["standardized_answer_cm"])
    opts, ans = fixed_cm_options(cm, answer_counts)
    rr["options"] = opts
    rr["answer"] = ans
    rr["answer_text"] = f"about {cm} cm"
    rr["question"] = (
        f"Given that the robot end effector moved {rr['subcat']} between Image A and Image B, "
        "approximately how far did it move in that direction?"
    )
    rr["final_subset_revision"] = {
        "revision": "final_interaction_subset_20260616",
        "task": "pure_two_frame_motion_magnitude",
        "selection_note": "4000-row final pure pool; selected from original pure rows plus clean action-to-image candidate pairs; distance answer must be within ±2cm of 5/10/15/20cm",
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return rr


def pure_from_action_candidates(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for row in action_rows:
        direction = row.get("subcat") or row.get("gt", {}).get("direction")
        paths = row.get("input", {}).get("frame_paths", [])
        labels = row.get("input", {}).get("frame_labels", [])
        if not paths or not labels:
            continue
        label_to_path = dict(zip(labels, paths))
        start_path = paths[0]
        for metric in row.get("gt", {}).get("candidate_metrics", []):
            label = str(metric.get("label"))
            candidate_path = label_to_path.get(f"Candidate {label}") or label_to_path.get(label)
            if not candidate_path:
                continue
            dist = float(metric.get("distance_cm", 1e9))
            cm = valid_standard_cm(dist)
            if cm is None:
                continue
            off = float(metric.get("off_axis_cm", 99.0))
            zoff = float(metric.get("vertical_off_cm", 99.0))
            grip = abs(float(metric.get("gripper_delta", 99.0)))
            if grip > 0.05 or off > 1.5 or (direction not in {"up", "down"} and zoff > 1.5):
                continue
            gt = row.get("gt", {})
            new_id = f"robot_pure_from_action_candidate_{row['id']}_{label}_{cm}"
            derived.append({
                "id": new_id,
                "task_type": "pure_two_frame_motion_magnitude",
                "subcat": direction,
                "source": row.get("source", "bridgedata_v2"),
                "question": "",
                "answer": "",
                "answer_text": "",
                "options": [],
                "input": {
                    "frame_paths": [start_path, candidate_path],
                    "frame_labels": ["Image A", "Image B"],
                },
                "gt": {
                    "direction": direction,
                    "distance_cm": dist,
                    "off_axis_cm": off,
                    "vertical_off_cm": zoff,
                    "gripper_delta": float(metric.get("gripper_delta", 0.0)),
                    "shard": gt.get("shard"),
                    "episode_index": gt.get("episode_index"),
                    "episode_id": gt.get("episode_id"),
                    "start_frame": gt.get("start_frame"),
                    "candidate_label": label,
                    "candidate_source_bin_index": metric.get("source_bin_index"),
                    "derived_from_action_to_image_id": row["id"],
                    "standardized_answer_cm": cm,
                    "raw_distance_cm": dist,
                    "standard_distance_tolerance_cm": TOL_CM,
                    "standard_distance_error_cm": abs(dist - cm),
                },
                "quality_filter": {
                    "derived_from_action_to_image": True,
                    "off_axis_cm": off,
                    "vertical_off_cm": zoff,
                    "gripper_abs_delta": grip,
                },
                "standard_cm_revision": {
                    "revision": "standardcm_20260616_derived_candidate",
                    "visible_distance_options_cm": LEVELS,
                    "distance_tolerance_cm": TOL_CM,
                },
            })
    return derived


def select_pure(rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    seen = set()
    for row in rows + pure_from_action_candidates(action_rows):
        gt = row.get("gt", {})
        cm = gt.get("standardized_answer_cm")
        if cm not in LEVELS:
            continue
        dist = float(gt.get("raw_distance_cm", gt.get("distance_cm", 1e9)))
        if abs(dist - int(cm)) > TOL_CM + 1e-9:
            continue
        direction = row.get("subcat") or gt.get("direction")
        if direction not in DIRECTIONS:
            continue
        frame_key = tuple(row.get("input", {}).get("frame_paths", []))
        key = (frame_key, direction, int(cm))
        if key in seen:
            continue
        seen.add(key)
        row = json.loads(json.dumps(row, ensure_ascii=False))
        row["_cm"] = int(cm)
        row["_direction"] = direction
        row["_score"] = motion_score(direction, gt)
        row["_episode"] = episode_key(row)
        row["_shard"] = shard_key(row)
        row["_derived"] = "derived_from_action_to_image_id" in gt
        candidates.append(row)

    per_cm_total = even_quotas(LEVELS, PURE_TOTAL)
    availability_by_cell = Counter((row["_cm"], row["_direction"]) for row in candidates)
    cell_quota: dict[tuple[int, str], int] = {}
    for cm in LEVELS:
        availability = Counter({d: availability_by_cell[(cm, d)] for d in DIRECTIONS})
        quotas = capped_balanced_quotas(DIRECTIONS, per_cm_total[cm], availability)
        for d, q in quotas.items():
            cell_quota[(cm, d)] = q

    selected: list[dict[str, Any]] = []
    selected_ids = set()
    ep_counts = Counter()
    shard_counts = Counter()

    for cm in [20, 15, 10, 5]:
        for direction in DIRECTIONS:
            quota = cell_quota[(cm, direction)]
            while quota > 0:
                feasible = [
                    row for row in candidates
                    if row["id"] not in selected_ids
                    and row["_cm"] == cm
                    and row["_direction"] == direction
                ]
                if not feasible:
                    raise RuntimeError(f"not enough pure candidates for {cm}cm/{direction}")
                row = min(
                    feasible,
                    key=lambda r: (
                        ep_counts[r["_episode"]],
                        shard_counts[r["_shard"]],
                        0 if r["_derived"] else 1 if r["_cm"] in {15, 20} else 0,
                        r["_score"],
                        r["id"],
                    ),
                )
                selected.append(row)
                selected_ids.add(row["id"])
                quota -= 1
                ep_counts[row["_episode"]] += 1
                shard_counts[row["_shard"]] += 1

    if len(selected) != PURE_TOTAL:
        raise RuntimeError(f"pure selection count mismatch: {len(selected)} != {PURE_TOTAL}")
    selected_cm = Counter(row["_cm"] for row in selected)
    if any(selected_cm[cm] != per_cm_total[cm] for cm in LEVELS):
        raise RuntimeError(f"pure cm quota mismatch: selected={dict(selected_cm)}, target={dict(per_cm_total)}")

    answer_counts = Counter()
    out = []
    for row in selected:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        out.append(finalize_pure_row(clean, answer_counts))
    out.sort(key=lambda r: (r["task_type"], r["subcat"], r["gt"]["standardized_answer_cm"], r["id"]))
    report = {
        "candidate_count": len(candidates),
        "selected_count": len(out),
        "selected_from_action_candidates": sum(1 for r in out if "derived_from_action_to_image_id" in r.get("gt", {})),
        "selected_from_original_pure": sum(1 for r in out if "derived_from_action_to_image_id" not in r.get("gt", {})),
        "cm_quota_target": dict(per_cm_total),
        "cell_quota_target": {f"{cm}:{d}": q for (cm, d), q in sorted(cell_quota.items())},
    }
    return out, report


def action_score(row: dict[str, Any]) -> float:
    gt = row.get("gt", {})
    metric = gt.get("selected_candidate_metric", {})
    direction = row.get("subcat") or gt.get("direction")
    off = float(metric.get("off_axis_cm", 99.0))
    zoff = float(metric.get("vertical_off_cm", 99.0))
    grip = abs(float(metric.get("gripper_delta", 99.0)))
    err = abs(float(metric.get("distance_cm", 0.0)) - float(gt.get("standardized_target_cm", 0.0)))
    return off + (0.0 if direction in {"up", "down"} else zoff) + 45.0 * grip + 0.05 * err


def action_candidate_spacing_ok(row: dict[str, Any]) -> bool:
    metrics = row.get("gt", {}).get("candidate_metrics", [])
    if len(metrics) != 4:
        return False
    bins = []
    dists = []
    for metric in metrics:
        vals = metric.get("valid_standard_distance_cm") or []
        if len(vals) != 1:
            return False
        bins.append(int(vals[0]))
        dists.append(float(metric.get("distance_cm", 1e9)))
    if sorted(bins) != LEVELS:
        return False
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            if abs(dists[i] - dists[j]) < MIN_ACTION_CANDIDATE_GAP_CM:
                return False
    return True


def select_action(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_count = len(rows)
    rows = [row for row in rows if action_candidate_spacing_ok(row)]
    if len(rows) < ACTION_TOTAL:
        raise RuntimeError(f"not enough well-spaced action-to-image rows: {len(rows)} < {ACTION_TOTAL}")
    avail_dir = Counter(row["subcat"] for row in rows)
    avail_cm = Counter(int(row["gt"]["standardized_target_cm"]) for row in rows)
    avail_label = Counter(row["answer"] for row in rows)
    cm_quota = capped_balanced_quotas(LEVELS, ACTION_TOTAL, avail_cm)
    label_target = even_quotas(LABELS, ACTION_TOTAL)
    direction_target = capped_balanced_quotas(DIRECTIONS, ACTION_TOTAL, avail_dir)
    label_counts = Counter()
    direction_counts = Counter()
    selected = []
    selected_ids = set()
    ep_counts = Counter()
    shard_counts = Counter()
    enriched = []
    for row in rows:
        rr = json.loads(json.dumps(row, ensure_ascii=False))
        rr["_cm"] = int(rr["gt"]["standardized_target_cm"])
        rr["_label"] = rr["answer"]
        rr["_direction"] = rr["subcat"]
        rr["_score"] = action_score(rr)
        rr["_episode"] = episode_key(rr)
        rr["_shard"] = shard_key(rr)
        enriched.append(rr)

    while len(selected) < ACTION_TOTAL:
        feasible = [
            row for row in enriched
            if row["id"] not in selected_ids
            and cm_quota[row["_cm"]] > 0
        ]
        if not feasible:
            raise RuntimeError(
                f"not enough action rows with quotas: selected={len(selected)}, cm={cm_quota}, label_counts={dict(label_counts)}, direction_counts={dict(direction_counts)}"
            )
        row = min(
            feasible,
            key=lambda r: (
                direction_counts[r["_direction"]] / max(1, direction_target[r["_direction"]]),
                label_counts[r["_label"]] / max(1, label_target[r["_label"]]),
                ep_counts[r["_episode"]],
                shard_counts[r["_shard"]],
                r["_score"],
                r["id"],
            ),
        )
        selected.append(row)
        selected_ids.add(row["id"])
        cm_quota[row["_cm"]] -= 1
        label_counts[row["_label"]] += 1
        direction_counts[row["_direction"]] += 1
        ep_counts[row["_episode"]] += 1
        shard_counts[row["_shard"]] += 1

    out = []
    for row in selected:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean["final_subset_revision"] = {
            "revision": "final_interaction_subset_20260616",
            "task": "action_to_image_choice",
            "selection_note": "2000-row final action-to-image subset; candidate images must cover 5/10/15/20cm with at least 3cm measured distance gap between every pair; target cm is balanced as closely as availability allows; answer labels are softly balanced; rare motion directions retained first",
            "min_candidate_distance_gap_cm": MIN_ACTION_CANDIDATE_GAP_CM,
            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        out.append(clean)
    out.sort(key=lambda r: (r["task_type"], r["subcat"], r["gt"]["standardized_target_cm"], r["id"]))
    return out, {
        "original_count": original_count,
        "well_spaced_count": len(rows),
        "min_candidate_distance_gap_cm": MIN_ACTION_CANDIDATE_GAP_CM,
        "available_by_direction": dict(avail_dir),
        "available_by_target_cm": dict(sorted(avail_cm.items())),
        "available_by_answer_label": dict(sorted(avail_label.items())),
        "target_cm_quota": dict(sorted(Counter(r["gt"]["standardized_target_cm"] for r in out).items())),
        "answer_label_quota": dict(sorted(Counter(r["answer"] for r in out).items())),
        "selected_direction_quota_target": {k: v for k, v in sorted((Counter(r["subcat"] for r in out)).items())},
        "unfilled_cm_quota": dict(cm_quota),
        "answer_label_target_soft": dict(label_target),
        "direction_target_soft": dict(direction_target),
    }


def long_cluster(row: dict[str, Any]) -> tuple[Any, ...]:
    gt = row.get("gt", {})
    s1 = gt.get("segment_1", {})
    s2 = gt.get("segment_2", {})
    return (
        s1.get("direction"), s1.get("rounded_cm"),
        s2.get("direction"), s2.get("rounded_cm"),
    )


def long_score(row: dict[str, Any]) -> float:
    gt = row.get("gt", {})
    s1 = gt.get("segment_1", {})
    s2 = gt.get("segment_2", {})
    err = abs(float(s1.get("distance_cm", 0.0)) - float(s1.get("rounded_cm", 0.0)))
    err += abs(float(s2.get("distance_cm", 0.0)) - float(s2.get("rounded_cm", 0.0)))
    # Keep substantial two-stage motions. Very tiny segments tend to be visually weak.
    tiny_penalty = 0.0
    if float(s1.get("rounded_cm", 0.0)) < 10:
        tiny_penalty += 1.5
    if float(s2.get("rounded_cm", 0.0)) < 10:
        tiny_penalty += 1.5
    return err + tiny_penalty


def select_long(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_quota = even_quotas(LABELS, LONG_TOTAL)
    selected = []
    selected_ids = set()
    ep_counts = Counter()
    shard_counts = Counter()
    cluster_counts = Counter()
    enriched = []
    for row in rows:
        rr = json.loads(json.dumps(row, ensure_ascii=False))
        rr["_label"] = rr["answer"]
        rr["_episode"] = episode_key(rr)
        rr["_shard"] = shard_key(rr)
        rr["_cluster"] = long_cluster(rr)
        rr["_score"] = long_score(rr)
        enriched.append(rr)

    while len(selected) < LONG_TOTAL:
        feasible = [row for row in enriched if row["id"] not in selected_ids and label_quota[row["_label"]] > 0]
        if not feasible:
            raise RuntimeError(f"not enough long-horizon rows with label quotas: {label_quota}")
        row = min(
            feasible,
            key=lambda r: (
                cluster_counts[r["_cluster"]],
                ep_counts[r["_episode"]],
                shard_counts[r["_shard"]],
                r["_score"],
                r["id"],
            ),
        )
        selected.append(row)
        selected_ids.add(row["id"])
        label_quota[row["_label"]] -= 1
        ep_counts[row["_episode"]] += 1
        shard_counts[row["_shard"]] += 1
        cluster_counts[row["_cluster"]] += 1

    out = []
    for row in selected:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean["final_subset_revision"] = {
            "revision": "final_interaction_subset_20260616",
            "task": "long_horizon_manipulation_program",
            "selection_note": "2000-row final long-horizon subset; answer labels balanced exactly; segment direction/distance clusters and episodes diversified",
            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        out.append(clean)
    out.sort(key=lambda r: (r["task_type"], r["subcat"], r["id"]))
    return out, {
        "selected_cluster_count": len(cluster_counts),
        "unfilled_label_quota": dict(label_quota),
        "top_clusters": {str(k): v for k, v in cluster_counts.most_common(20)},
    }


def sorting_cluster(row: dict[str, Any]) -> tuple[Any, ...]:
    gt = row.get("gt", {})
    return (
        gt.get("instruction"),
        tuple(gt.get("ordered_frame_indices", [])),
        row.get("answer_text"),
    )


def sorting_score(row: dict[str, Any]) -> float:
    gt = row.get("gt", {})
    return -float(gt.get("gripper_range", 0.0))


def select_sorting(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) <= SORTING_TOTAL:
        return rows, {"selected_count": len(rows), "available_count": len(rows), "note": "available rows <= target; kept all"}

    label_quota = even_quotas(LABELS, SORTING_TOTAL)
    selected = []
    selected_ids = set()
    ep_counts = Counter()
    shard_counts = Counter()
    instruction_counts = Counter()
    cluster_counts = Counter()
    enriched = []
    for row in rows:
        rr = json.loads(json.dumps(row, ensure_ascii=False))
        rr["_label"] = rr["answer"]
        rr["_episode"] = episode_key(rr)
        rr["_shard"] = shard_key(rr)
        rr["_instruction"] = rr.get("gt", {}).get("instruction")
        rr["_cluster"] = sorting_cluster(rr)
        rr["_score"] = sorting_score(rr)
        enriched.append(rr)

    while len(selected) < SORTING_TOTAL:
        feasible = [row for row in enriched if row["id"] not in selected_ids and label_quota[row["_label"]] > 0]
        if not feasible:
            raise RuntimeError(f"not enough sorting rows with label quotas: {label_quota}")
        row = min(
            feasible,
            key=lambda r: (
                cluster_counts[r["_cluster"]],
                instruction_counts[r["_instruction"]],
                ep_counts[r["_episode"]],
                shard_counts[r["_shard"]],
                r["_score"],
                r["id"],
            ),
        )
        selected.append(row)
        selected_ids.add(row["id"])
        label_quota[row["_label"]] -= 1
        ep_counts[row["_episode"]] += 1
        shard_counts[row["_shard"]] += 1
        instruction_counts[row["_instruction"]] += 1
        cluster_counts[row["_cluster"]] += 1

    out = []
    for row in selected:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        clean["final_subset_revision"] = {
            "revision": "final_interaction_subset_20260616",
            "task": "temporal_sequence_sorting",
            "selection_note": "2000-row final sorting subset; answer labels balanced exactly; instructions, episodes, and frame-order clusters diversified",
            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        out.append(clean)
    out.sort(key=lambda r: (r["task_type"], r["subcat"], r["id"]))
    return out, {
        "available_count": len(rows),
        "selected_count": len(out),
        "unique_instruction_count": len(instruction_counts),
        "selected_cluster_count": len(cluster_counts),
        "unfilled_label_quota": dict(label_quota),
        "answer_label_quota": dict(sorted(Counter(r["answer"] for r in out).items())),
        "top_instructions": {str(k): v for k, v in instruction_counts.most_common(20)},
    }


def summarize(rows: list[dict[str, Any]], reports: dict[str, Any]) -> dict[str, Any]:
    by_task = Counter(r["task_type"] for r in rows)
    by_source = Counter(r["source"] for r in rows)
    by_sub = Counter((r["task_type"], r["subcat"]) for r in rows)
    ans_by_task = defaultdict(Counter)
    for row in rows:
        ans_by_task[row["task_type"]][row["answer"]] += 1

    pure = [r for r in rows if r["task_type"] == "pure_two_frame_motion_magnitude"]
    action = [r for r in rows if r["task_type"] == "action_to_image_choice"]
    long = [r for r in rows if r["task_type"] == "long_horizon_manipulation_program"]
    reverse = [r for r in rows if r["task_type"] == "node_reverse_path_planning"]
    sorting = [r for r in rows if r["task_type"] == "temporal_sequence_sorting"]

    pure_bad = [
        r["id"] for r in pure
        if abs(float(r["gt"]["raw_distance_cm"]) - float(r["gt"]["standardized_answer_cm"])) > TOL_CM + 1e-9
    ]
    action_bad = [
        r["id"] for r in action
        if abs(float(r["gt"]["selected_candidate_metric"]["distance_cm"]) - float(r["gt"]["standardized_target_cm"])) > TOL_CM + 1e-9
    ]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_file": str(SRC),
        "output_dir": str(OUT),
        "total_samples": len(rows),
        "by_source": dict(by_source),
        "by_task_type": dict(by_task),
        "by_task_type_subcat": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_sub.items())},
        "answer_distribution_by_task_type": {k: dict(v) for k, v in sorted(ans_by_task.items())},
        "pure_distribution": {
            "count": len(pure),
            "cm": dict(sorted(Counter(r["gt"]["standardized_answer_cm"] for r in pure).items())),
            "direction": dict(sorted(Counter(r["subcat"] for r in pure).items())),
            "direction_cm": {f"{d}:{cm}": n for (d, cm), n in sorted(Counter((r["subcat"], r["gt"]["standardized_answer_cm"]) for r in pure).items())},
            "derived_from_action_candidates": sum(1 for r in pure if "derived_from_action_to_image_id" in r.get("gt", {})),
        },
        "action_to_image_distribution": {
            "count": len(action),
            "target_cm": dict(sorted(Counter(r["gt"]["standardized_target_cm"] for r in action).items())),
            "direction": dict(sorted(Counter(r["subcat"] for r in action).items())),
            "direction_cm": {f"{d}:{cm}": n for (d, cm), n in sorted(Counter((r["subcat"], r["gt"]["standardized_target_cm"]) for r in action).items())},
        },
        "long_horizon_distribution": {
            "count": len(long),
            "segment_cluster_count": len(Counter(long_cluster(r) for r in long)),
            "segment1_direction": dict(sorted(Counter(r["gt"]["segment_1"]["direction"] for r in long).items())),
            "segment2_direction": dict(sorted(Counter(r["gt"]["segment_2"]["direction"] for r in long).items())),
            "segment1_cm": dict(sorted(Counter(r["gt"]["segment_1"]["rounded_cm"] for r in long).items())),
            "segment2_cm": dict(sorted(Counter(r["gt"]["segment_2"]["rounded_cm"] for r in long).items())),
        },
        "kept_all_reverse_path_planning": len(reverse),
        "temporal_sequence_sorting_distribution": {
            "count": len(sorting),
            "answer": dict(sorted(Counter(r["answer"] for r in sorting).items())),
            "unique_instruction_count": len({r.get("gt", {}).get("instruction") for r in sorting}),
            "unique_episode_count": len({episode_key(r) for r in sorting}),
        },
        "quality_checks": {
            "pure_distance_tolerance_violations": len(pure_bad),
            "pure_distance_tolerance_bad_ids": pure_bad[:20],
            "action_distance_tolerance_violations": len(action_bad),
            "action_distance_tolerance_bad_ids": action_bad[:20],
            "duplicate_ids": sum(1 for v in Counter(r["id"] for r in rows).values() if v > 1),
        },
        "selection_reports": reports,
        "notes": [
            "Pure two-frame is now 4000 rows and uses clean action-to-image candidate pairs to supplement larger 15/20cm motions.",
            "Action-to-image is now 2000 rows; 5/10/15/20cm targets are balanced as closely as strict candidate-spacing availability allows, and A/B/C/D labels are softly balanced.",
            "Long-horizon manipulation is now 2000 rows with answer labels balanced and segment clusters diversified.",
            "All node_reverse_path_planning rows are kept. Temporal sequence sorting is downsampled to 2000 with label, instruction, and episode diversity.",
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SRC)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)

    pure_rows, pure_report = select_pure(by_task["pure_two_frame_motion_magnitude"], by_task["action_to_image_choice"])
    action_rows, action_report = select_action(by_task["action_to_image_choice"])
    long_rows, long_report = select_long(by_task["long_horizon_manipulation_program"])
    reverse_rows = by_task["node_reverse_path_planning"]
    sorting_rows, sorting_report = select_sorting(by_task["temporal_sequence_sorting"])

    final_rows = pure_rows + action_rows + long_rows + reverse_rows + sorting_rows
    final_rows.sort(key=lambda r: (r["task_type"], r["subcat"], r["id"]))

    write_jsonl(OUT / "qa_data.final_subset.jsonl", final_rows)
    write_jsonl(OUT / "llamafactory_data.final_subset.jsonl", [make_lf_row(r) for r in final_rows])
    summary = summarize(final_rows, {
        "pure": pure_report,
        "action_to_image": action_report,
        "long_horizon": long_report,
        "temporal_sequence_sorting": sorting_report,
    })
    (OUT / "final_subset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
