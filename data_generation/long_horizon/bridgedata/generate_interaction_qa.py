#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tfrecord.reader import tfrecord_loader


BRIDGE_DIR = Path("/path/to/workspace/BridgeDataV2_full/1.0.0")
PILOT_TFR = Path("/path/to/workspace/robot_real_qa_pilot_20260614/bridgedata_v2/bridge_dataset-train.tfrecord-00000-of-01024")
VIDEO_FULL = Path("/path/to/workspace/videoqa_extra_recommended_additions_fullsimsv_three_slot_pathfix_20260608/videoqa_extra_recommended_additions_fullsimsv_three_slot_pathfix.jsonl")
OUT_DEFAULT = Path("/path/to/workspace/high_quality_new_interaction_qa_full_20260615_work")
DATASET2_TAR = Path("/path/to/workspace/high_quality_new_interaction_qa_full_20260615.tar")

DIRS = {
    "forward": (0, 1),
    "backward": (0, -1),
    "left": (1, 1),
    "right": (1, -1),
    "up": (2, 1),
    "down": (2, -1),
}


def clean_instruction(raw: Any) -> str:
    vals = []
    try:
        for x in raw:
            s = x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
            s = re.sub(r"\s+", " ", s).strip()
            if s:
                vals.append(s)
    except Exception:
        return ""
    return Counter(vals).most_common(1)[0][0] if vals else ""


def valid_instruction(s: str) -> bool:
    if not (6 <= len(s) <= 180):
        return False
    if s.count(" ") < 1 or sum(c.isalpha() for c in s) < 5:
        return False
    if re.fullmatch(r"[a-z]{10,}", s.lower()):
        return False
    return True


def list_tfrecords(bridge_dir: Path, use_pilot_if_missing: bool) -> list[Path]:
    files = sorted(bridge_dir.glob("bridge_dataset-train.tfrecord-*-of-01024"))
    if not files and use_pilot_if_missing and PILOT_TFR.exists():
        files = [PILOT_TFR]
    return files


def count_full_tfrecords(bridge_dir: Path) -> int:
    return len(list(bridge_dir.glob("bridge_dataset-train.tfrecord-*-of-01024")))


def wait_for_tfrecords(bridge_dir: Path, expected: int, poll: int) -> None:
    while True:
        n = count_full_tfrecords(bridge_dir)
        if n >= expected:
            return
        print(f"[wait] BridgeData shards ready: {n}/{expected}; sleeping {poll}s", flush=True)
        time.sleep(poll)


def ensure_dir(p: Path, made: set[Path]) -> None:
    if p in made:
        return
    p.mkdir(parents=True, exist_ok=True)
    made.add(p)


def write_robot_image(ep: dict[str, Any], frame: int, out: Path, made: set[Path], cache: dict[tuple[str, int, int], str]) -> str:
    key = (ep["shard"], ep["epidx"], int(frame))
    if key in cache:
        return cache[key]
    rel = f"images/robot/{ep['shard']}_ep{ep['epidx']:05d}_id{ep['episode_id']:05d}/frame_{frame:04d}.jpg"
    path = out / rel
    ensure_dir(path.parent, made)
    if not path.exists():
        path.write_bytes(bytes(ep["images"][frame]))
    cache[key] = rel
    return rel


def segment_path_len(pos: np.ndarray, i: int, j: int, axes=(0, 1, 2)) -> float:
    if j <= i:
        return 0.0
    diff = np.diff(pos[i:j + 1, list(axes)], axis=0)
    return float(np.sum(np.linalg.norm(diff, axis=1)) * 100.0)


def motion_metrics(ep: dict[str, Any], i: int, j: int, direction: str) -> dict[str, float] | None:
    axis, sgn = DIRS[direction]
    d = (ep["pos"][j] - ep["pos"][i]) * 100.0
    main = float(sgn * d[axis])
    if main <= 0:
        return None
    grip_delta = float(abs(ep["grip"][j] - ep["grip"][i]))
    if axis == 2:
        off = float(math.hypot(d[0], d[1]))
        z_off = 0.0
    else:
        other_h = 1 if axis == 0 else 0
        off = float(abs(d[other_h]))
        z_off = float(abs(d[2]))
    path = segment_path_len(ep["pos"], i, j)
    return {"main": main, "off": off, "z_off": z_off, "grip_delta": grip_delta, "path": path}


def is_pure(m: dict[str, float], direction: str) -> bool:
    if not (5.0 <= m["main"] <= 25.0):
        return False
    if m["grip_delta"] > 0.035:
        return False
    if direction in {"up", "down"}:
        if m["off"] > 2.5:
            return False
    elif m["off"] > 2.5 or m["z_off"] > 2.5:
        return False
    return m["path"] <= m["main"] * 1.35 + 2.0


def shuffled_options(correct: str, distractors: list[str], pos: int) -> tuple[list[dict[str, str]], str]:
    labels = "ABCD"
    ds = []
    for x in distractors:
        if x != correct and x not in ds:
            ds.append(x)
    while len(ds) < 3:
        ds.append(f"none of the above {len(ds)}")
    opts = []
    di = 0
    for k, lab in enumerate(labels):
        if k == pos % 4:
            opts.append({"label": lab, "text": correct})
        else:
            opts.append({"label": lab, "text": ds[di]})
            di += 1
    return opts, labels[pos % 4]


def numeric_options_cm(v: float, pos: int) -> tuple[list[dict[str, str]], str]:
    val = int(round(v))
    d = [max(1, val - 8), max(1, val - 4), val + 4, val + 8, max(1, val - 6), val + 6]
    return shuffled_options(f"about {val} cm", [f"about {x} cm" for x in d], pos)


def mcq_text(opts: list[dict[str, Any]]) -> str:
    return "\n".join(f"{o['label']}. {o.get('text', 'Candidate ' + o['label'])}" for o in opts)


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.qa = (out / "qa_data.jsonl").open("w")
        self.lf = (out / "llamafactory_data.jsonl").open("w")
        self.counts = Counter()
        self.subcounts = Counter()
        self.answers = defaultdict(Counter)
        self.sources = Counter()

    def add(self, row: dict[str, Any]) -> None:
        self.qa.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.counts[row["task_type"]] += 1
        self.subcounts[(row["task_type"], row["subcat"])] += 1
        self.answers[row["task_type"]][row["answer"]] += 1
        self.sources[row["source"]] += 1
        labels = row["input"].get("frame_labels") or [str(i + 1) for i in range(len(row["input"]["frame_paths"]))]
        user = "\n".join(f"<image> {lab}" for lab in labels)
        user += "\n" + row["question"] + "\n" + mcq_text(row["options"]) + "\nAnswer with the option letter only."
        lf_row = {
            "id": row["id"],
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": row["answer"]}],
            "images": row["input"]["frame_paths"],
            "metadata": {"task_type": row["task_type"], "subcat": row["subcat"], "source": row["source"], "gt": row["gt"], "answer_text": row["answer_text"]},
        }
        self.lf.write(json.dumps(lf_row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.qa.close()
        self.lf.close()


def task_key(task_type: str, subcat: str) -> tuple[str, str]:
    return task_type, subcat


def cap_ok(writer: Writer, task_type: str, subcat: str, cap: int) -> bool:
    return writer.subcounts[task_key(task_type, subcat)] < cap


def best_pure_segments(ep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    n = len(ep["state"])
    for direction in DIRS:
        best = None
        for i in range(n - 1):
            for j in range(i + 1, min(n, i + 22)):
                m = motion_metrics(ep, i, j, direction)
                if not m or not is_pure(m, direction):
                    continue
                score = m["off"] + m["z_off"] + abs(m["path"] - m["main"]) * 0.35 + m["grip_delta"] * 20
                if best is None or score < best["score"]:
                    best = {"i": i, "j": j, "m": m, "score": score}
        if best:
            out[direction] = best
    return out


def action_to_image_candidate(ep: dict[str, Any], direction: str) -> dict[str, Any] | None:
    bins = [5.0, 10.0, 15.0, 20.0]
    n = len(ep["state"])
    best = None
    for i in range(n - 2):
        cand = []
        for j in range(i + 1, n):
            m = motion_metrics(ep, i, j, direction)
            if m and is_pure(m, direction):
                cand.append((j, m))
        if len(cand) < 4:
            continue
        chosen = []
        used = set()
        err = 0.0
        for b in bins:
            pool = [(j, m) for j, m in cand if j not in used]
            if not pool:
                break
            j, m = min(pool, key=lambda x: abs(x[1]["main"] - b))
            if abs(m["main"] - b) > 4.0:
                break
            used.add(j)
            chosen.append((j, m))
            err += abs(m["main"] - b)
        if len(chosen) == 4:
            purity = sum(m["off"] + m["z_off"] + m["grip_delta"] * 20 for _, m in chosen)
            if best is None or (err, purity) < (best["err"], best["purity"]):
                best = {"i": i, "chosen": chosen, "err": err, "purity": purity}
    return best


def sorting_candidate(ep: dict[str, Any]) -> dict[str, Any] | None:
    instr = ep["instruction"]
    if not valid_instruction(instr):
        return None
    grip = ep["grip"]
    n = len(grip)
    if n < 18:
        return None
    gr = float(grip.max() - grip.min())
    if gr < 0.35:
        return None
    low = float(grip.min() + 0.25 * gr)
    high = float(grip.min() + 0.65 * gr)
    lows = [k for k in range(3, n - 5) if grip[k] <= low]
    if not lows:
        return None
    grasp = lows[0]
    rels = [k for k in range(grasp + 3, n - 2) if grip[k] >= high]
    if not rels:
        return None
    release = rels[0]
    end = n - 1
    if release >= end - 2 or segment_path_len(ep["pos"], 0, end, axes=(0, 1)) < 20:
        return None
    frames = [0, grasp, release, end]
    return {"frames": frames, "gripper_range": gr}


def dir_name(dx: float, dy: float) -> str:
    m = max(abs(dx), abs(dy), 1e-9)
    parts = []
    if abs(dx) >= 0.45 * m:
        parts.append("forward" if dx > 0 else "backward")
    if abs(dy) >= 0.45 * m:
        parts.append("left" if dy > 0 else "right")
    return "-".join(parts) if parts else "slightly"


def round5(x: float) -> int:
    return int(max(5, round(x / 5.0) * 5))


def sample_9(n: int) -> list[int]:
    if n <= 9:
        return list(range(n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, 9)))


def long_program_candidate(ep: dict[str, Any]) -> dict[str, Any] | None:
    instr = ep["instruction"]
    if not valid_instruction(instr):
        return None
    pos = ep["pos"][:, :2] * 100.0
    n = len(pos)
    if n < 18:
        return None
    total = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if total < 25:
        return None
    start, end = pos[0], pos[-1]
    best = None
    for k in range(4, n - 4):
        d1 = float(np.linalg.norm(pos[k] - start))
        d2 = float(np.linalg.norm(end - pos[k]))
        chord = float(np.linalg.norm(end - start))
        if d1 < 8 or d2 < 8:
            continue
        v1 = pos[k] - start
        v2 = end - pos[k]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)))))
        extra = d1 + d2 - chord
        if ang < 30 or extra < 4:
            continue
        score = extra + min(d1, d2) * 0.15
        if best is None or score > best[0]:
            best = (score, k, d1, d2, ang, v1, v2)
    if not best:
        return None
    _, k, d1, d2, ang, v1, v2 = best
    return {"k": k, "d1": d1, "d2": d2, "ang": ang, "dir1": dir_name(float(v1[0]), float(v1[1])), "dir2": dir_name(float(v2[0]), float(v2[1])), "frames": sample_9(n)}


def stream_episodes(tfrecords: list[Path], seed: int):
    files = list(tfrecords)
    random.Random(seed).shuffle(files)
    desc = {
        "steps/observation/image_0": "byte",
        "steps/observation/state": "float",
        "steps/language_instruction": "byte",
        "episode_metadata/episode_id": "int",
    }
    for shard_path in files:
        shard = shard_path.name.replace("bridge_dataset-train.tfrecord-", "").replace("-of-01024", "")
        for epidx, ex in enumerate(tfrecord_loader(str(shard_path), None, description=desc)):
            state = np.array(ex["steps/observation/state"], dtype=float).reshape(-1, 7)
            yield {
                "shard": shard,
                "epidx": epidx,
                "episode_id": int(ex["episode_metadata/episode_id"][0]) if "episode_metadata/episode_id" in ex else epidx,
                "state": state,
                "pos": state[:, :3],
                "grip": state[:, 6],
                "images": ex["steps/observation/image_0"],
                "instruction": clean_instruction(ex.get("steps/language_instruction", [])),
            }


def generate_robot(writer: Writer, out: Path, tfrecords: list[Path], cap: int, seed: int, max_episodes: int | None) -> dict[str, Any]:
    made = set()
    img_cache = {}
    sid = Counter()
    orders = [[2, 0, 3, 1], [1, 3, 0, 2], [3, 1, 2, 0], [0, 2, 1, 3]]
    episodes = 0
    for ep in stream_episodes(tfrecords, seed):
        episodes += 1
        if max_episodes and episodes > max_episodes:
            break
        # Pure two-frame: at most one per direction per episode.
        for direction, best in best_pure_segments(ep).items():
            task = "pure_two_frame_motion_magnitude"
            if not cap_ok(writer, task, direction, cap):
                continue
            sid[task] += 1
            a = write_robot_image(ep, best["i"], out, made, img_cache)
            b = write_robot_image(ep, best["j"], out, made, img_cache)
            opts, correct = numeric_options_cm(best["m"]["main"], writer.subcounts[(task, direction)])
            q = f"Given that the robot end effector moved {direction} between Image A and Image B, approximately how far did it move in that direction?"
            writer.add({"id": f"{task}_{sid[task]:07d}", "source": "bridgedata_v2", "task_family": "robot_manipulation", "task_type": task, "subcat": direction, "question": q, "options": opts, "answer": correct, "answer_text": next(o["text"] for o in opts if o["label"] == correct), "input": {"frame_paths": [a, b], "frame_labels": ["Image A", "Image B"]}, "gt": {"direction": direction, "distance_cm": round(best["m"]["main"], 2), "off_axis_cm": round(best["m"]["off"], 2), "vertical_off_cm": round(best["m"]["z_off"], 2), "gripper_delta": round(best["m"]["grip_delta"], 4), "shard": ep["shard"], "episode_index": ep["epidx"], "episode_id": ep["episode_id"], "start_frame": best["i"], "end_frame": best["j"]}})

        # Action-to-image: at most one per direction per episode.
        for direction in DIRS:
            task = "action_to_image_choice"
            if not cap_ok(writer, task, direction, cap):
                continue
            cand = action_to_image_candidate(ep, direction)
            if not cand:
                continue
            sid[task] += 1
            start = write_robot_image(ep, cand["i"], out, made, img_cache)
            cands = [write_robot_image(ep, j, out, made, img_cache) for j, _ in cand["chosen"]]
            order = orders[sid[task] % len(orders)]
            correct_orig = 2
            correct = "ABCD"[order.index(correct_orig)]
            image_opts = [{"label": "ABCD"[k], "text": f"Candidate {'ABCD'[k]}", "image": cands[orig]} for k, orig in enumerate(order)]
            dist = cand["chosen"][correct_orig][1]["main"]
            q = f"Starting from Image A, the robot end effector should move {direction} by about {round(dist):.0f} cm. Which candidate image best matches the resulting frame?"
            writer.add({"id": f"{task}_{sid[task]:07d}", "source": "bridgedata_v2", "task_family": "robot_manipulation", "task_type": task, "subcat": direction, "question": q, "options": image_opts, "answer": correct, "answer_text": f"Candidate {correct}", "input": {"frame_paths": [start] + [o["image"] for o in image_opts], "frame_labels": ["Image A", "Candidate A", "Candidate B", "Candidate C", "Candidate D"]}, "gt": {"direction": direction, "target_distance_cm": round(dist, 2), "shard": ep["shard"], "episode_index": ep["epidx"], "episode_id": ep["episode_id"], "start_frame": cand["i"], "candidate_metrics": [{"label": "ABCD"[k], "source_bin_index": orig, "distance_cm": round(cand["chosen"][orig][1]["main"], 2), "off_axis_cm": round(cand["chosen"][orig][1]["off"], 2), "vertical_off_cm": round(cand["chosen"][orig][1]["z_off"], 2), "gripper_delta": round(cand["chosen"][orig][1]["grip_delta"], 4)} for k, orig in enumerate(order)]}})

        # Sorting.
        task = "temporal_sequence_sorting"
        sub = "start_grasp_place_return"
        if cap_ok(writer, task, sub, cap):
            cand = sorting_candidate(ep)
            if cand:
                sid[task] += 1
                frames = cand["frames"]
                rel = [write_robot_image(ep, f, out, made, img_cache) for f in frames]
                order = orders[sid[task] % len(orders)]
                shuffled = [rel[i] for i in order]
                labels = "ABCD"
                correct_seq = " -> ".join(labels[order.index(i)] for i in range(4))
                opts, correct = shuffled_options(correct_seq, ["A -> B -> C -> D", "B -> A -> C -> D", "A -> D -> C -> B", "C -> A -> D -> B"], writer.subcounts[(task, sub)])
                q = f"Instruction: {ep['instruction']}. The four images are unordered key frames from one robot manipulation episode. Sort images A, B, C, and D from earliest to latest."
                writer.add({"id": f"{task}_{sid[task]:07d}", "source": "bridgedata_v2", "task_family": "robot_manipulation", "task_type": task, "subcat": sub, "question": q, "options": opts, "answer": correct, "answer_text": correct_seq, "input": {"frame_paths": shuffled, "frame_labels": ["A", "B", "C", "D"]}, "gt": {"ordered_frame_indices": frames, "shuffled_order_source_indices": order, "instruction": ep["instruction"], "gripper_range": round(cand["gripper_range"], 4), "shard": ep["shard"], "episode_index": ep["epidx"], "episode_id": ep["episode_id"]}})

        # Long-horizon.
        task = "long_horizon_manipulation_program"
        sub = "two_stage_horizontal_program"
        if cap_ok(writer, task, sub, cap):
            cand = long_program_candidate(ep)
            if cand:
                sid[task] += 1
                rel = [write_robot_image(ep, f, out, made, img_cache) for f in cand["frames"]]
                d1r, d2r = round5(cand["d1"]), round5(cand["d2"])
                correct_text = f"first moves {cand['dir1']} about {d1r} cm, then moves {cand['dir2']} about {d2r} cm"
                opts, correct = shuffled_options(correct_text, [f"first moves {cand['dir1']} about {d2r} cm, then moves {cand['dir2']} about {d1r} cm", f"first moves {cand['dir2']} about {d1r} cm, then moves {cand['dir1']} about {d2r} cm", f"first moves {cand['dir1']} about {max(5, d1r-10)} cm, then moves {cand['dir2']} about {d2r+10} cm"], writer.subcounts[(task, sub)])
                q = f"Instruction: {ep['instruction']}. Which coarse horizontal manipulation program best matches this robot video?"
                writer.add({"id": f"{task}_{sid[task]:07d}", "source": "bridgedata_v2", "task_family": "robot_manipulation", "task_type": task, "subcat": sub, "question": q, "options": opts, "answer": correct, "answer_text": correct_text, "input": {"frame_paths": rel, "frame_labels": [str(i + 1) for i in range(len(rel))]}, "gt": {"instruction": ep["instruction"], "turn_frame": cand["k"], "segment_1": {"direction": cand["dir1"], "distance_cm": round(cand["d1"], 2), "rounded_cm": d1r}, "segment_2": {"direction": cand["dir2"], "distance_cm": round(cand["d2"], 2), "rounded_cm": d2r}, "turn_angle_deg": round(cand["ang"], 1), "sampled_frame_indices": cand["frames"], "shard": ep["shard"], "episode_index": ep["epidx"], "episode_id": ep["episode_id"]}})

        if episodes % 1000 == 0:
            print(f"[robot] episodes={episodes} samples={sum(writer.counts.values())} subcounts={dict(writer.subcounts)}", flush=True)

        # Stop early if every robot subcat known in this script is capped or no further useful categories are likely.
        # We still need to scan many episodes for sparse action_to_image directions, so keep going until all files are done.
    return {"episodes_processed": episodes, "image_cache_size": len(img_cache)}


def reverse_program_from_label(label: str) -> str | None:
    label = (label or "").lower().strip()
    if label == "moves mostly straight forward":
        return "turn around, move forward"
    if not label.startswith("moves forward"):
        return None
    turns = re.findall(r"turns (left|right)", label)
    if not turns:
        return "turn around, move forward"
    rev = ["turn around, move forward"]
    for t in reversed(turns):
        rev.append(f"turn {'right' if t == 'left' else 'left'} and move forward")
    return ", then ".join(rev)


def generate_video_reverse(writer: Writer, cap: int) -> dict[str, Any]:
    seen_path = set()
    per_video = Counter()
    source_counts = Counter()
    rows = 0
    for line in VIDEO_FULL.open():
        r = json.loads(line)
        if "simple_path_shape" not in r.get("task_type", ""):
            continue
        rows += 1
        path_id = r.get("path_id") or r.get("qa_id")
        if path_id in seen_path:
            continue
        label = r.get("answer_text") or (r.get("gt") or {}).get("motion_label")
        answer = reverse_program_from_label(label)
        if not answer:
            continue
        vals = ((r.get("gt") or {}).get("exact_pose_values") or {})
        dist = float(vals.get("straight_line_distance_m") or 0)
        plen = float(vals.get("cumulative_path_length_m") or 0)
        if dist < 3.0 or plen < 4.2:
            continue
        source = r.get("source") or "video"
        video_id = r.get("video_id") or path_id
        sub = answer
        task = "node_reverse_path_planning"
        if not cap_ok(writer, task, sub, cap):
            continue
        # Scene/video diversity cap: no single source video dominates.
        if per_video[(source, video_id)] >= 20:
            continue
        seen_path.add(path_id)
        per_video[(source, video_id)] += 1
        source_counts[source] += 1
        distractors = [answer.replace("left", "TMP").replace("right", "left").replace("TMP", "right"), "turn around, move forward", "turn around, move forward, then turn left and move forward", "turn around, move forward, then turn right and move forward", "move straight forward"]
        opts, correct = shuffled_options(answer, distractors, writer.subcounts[(task, sub)])
        q = "From the final frame, return to the starting location. The input is the original egocentric video from the target location to the current location. Assume the current facing direction follows the final stable path segment. Which coarse reverse path should the agent take?"
        inp = dict(r.get("input") or {})
        frame_paths = []
        # Keep video zip reference as a pseudo-media path in metadata; model packaging can decide whether to materialize frames or use video input.
        if inp.get("video_zip"):
            frame_paths = [inp["video_zip"]]
        writer.add({"id": f"video_reverse_path_{sum(writer.counts.values()) + 1:07d}", "source": source, "task_family": "video_navigation", "task_type": task, "subcat": sub, "question": q, "options": opts, "answer": correct, "answer_text": answer, "input": {"frame_paths": frame_paths, "frame_labels": ["video"]}, "gt": {"source_qa_id": r.get("qa_id"), "path_id": path_id, "video_id": video_id, "original_motion_label": label, "reverse_program": answer, "straight_line_distance_m": dist, "cumulative_path_length_m": plen, "source_input": inp, "source_gt": r.get("gt"), "source_metadata": r.get("metadata")}})
    return {"simple_rows_seen": rows, "unique_paths": len(seen_path), "source_counts": dict(source_counts)}


def write_summary(out: Path, writer: Writer, extra: dict[str, Any]) -> None:
    summary = {
        "total_samples": sum(writer.counts.values()),
        "by_task_type": dict(sorted(writer.counts.items())),
        "by_task_type_subcat": {f"{k[0]}::{k[1]}": v for k, v in sorted(writer.subcounts.items())},
        "by_source": dict(sorted(writer.sources.items())),
        "answer_distribution_by_task_type": {k: dict(v) for k, v in writer.answers.items()},
        "extra": extra,
        "notes": [
            "Each task_type::subcat is capped by --cap-per-subcat.",
            "BridgeData is processed in shuffled shard order for diversity; at most one sample per episode per robot subcategory is generated.",
            "Video reverse-path samples are derived from full pathfix simple-path rows; per source video is capped at 20.",
            "HTTP/HTTPS proxy is not used by the download script.",
        ],
    }
    (out / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def make_tar(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with tarfile.open(dst, "w") as tar:
        tar.add(src, arcname=src.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--bridge-dir", type=Path, default=BRIDGE_DIR)
    ap.add_argument("--cap-per-subcat", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--wait-for-bridge", action="store_true")
    ap.add_argument("--expected-shards", type=int, default=1024)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--video-only", action="store_true")
    ap.add_argument("--robot-only", action="store_true")
    ap.add_argument("--use-pilot-if-missing", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--tar-to-dataset2", action="store_true")
    args = ap.parse_args()

    if args.wait_for_bridge and not args.video_only:
        wait_for_tfrecords(args.bridge_dir, args.expected_shards, args.poll_seconds)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)

    writer = Writer(args.out)
    extra = {}
    try:
        if not args.robot_only:
            extra["video_reverse"] = generate_video_reverse(writer, args.cap_per_subcat)
        if not args.video_only:
            tfrecords = list_tfrecords(args.bridge_dir, args.use_pilot_if_missing)
            extra["robot_tfrecord_count"] = len(tfrecords)
            if not tfrecords:
                raise FileNotFoundError(f"No BridgeData tfrecords found in {args.bridge_dir}")
            extra["robot"] = generate_robot(writer, args.out, tfrecords, args.cap_per_subcat, args.seed, args.max_episodes)
    finally:
        writer.close()
    write_summary(args.out, writer, extra)
    if args.tar_to_dataset2:
        make_tar(args.out, DATASET2_TAR)
    print((args.out / "generation_summary.json").read_text())


if __name__ == "__main__":
    main()

