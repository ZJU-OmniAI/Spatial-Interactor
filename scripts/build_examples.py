#!/usr/bin/env python3
"""Build the website example index from the paper's selected appendix cases."""

import argparse
import json
import re
from pathlib import Path


TITLES = [
    "Object toggle",
    "Open / close",
    "Object rotation",
    "Object removal",
    "Position swapping",
    "Dynamic occlusion",
    "Dynamic distance",
    "Action-to-image matching",
    "Operation magnitude",
    "Operation ordering",
    "Metric camera motion",
    "Camera rotation",
    "Composed camera motion",
    "Multi-frame action chain",
    "Translation comparison",
    "Translation comparison",
    "Rotation comparison",
    "Rotation comparison",
    "Camera temporal ordering",
    "Camera temporal ordering",
    "Overlap-based localization",
    "Parallax depth",
    "Imagined perspective",
    "Direction after motion",
    "Two-view direction",
    "Visibility after motion",
    "Disappearance trend",
    "Multi-step operation",
    "Reverse-path reasoning",
    "Metric trajectory",
    "Simulated trajectory shape",
    "Real-world trajectory shape",
    "Metric trajectory",
    "Turning-interval identification",
    "Revisited-location identification",
]

QUESTION_OVERRIDES = {
    "E08": "Starting from the leftmost view, which candidate matches a 20 cm rightward end-effector motion?",
    "E10": "For the instruction 'put the silver pot in the bottom-right corner of the sink,' what is the chronological order of the four keyframes?",
    "E14": "Which motion chain connects the consecutive views?",
    "E15": "From the same start view, which rightward endpoint has the larger translation?",
    "E16": "From the same start view, which backward endpoint has the larger translation?",
    "E17": "From the same start view, which endpoint has the larger upward rotation?",
    "E18": "From the same start view, which endpoint has the larger leftward rotation?",
    "E19": "After the fixed start view, what is the chronological order of the other three views?",
    "E20": "What is the chronological order of the four displayed camera views?",
    "E28": "For the instruction 'put pepper in pan,' which coarse horizontal manipulation program matches the sequence?",
    "E29": "Which reverse path returns to the starting location?",
    "E30": "What are the endpoint displacement, initial-view turn, and total path length?",
    "E33": "What are the endpoint displacement, initial-view turn, and total path length?",
    "E34": "Which temporal quarter contains a moving turn?",
}

ANSWER_TEXT = {
    "E14": "C. Rotate right 60 degrees, rotate right 45 degrees, then move forward 0.4 m.",
    "E15": "B. The rightmost endpoint: 0.86 m, versus 0.44 m for the middle endpoint.",
    "E16": "D. The middle endpoint: 0.50 m, versus 0.25 m for the rightmost endpoint.",
    "E17": "D. The rightmost endpoint: 36 degrees, versus 12 degrees for the middle endpoint.",
    "E18": "C. The middle endpoint: 60 degrees, versus 45 degrees for the rightmost endpoint.",
}

SOURCE_NAMES = {
    "AI2THOR": "AI2-THOR",
    "PROC": "ProcTHOR",
    "bridgedata_v2": "BridgeData V2",
    "scannetv2": "ScanNet",
    "scannetpp": "ScanNet++",
    "multiscan": "MultiScan",
    "roomtour3d": "RoomTour3D",
    "SIMS-V": "SIMS-V",
    "ScanNet++": "ScanNet++",
}


def domain_for(source):
    if source in {"AI2THOR", "PROC", "SIMS-V"}:
        return "Simulated"
    if source == "bridgedata_v2":
        return "Robot trajectory"
    if source == "roomtour3d":
        return "Real-world video"
    return "Real-world views"


def clean_question(raw):
    lines = [line.strip() for line in raw.splitlines()]
    kept = []
    for line in lines:
        if not line:
            continue
        if "<image>" in line or line == "<video>":
            continue
        if line == "Options:" or re.match(r"^[A-E]\.\s", line):
            continue
        if line.startswith(("Answer with", "Reply exactly", "Reply with")):
            continue
        kept.append(line)
    return " ".join(kept)


def parse_options(raw):
    options = []
    for line in raw.splitlines():
        match = re.match(r"^([A-E])\.\s*(.+)$", line.strip())
        if match:
            options.append(match.group(2).strip())
    return options


def answer_letter(raw):
    direct = raw.strip()
    if re.fullmatch(r"[A-E]", direct):
        return direct
    match = re.search(r"Answer:\s*([A-E])(?:\s|$)", raw)
    if not match:
        raise ValueError(f"Could not parse answer from: {raw[-100:]}")
    return match.group(1)


def trajectory_fields(case_id):
    if case_id == "E30":
        return [
            {"label": "Endpoint displacement", "options": ["4.0 m", "3.5 m", "2.5 m", "3.0 m"], "answer": "B"},
            {"label": "Initial-view turn", "options": ["0 degrees", "Right 30 degrees", "Right 90 degrees", "Right 60 degrees"], "answer": "D"},
            {"label": "Path length", "options": ["4.5 m", "6.5 m", "5.5 m", "3.5 m"], "answer": "A"},
        ]
    if case_id == "E33":
        return [
            {"label": "Endpoint displacement", "options": ["6.5 m", "7.0 m", "6.0 m", "5.5 m"], "answer": "A"},
            {"label": "Initial-view turn", "options": ["Right 30 degrees", "Right 90 degrees", "Right 60 degrees", "0 degrees"], "answer": "A"},
            {"label": "Path length", "options": ["6.5 m", "8.5 m", "9.5 m", "7.5 m"], "answer": "B"},
        ]
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=Path,
        default=root.parent / "AuthorKit27" / "appendix_artifacts" / "selected_case_index.json",
    )
    parser.add_argument("--output", type=Path, default=root / "examples.js")
    args = parser.parse_args()

    records = json.loads(args.index.read_text(encoding="utf-8"))["records"]
    if len(records) != 35:
        raise ValueError(f"Expected 35 appendix cases, found {len(records)}")

    cases = {}
    for index, record in enumerate(records, start=1):
        case_id = f"E{index:02d}"
        case = {
            "title": TITLES[index - 1],
            "source": SOURCE_NAMES[record["source"]],
            "domain": domain_for(record["source"]),
            "question": QUESTION_OVERRIDES.get(case_id, clean_question(record["question"])),
        }
        fields = trajectory_fields(case_id)
        if fields:
            case["fields"] = fields
            case["answer"] = " / ".join(field["answer"] for field in fields)
        elif case_id in ANSWER_TEXT:
            case["options"] = []
            case["answer"] = answer_letter(record["answer"])
            case["answerText"] = ANSWER_TEXT[case_id]
        else:
            case["options"] = parse_options(record["question"])
            case["answer"] = answer_letter(record["answer"])
        cases[case_id] = case

    # The paper defines L3 over complete camera trajectories. The multi-stage
    # robot operation is therefore displayed with L1 operation reasoning.
    cases["E34"]["options"] = [
        "The final quarter, around frames 25-32",
        "The first quarter, around frames 1-8",
        "The third quarter, around frames 17-24",
        "The second quarter, around frames 9-16",
    ]
    cases["E34"]["answer"] = "D"

    levels = {
        "l1": {
            "title": "L1: Passive world-state transitions",
            "count": "15,109",
            "description": "Explain external-world changes under an approximately stable viewpoint, from object displacement and articulation to ordered operations and their outcomes.",
            "tasks": ["Object displacement", "Attribute & articulation changes", "Occlusion & visibility", "Relative configuration", "Single- & multi-step operations"],
            "cases": [f"E{i:02d}" for i in range(1, 11)] + ["E28"],
        },
        "l2": {
            "title": "L2: Active self-state transitions",
            "count": "69,487",
            "description": "Attribute observation changes to camera translation, rotation, and elevation while preserving scene identity across viewpoints.",
            "tasks": ["Motion inference", "Magnitude comparison", "Motion composition", "Temporal ordering", "Cross-view spatial inference"],
            "cases": [f"E{i:02d}" for i in range(11, 28)],
        },
        "l3": {
            "title": "L3: Long-horizon transition integration",
            "count": "22,922",
            "description": "Compose successive transitions over complete camera trajectories to recover global trajectory properties and key locations.",
            "tasks": ["Path length", "Endpoint displacement", "Trajectory shape", "Turning intervals", "Revisited locations & reverse paths"],
            "cases": [f"E{i:02d}" for i in range(29, 36)],
        },
    }
    payload = json.dumps({"levels": levels, "cases": cases}, ensure_ascii=True, indent=2)
    args.output.write_text(
        "/* Generated from the selected appendix cases. */\n"
        f"Object.assign(window.SPATIAL_DATA, {payload});\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
