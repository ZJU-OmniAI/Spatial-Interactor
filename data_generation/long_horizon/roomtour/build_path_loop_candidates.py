#!/usr/bin/env python3
import html
import json
import math
import re
from pathlib import Path

import build_video_stream_task_candidates_20260602 as base


COMBINED = Path("/path/to/workspace/combined_camera_roomtour_metric_m_no_boundary_20260529/combined_camera_roomtour_metric_m_no_boundary.jsonl")
OUT = Path("/path/to/workspace/roomtour_path_loop_candidates_20260602")
ASSETS = OUT / "assets"

SIMPLE_LABELS = {
    "mostly moves forward through the scene",
    "moves forward and turns right",
    "moves forward and turns left",
}


def load_rows():
    rows = []
    with COMBINED.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fnum(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def qa_id(row):
    return row.get("qa_id") or ""


def video_key(row):
    inp = row.get("input") or {}
    if inp.get("video_zip"):
        return Path(inp["video_zip"]).stem
    return qa_id(row).split("_coarse_video_motion")[0].split("_metric_m_pose_interval_summary")[0]


def metrics(row):
    md = row.get("metadata") or {}
    return md.get("pose_metrics") or base.metrics(row)


def metric_text(row):
    m = metrics(row)
    parts = []
    for k in [
        "straight_line_distance_m",
        "axis_angle_deg",
        "cumulative_path_length_m",
        "path_to_straight_ratio",
    ]:
        v = fnum(m.get(k))
        if v is not None:
            parts.append(f"{k}: {v:.2f}")
    return "; ".join(parts)


def simple_path_score(row):
    ans = row.get("answer_text") or ""
    m = metrics(row)
    d = fnum(m.get("straight_line_distance_m")) or 0.0
    path = fnum(m.get("cumulative_path_length_m")) or 0.0
    ratio = fnum(m.get("path_to_straight_ratio")) or 9.0
    axis = fnum(m.get("axis_angle_deg")) or 0.0

    score = 0.0
    if ans == "mostly moves forward through the scene":
        score += 60
        score += max(0, 12 - abs(axis) * 0.2)
        score += max(0, 10 - abs(ratio - 1.15) * 10)
    elif ans == "moves forward and turns right":
        score += 48
        score += max(0, 8 - abs(ratio - 1.45) * 5)
    elif ans == "moves forward and turns left":
        score += 48
        score += max(0, 8 - abs(ratio - 1.45) * 5)

    if 3.0 <= d <= 9.0:
        score += 12
    if 4.0 <= path <= 12.0:
        score += 12
    if ratio > 2.4:
        score -= 18
    return score


def loop_score(row):
    m = metrics(row)
    d = fnum(m.get("straight_line_distance_m"))
    path = fnum(m.get("cumulative_path_length_m"))
    ratio = fnum(m.get("path_to_straight_ratio"))
    if d is None or path is None or ratio is None:
        return None
    if d > 3.0 or path < 8.0 or ratio < 3.0:
        return None

    score = 0.0
    score += max(0, 40 - d * 10)
    score += min(35, ratio * 2.0)
    if 9.0 <= path <= 16.0:
        score += 25
    elif path > 16.0:
        score += 12
    else:
        score += 10
    ans = row.get("answer_text") or ""
    if "total path length more than 12 m" in ans:
        score += 10
    elif "total path length 8-12 m" in ans:
        score += 8
    return score


def pick_ranked(rows, score_fn, limit, per_answer=10):
    scored = []
    for row in rows:
        score = score_fn(row)
        if score is not None:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = []
    seen_video = set()
    answer_counts = {}
    for score, row in scored:
        vk = video_key(row)
        ans = row.get("answer_text") or ""
        if vk in seen_video:
            continue
        if answer_counts.get(ans, 0) >= per_answer:
            continue
        picked.append((score, row))
        seen_video.add(vk)
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
        if len(picked) >= limit:
            break
    return picked


def options_block(row):
    lines = []
    for opt in row.get("options") or []:
        lines.append(f"{html.escape(str(opt.get('key')))}. {html.escape(str(opt.get('text')))}")
    return "<br>".join(lines)


def card(entry):
    row = entry["row"]
    code = entry["code"]
    q = html.escape(row.get("question") or "")
    options = options_block(row)
    ans = html.escape(str(row.get("answer") or row.get("correct_option") or ""))
    ans_text = html.escape(row.get("answer_text") or "")
    img = html.escape(entry["image"])
    qid = html.escape(qa_id(row))
    metrics_line = html.escape(metric_text(row))
    return f"""
    <article class="card">
      <div class="code">{html.escape(code)}</div>
      <div class="meta"><span>{html.escape(video_key(row))}</span><span>{html.escape(row.get('task_type') or '')}</span></div>
      <figure><img src="{img}" loading="lazy"><figcaption>8 sampled frames from the 32-frame RoomTour3D window</figcaption></figure>
      <div class="qa">
        <div class="label">Question</div>
        <p>{q}<br>{options}</p>
        <div class="label">Answer</div>
        <p class="answer">{ans} <span>· {ans_text}</span></p>
      </div>
      <div class="metrics"><b>GT:</b> {metrics_line}<br><b>qa_id:</b> {qid} · score {entry['score']:.1f}</div>
    </article>"""


def write_page(sections):
    chunks = []
    for title, note, entries in sections:
        chunks.append(f"<h2>{html.escape(title)}</h2><p class=\"section-note\">{html.escape(note)}</p><div class=\"grid\">")
        chunks.extend(card(e) for e in entries)
        chunks.append("</div>")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoomTour3D Path and Loop Candidates</title>
<style>
:root{{--ink:#17202b;--muted:#5b6572;--line:#d8dee8;--soft:#f7f9fc;--ok:#1f6f56}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:#fff;line-height:1.45}}
header{{padding:28px 38px 20px;background:#f8fafc;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}}
h1{{margin:0 0 7px;font-size:30px}}.hint{{margin:0;color:var(--muted)}}main{{padding:24px 38px 54px}}
h2{{border-top:3px solid var(--ink);padding-top:16px;margin:30px 0 8px;font-size:22px}}.section-note{{margin:0 0 14px;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.card{{position:relative;border:1px solid var(--line);border-radius:8px;padding:13px;background:#fff}}
.code{{position:absolute;right:13px;top:11px;background:#1f2933;color:white;border-radius:6px;padding:4px 9px;font-weight:800}}
.meta{{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px;border-bottom:1px solid var(--line);padding:0 58px 8px 0;margin-bottom:10px}}
figure{{margin:0 0 10px;background:#101418;border-radius:6px;overflow:hidden;border:1px solid #101418}}
img{{display:block;width:100%;aspect-ratio:20/7;object-fit:contain;background:#101418}}
figcaption{{padding:5px 7px;background:#18202a;color:#e8edf4;font-size:12px}}
.qa{{font-size:13px;border-top:1px solid var(--line);padding-top:9px}}.label{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:800;margin-bottom:4px}}
.qa p{{margin:0 0 8px}}.answer{{font-weight:800;color:var(--ok)}}.answer span{{color:var(--ink);font-weight:500}}
.metrics{{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px;margin-top:8px;word-break:break-word}}
@media(max-width:1100px){{header,main{{padding-left:16px;padding-right:16px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>RoomTour3D Path and Loop Candidates</h1><p class="hint">先看图和编号。P 是简单路径形状候选；L 是直线位移很小、总路程很长，接近绕圈回原点的 metric 候选。</p></header>
<main>{''.join(chunks)}</main></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    coarse_rows = [
        r for r in rows
        if r.get("source") == "roomtour3d"
        and r.get("task_type") == "roomtour3d_coarse_video_motion"
        and r.get("answer_text") in SIMPLE_LABELS
    ]
    metric_rows = [
        r for r in rows
        if r.get("source") == "roomtour3d"
        and r.get("task_type") == "roomtour3d_metric_pose_interval_summary"
    ]

    path_picked = pick_ranked(coarse_rows, simple_path_score, 30, per_answer=12)
    loop_picked = pick_ranked(metric_rows, loop_score, 36, per_answer=12)

    all_entries = []
    sections = []
    for prefix, title, note, picked in [
        ("P", "Class 1: Simple Path Shape Candidates", "只保留直走、直走后左/右转这类路径形状；旧语义类已经排除。", path_picked),
        ("L", "Class 2: Loop / Return-Near-Start Metric Candidates", "这些是直线距离小但总路径长的窗口，适合做“绕一圈几乎回原地”的展示候选。", loop_picked),
    ]:
        entries = []
        for i, (score, row) in enumerate(picked, 1):
            code = f"{prefix}{i}"
            image_rel = f"assets/{code}.jpg"
            base.make_contact(row, OUT / image_rel)
            entry = {"code": code, "score": score, "row": row, "image": image_rel}
            entries.append(entry)
            m = metrics(row)
            inp = row.get("input") or {}
            all_entries.append({
                "code": code,
                "source": row.get("source"),
                "task_type": row.get("task_type"),
                "video_key": video_key(row),
                "qa_id": qa_id(row),
                "answer": row.get("answer") or row.get("correct_option"),
                "answer_text": row.get("answer_text"),
                "metrics": {
                    "straight_line_distance_m": m.get("straight_line_distance_m"),
                    "axis_angle_deg": m.get("axis_angle_deg"),
                    "cumulative_path_length_m": m.get("cumulative_path_length_m"),
                    "path_to_straight_ratio": m.get("path_to_straight_ratio"),
                },
                "image": image_rel,
                "video_zip": inp.get("video_zip"),
                "sampled_frame_names_32": inp.get("sampled_frame_names_32"),
                "candidate_score": round(score, 3),
            })
        sections.append((title, note, entries))

    write_page(sections)
    (OUT / "selected_candidates.json").write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    page = (OUT / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'<img src="([^"]+)"', page)
    missing = [r for r in refs if not (OUT / r).exists()]
    print(f"wrote {OUT / 'index.html'}")
    print(f"cards {len(all_entries)}, refs {len(refs)}, missing {len(missing)}")
    print(f"path candidates {len(path_picked)}, loop candidates {len(loop_picked)}")


if __name__ == "__main__":
    main()
