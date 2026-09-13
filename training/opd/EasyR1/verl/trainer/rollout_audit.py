from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import torch

from ..protocol import DataProto
from .core_algos import compute_kl


_ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+)$", re.I | re.S)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def _canonical_answer(response: str) -> str:
    match = _ANSWER_RE.search(response or "")
    if not match:
        return "<missing>"
    answer = match.group(1).strip().splitlines()[0].strip()
    try:
        parsed = json.loads(answer)
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return answer[:256] or "<empty>"


def _masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum(dim=-1).clamp_min(1)
    means = (values * mask).sum(dim=-1) / denominator
    return means.detach().float().cpu().numpy()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return _float(np.corrcoef(left, right)[0, 1])


class RolloutAuditLogger:
    """Write reward-hacking diagnostics and sampled group-level HTML reports."""

    def __init__(
        self,
        output_dir: str,
        tokenizer: Any,
        log_freq: int,
        groups_per_step: int,
        expected_group_size: int,
        assistant_prefill: str = "",
        include_privileged_context: bool = True,
        max_response_chars: int = 8000,
        kl_penalty: str = "low_var_kl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.media_dir = self.output_dir / "media"
        self.tokenizer = tokenizer
        self.log_freq = max(1, int(log_freq))
        self.groups_per_step = max(0, int(groups_per_step))
        self.expected_group_size = max(1, int(expected_group_size))
        self.assistant_prefill = str(assistant_prefill or "")
        self.include_privileged_context = bool(include_privileged_context)
        self.max_response_chars = max(256, int(max_response_chars))
        self.kl_penalty = str(kl_penalty)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def _value(self, batch: DataProto, key: str, row: int, default: Any = "") -> Any:
        values = batch.non_tensor_batch.get(key)
        if values is None or row >= len(values):
            return default
        return _json_value(values[row])

    def _responses(self, batch: DataProto) -> list[str]:
        response_ids = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        outputs = []
        for row in range(len(batch)):
            length = int(response_mask[row].sum().item())
            generated = self.tokenizer.decode(response_ids[row, :length], skip_special_tokens=True)
            outputs.append((self.assistant_prefill + generated)[: self.max_response_chars])
        return outputs

    def _groups(self, batch: DataProto) -> list[list[int]]:
        uids = batch.non_tensor_batch.get("uid")
        if uids is None:
            return [
                list(range(start, min(start + self.expected_group_size, len(batch))))
                for start in range(0, len(batch), self.expected_group_size)
            ]
        grouped: dict[str, list[int]] = {}
        for row, uid in enumerate(uids):
            grouped.setdefault(str(uid), []).append(row)
        return list(grouped.values())

    def _reward_arrays(self, reward_metrics: dict[str, list[float]], batch_size: int) -> dict[str, np.ndarray]:
        arrays = {}
        for key, values in reward_metrics.items():
            if len(values) != batch_size:
                continue
            try:
                arrays[str(key)] = np.asarray(values, dtype=np.float64)
            except Exception:
                continue
        return arrays

    def _sequence_diagnostics(self, batch: DataProto) -> dict[str, np.ndarray]:
        response_mask = batch.batch["response_mask"]
        diagnostics = {
            "response_tokens": response_mask.sum(dim=-1).detach().float().cpu().numpy(),
        }
        advantage_key = "grpo_advantages" if "grpo_advantages" in batch.batch else "advantages"
        if advantage_key in batch.batch:
            diagnostics["grpo_advantage"] = _masked_row_mean(batch.batch[advantage_key], response_mask)
        if "opd_teacher_delta" in batch.batch and "opd_cot_mask" in batch.batch:
            diagnostics["opd_delta"] = _masked_row_mean(
                batch.batch["opd_teacher_delta"], batch.batch["opd_cot_mask"]
            )
        if "opd_teacher_fkl" in batch.batch and "opd_cot_mask" in batch.batch:
            diagnostics["opd_topk_fkl"] = _masked_row_mean(
                batch.batch["opd_teacher_fkl"], batch.batch["opd_cot_mask"]
            )
            diagnostics["opd_weighted_fkl"] = (
                diagnostics["opd_topk_fkl"] * float(batch.meta_info.get("opd_loss_weight", 0.0))
            )
        if "old_log_probs" in batch.batch and "ref_log_probs" in batch.batch:
            token_kl = compute_kl(
                batch.batch["old_log_probs"],
                batch.batch["ref_log_probs"],
                kl_penalty=self.kl_penalty,
            )
            diagnostics["sequence_kl"] = _masked_row_mean(token_kl, response_mask)
        return diagnostics

    def _scalar_metrics(
        self,
        batch: DataProto,
        groups: list[list[int]],
        responses: list[str],
        rewards: dict[str, np.ndarray],
        sequence: dict[str, np.ndarray],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        overall = rewards.get("overall", np.zeros(len(batch), dtype=np.float64))
        task = rewards.get("task", np.zeros(len(batch), dtype=np.float64))
        format_score = rewards.get("format", np.zeros(len(batch), dtype=np.float64))
        metrics = {
            "audit/reward/overall_std": _float(np.std(overall)),
            "audit/reward/overall_q10": _float(np.quantile(overall, 0.10)),
            "audit/reward/overall_q50": _float(np.quantile(overall, 0.50)),
            "audit/reward/overall_q90": _float(np.quantile(overall, 0.90)),
            "audit/reward/zero_fraction": _float(np.mean(overall <= 1e-8)),
            "audit/reward/perfect_fraction": _float(np.mean(overall >= 1.0 - 1e-8)),
            "audit/reward/format_only_fraction": _float(np.mean((task <= 1e-8) & (format_score > 1e-8))),
            "audit/reward/task_without_format_fraction": _float(
                np.mean((task > 1e-8) & (format_score <= 1e-8))
            ),
            "audit/reward/format_minus_task": _float(np.mean(format_score) - np.mean(task)),
            "audit/reward_length_correlation": _correlation(
                overall, sequence["response_tokens"]
            ),
        }
        parsed_answers = [_canonical_answer(response) for response in responses]
        answer_counts = Counter(parsed_answers)
        numeric_answers = []
        for answer in parsed_answers:
            try:
                numeric_answers.append(float(answer))
            except Exception:
                continue
        metrics.update(
            {
                "audit/answer/missing_fraction": _float(
                    np.mean([answer in {"<missing>", "<empty>"} for answer in parsed_answers])
                ),
                "audit/answer/mode_fraction": max(answer_counts.values(), default=0) / max(len(batch), 1),
                "audit/answer/unique_fraction": len(answer_counts) / max(len(batch), 1),
            }
        )
        if numeric_answers:
            numeric_array = np.asarray(numeric_answers, dtype=np.float64)
            metrics.update(
                {
                    "audit/numeric/count": float(len(numeric_array)),
                    "audit/numeric/nonpositive_fraction": _float(np.mean(numeric_array <= 0)),
                    "audit/numeric/q10": _float(np.quantile(numeric_array, 0.10)),
                    "audit/numeric/q50": _float(np.quantile(numeric_array, 0.50)),
                    "audit/numeric/q90": _float(np.quantile(numeric_array, 0.90)),
                }
            )

        group_rows = []
        for rows in groups:
            group_rewards = overall[rows]
            answers = [_canonical_answer(responses[row]) for row in rows]
            answer_counts = Counter(answers)
            probabilities = np.asarray(list(answer_counts.values()), dtype=np.float64) / max(len(rows), 1)
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            normalized_entropy = entropy / max(math.log(max(len(rows), 2)), 1e-12)
            reward_std = _float(np.std(group_rewards))
            format_only = bool(np.all(task[rows] <= 1e-8) and np.mean(format_score[rows]) > 0.99)
            collapsed = len(answer_counts) == 1
            no_signal = reward_std <= 1e-8
            suspicious = no_signal or collapsed or format_only
            group_rows.append(
                {
                    "rows": rows,
                    "reward_mean": _float(np.mean(group_rewards)),
                    "reward_std": reward_std,
                    "answer_entropy": _float(normalized_entropy),
                    "unique_answer_fraction": len(answer_counts) / max(len(rows), 1),
                    "answer_collapsed": collapsed,
                    "no_reward_signal": no_signal,
                    "format_only": format_only,
                    "suspicious": suspicious,
                    "size_mismatch": len(rows) != self.expected_group_size,
                }
            )

        if group_rows:
            metrics.update(
                {
                    "audit/group/reward_std_mean": _float(
                        np.mean([group["reward_std"] for group in group_rows])
                    ),
                    "audit/group/zero_variance_fraction": _float(
                        np.mean([group["no_reward_signal"] for group in group_rows])
                    ),
                    "audit/group/answer_collapse_fraction": _float(
                        np.mean([group["answer_collapsed"] for group in group_rows])
                    ),
                    "audit/group/answer_entropy_mean": _float(
                        np.mean([group["answer_entropy"] for group in group_rows])
                    ),
                    "audit/group/unique_answer_fraction_mean": _float(
                        np.mean([group["unique_answer_fraction"] for group in group_rows])
                    ),
                    "audit/group/suspicious_fraction": _float(
                        np.mean([group["suspicious"] for group in group_rows])
                    ),
                    "audit/group/size_mismatch_fraction": _float(
                        np.mean([group["size_mismatch"] for group in group_rows])
                    ),
                }
            )

        for name in (
            "grpo_advantage",
            "opd_delta",
            "opd_topk_fkl",
            "opd_weighted_fkl",
            "sequence_kl",
        ):
            values = sequence.get(name)
            if values is None:
                continue
            metrics[f"audit/{name}/mean"] = _float(np.mean(values))
            metrics[f"audit/{name}/std"] = _float(np.std(values))
            metrics[f"audit/{name}/q90_abs"] = _float(np.quantile(np.abs(values), 0.90))

        task_types = batch.non_tensor_batch.get("task_type")
        if task_types is not None:
            by_task: dict[str, list[int]] = defaultdict(list)
            for row, task_type in enumerate(task_types):
                by_task[str(task_type)].append(row)
            for task_type, rows in by_task.items():
                safe_task = _SAFE_NAME_RE.sub("_", task_type).strip("_") or "unknown"
                metrics[f"audit/task/{safe_task}/overall_mean"] = _float(np.mean(overall[rows]))
                metrics[f"audit/task/{safe_task}/task_mean"] = _float(np.mean(task[rows]))

        return metrics, group_rows

    def _media_link(self, media_path: str) -> Optional[str]:
        source = Path(media_path)
        if not source.is_file():
            return None
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
        suffix = source.suffix.lower() or ".bin"
        destination = self.media_dir / f"{digest}{suffix}"
        if not os.path.lexists(destination):
            destination.symlink_to(source.resolve())
        return f"../media/{quote(destination.name)}"

    def _response_record(
        self,
        row: int,
        response: str,
        rewards: dict[str, np.ndarray],
        sequence: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        return {
            "rollout_index": row,
            "response": response,
            "parsed_answer": _canonical_answer(response),
            "rewards": {key: _float(values[row]) for key, values in rewards.items()},
            "response_tokens": _float(sequence["response_tokens"][row]),
            "grpo_advantage": _float(sequence.get("grpo_advantage", np.zeros(row + 1))[row]),
            "opd_delta": _float(sequence.get("opd_delta", np.zeros(row + 1))[row]),
            "opd_topk_fkl": _float(sequence.get("opd_topk_fkl", np.zeros(row + 1))[row]),
            "opd_weighted_fkl": _float(sequence.get("opd_weighted_fkl", np.zeros(row + 1))[row]),
            "sequence_kl": _float(sequence.get("sequence_kl", np.zeros(row + 1))[row]),
        }

    def _record_group(
        self,
        step: int,
        audit_index: int,
        batch: DataProto,
        group: dict[str, Any],
        responses: list[str],
        rewards: dict[str, np.ndarray],
        sequence: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        rows = group["rows"]
        first = rows[0]
        media_path = str(self._value(batch, "audit_media_path", first, ""))
        privileged = str(self._value(batch, "audit_privileged_prompt", first, ""))
        record = {
            "schema_version": 1,
            "step": int(step),
            "audit_index": int(audit_index),
            "uid": str(self._value(batch, "uid", first, "")),
            "source_id": str(self._value(batch, "source_id", first, "")),
            "task_type": str(self._value(batch, "task_type", first, "")),
            "trace_id": str(self._value(batch, "trace_id", first, "")),
            "media_path": media_path,
            "media_exists": bool(media_path and Path(media_path).is_file()),
            "question": str(self._value(batch, "audit_prompt", first, "")),
            "privileged_context": privileged if self.include_privileged_context else "",
            "ground_truth": self._value(batch, "ground_truth", first, ""),
            "group": {key: value for key, value in group.items() if key != "rows"},
            "responses": [
                self._response_record(row, responses[row], rewards, sequence) for row in rows
            ],
        }
        record["responses"].sort(key=lambda item: item["rewards"].get("overall", 0.0), reverse=True)
        for rank, response in enumerate(record["responses"], start=1):
            response["reward_rank"] = rank
        return record

    def _write_group_html(self, record: dict[str, Any]) -> str:
        step_dir = self.output_dir / f"step_{record['step']:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        filename = f"group_{record['audit_index']:02d}.html"
        output_path = step_dir / filename
        media_href = self._media_link(record["media_path"])
        suffix = Path(record["media_path"]).suffix.lower()
        if media_href and suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            media_html = f'<img class="media" src="{html.escape(media_href)}" alt="audit media">'
        elif media_href:
            media_html = (
                f'<video class="media" controls preload="metadata" src="{html.escape(media_href)}"></video>'
            )
        else:
            media_html = '<p class="warning">Media path is unavailable on this server.</p>'

        response_rows = []
        for item in record["responses"]:
            scores = item["rewards"]
            response_rows.append(
                "<tr>"
                f"<td>{item['reward_rank']}</td>"
                f"<td>{scores.get('overall', 0.0):.4f}</td>"
                f"<td>{scores.get('task', 0.0):.4f}</td>"
                f"<td>{scores.get('format', 0.0):.4f}</td>"
                f"<td>{item['grpo_advantage']:.5f}</td>"
                f"<td>{item['opd_delta']:.5f}</td>"
                f"<td>{item['opd_topk_fkl']:.6f}</td>"
                f"<td>{item['opd_weighted_fkl']:.6f}</td>"
                f"<td>{item['sequence_kl']:.5f}</td>"
                f"<td><code>{html.escape(item['parsed_answer'])}</code></td>"
                f"<td><pre>{html.escape(item['response'])}</pre></td>"
                "</tr>"
            )

        privileged_html = ""
        if record["privileged_context"]:
            privileged_html = (
                "<details><summary>Privileged visual context</summary>"
                f"<pre>{html.escape(record['privileged_context'])}</pre></details>"
            )
        flags = [name for name in ("no_reward_signal", "answer_collapsed", "format_only") if record["group"][name]]
        flags_text = ", ".join(flags) if flags else "none"
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rollout audit step {record['step']}</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;color:#202124;background:#f7f8fa}}header,main{{max-width:1500px;margin:auto;padding:18px 24px}}header{{background:#202124;color:white;max-width:none}}h1{{font-size:20px;margin:0}}h2{{font-size:16px;margin-top:24px}}.meta{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:8px}}.meta div{{background:white;border:1px solid #dfe1e5;padding:8px}}.media{{display:block;max-width:960px;max-height:540px;width:100%;background:#111}}pre{{white-space:pre-wrap;word-break:break-word;margin:0;max-height:320px;overflow:auto}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{border:1px solid #dfe1e5;padding:7px;vertical-align:top;text-align:left}}th{{position:sticky;top:0;background:#eef1f4}}code{{white-space:nowrap}}details{{margin-top:16px;background:white;border:1px solid #dfe1e5;padding:10px}}.warning{{color:#a33}}@media(max-width:800px){{.meta{{grid-template-columns:1fr 1fr}}header,main{{padding:12px}}table{{display:block;overflow-x:auto}}}}
</style></head><body>
<header><h1>Rollout audit: step {record['step']} / {html.escape(record['task_type'])}</h1></header><main>
<div class="meta"><div><b>Source</b><br>{html.escape(record['source_id'])}</div><div><b>Trace</b><br>{html.escape(record['trace_id'])}</div><div><b>Reward mean / std</b><br>{record['group']['reward_mean']:.4f} / {record['group']['reward_std']:.4f}</div><div><b>Risk flags</b><br>{html.escape(flags_text)}</div></div>
<h2>Video</h2>{media_html}
<h2>Question</h2><pre>{html.escape(record['question'])}</pre>
<h2>Ground truth</h2><pre>{html.escape(str(record['ground_truth']))}</pre>{privileged_html}
<h2>G={len(record['responses'])} responses, sorted by overall reward</h2>
<table><thead><tr><th>Rank</th><th>Overall</th><th>Task</th><th>Format</th><th>GRPO adv</th><th>Sampled-token delta</th><th>Top-k FKL</th><th>Weighted FKL</th><th>Ref KL</th><th>Parsed</th><th>Response</th></tr></thead><tbody>{''.join(response_rows)}</tbody></table>
</main></body></html>"""
        output_path.write_text(document, encoding="utf-8")
        return f"step_{record['step']:08d}/{filename}"

    def _append_json(self, path: Path, value: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=_json_value) + "\n")

    def _write_index(self) -> None:
        entries_path = self.output_dir / "index.jsonl"
        entries = []
        if entries_path.exists():
            with entries_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
        rows = []
        for entry in reversed(entries):
            rows.append(
                "<tr>"
                f"<td>{entry['step']}</td><td>{html.escape(entry['task_type'])}</td>"
                f"<td>{html.escape(entry['source_id'])}</td><td>{entry['reward_mean']:.4f}</td>"
                f"<td>{entry['reward_std']:.4f}</td><td>{html.escape(entry['flags'])}</td>"
                f"<td><a href=\"{html.escape(entry['href'])}\">open</a></td></tr>"
            )
        document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spatial-Interactor rollout audit</title><style>body{{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#202124}}h1{{font-size:22px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f1f3f4;position:sticky;top:0}}a{{color:#075fc7}}</style></head><body><h1>Spatial-Interactor rollout reward audit</h1><p>Groups are sampled for reward variance, answer collapse, and format-only behavior. Flags are indicators for review, not proof of reward hacking.</p><table><thead><tr><th>Step</th><th>Task</th><th>Source</th><th>Reward</th><th>Std</th><th>Flags</th><th>Report</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
        temporary = self.output_dir / "index.html.tmp"
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(self.output_dir / "index.html")

    def log_step(
        self,
        step: int,
        batch: DataProto,
        reward_metrics: dict[str, list[float]],
    ) -> dict[str, float]:
        responses = self._responses(batch)
        groups = self._groups(batch)
        rewards = self._reward_arrays(reward_metrics, len(batch))
        sequence = self._sequence_diagnostics(batch)
        metrics, group_rows = self._scalar_metrics(batch, groups, responses, rewards, sequence)
        metrics["audit/pages_written"] = 0.0

        if self.groups_per_step <= 0 or step % self.log_freq != 0:
            return metrics

        ranked_groups = sorted(
            group_rows,
            key=lambda group: (
                group["suspicious"],
                group["format_only"],
                group["answer_collapsed"],
                group["no_reward_signal"],
                -group["reward_std"],
            ),
            reverse=True,
        )[: self.groups_per_step]
        for audit_index, group in enumerate(ranked_groups):
            record = self._record_group(
                step, audit_index, batch, group, responses, rewards, sequence
            )
            self._append_json(self.output_dir / "rollout_audit.jsonl", record)
            href = self._write_group_html(record)
            flags = [
                name
                for name in ("no_reward_signal", "answer_collapsed", "format_only")
                if group[name]
            ]
            self._append_json(
                self.output_dir / "index.jsonl",
                {
                    "step": step,
                    "task_type": record["task_type"],
                    "source_id": record["source_id"],
                    "reward_mean": group["reward_mean"],
                    "reward_std": group["reward_std"],
                    "flags": ", ".join(flags) if flags else "none",
                    "href": href,
                },
            )
        self._write_index()
        metrics["audit/pages_written"] = float(len(ranked_groups))
        return metrics
