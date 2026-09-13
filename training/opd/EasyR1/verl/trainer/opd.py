from __future__ import annotations

from typing import Any, Optional

import torch

from ..protocol import DataProto


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    stop = len(sequence) - len(pattern) + 1
    for index in range(start, max(start, stop)):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def build_cot_token_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    open_tag: str = "<COT>",
    close_tag: str = "</COT>",
    allow_legacy_cot: bool = False,
    response_starts_in_cot: bool = False,
) -> torch.Tensor:
    """Mask reasoning enclosed by a complete tagged or legacy delimiter pair."""
    open_ids = tokenizer.encode(open_tag, add_special_tokens=False)
    close_ids = tokenizer.encode(close_tag, add_special_tokens=False)
    legacy_open_ids = tokenizer.encode("COT:", add_special_tokens=False) if allow_legacy_cot else []
    legacy_close_ids = (
        tokenizer.encode("Answer:", add_special_tokens=False)
        if allow_legacy_cot or response_starts_in_cot
        else []
    )
    cot_mask = torch.zeros_like(response_mask)

    for row_index in range(responses.shape[0]):
        valid_length = int(response_mask[row_index].sum().item())
        token_ids = responses[row_index, :valid_length].tolist()
        open_at = _find_subsequence(token_ids, open_ids)
        if open_at < 0:
            open_at = _find_subsequence(token_ids, legacy_open_ids)
            if open_at < 0:
                if response_starts_in_cot:
                    close_at = _find_subsequence(token_ids, legacy_close_ids)
                    if close_at > 0:
                        cot_mask[row_index, :close_at] = 1
                continue
            cot_start = open_at + len(legacy_open_ids)
            close_at = _find_subsequence(token_ids, legacy_close_ids, cot_start)
            if close_at <= cot_start:
                continue
            cot_mask[row_index, cot_start:close_at] = 1
            continue
        cot_start = open_at + len(open_ids)
        close_at = _find_subsequence(token_ids, close_ids, cot_start)
        if close_at <= cot_start:
            continue
        cot_mask[row_index, cot_start:close_at] = 1

    return cot_mask * response_mask


def build_privileged_scoring_batch(data: DataProto) -> DataProto:
    """Replace the rollout prompt with its privileged counterpart, keeping responses fixed."""
    required = (
        "privileged_input_ids",
        "privileged_attention_mask",
        "privileged_position_ids",
        "responses",
        "response_mask",
    )
    missing = [key for key in required if key not in data.batch]
    if missing:
        raise KeyError(f"OPD privileged scoring batch is missing tensor keys: {missing}")

    prompts = data.batch["privileged_input_ids"]
    prompt_mask = data.batch["privileged_attention_mask"]
    prompt_position_ids = data.batch["privileged_position_ids"]
    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    response_length = responses.shape[-1]

    delta = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta = delta.view(1, -1).expand(prompts.shape[0], -1)
    if prompt_position_ids.ndim == 3:
        delta = delta.view(prompts.shape[0], 1, -1).expand(
            prompts.shape[0], prompt_position_ids.shape[1], -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta

    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": torch.cat((prompt_mask, response_mask), dim=-1),
        "response_mask": response_mask,
        "position_ids": torch.cat((prompt_position_ids, response_position_ids), dim=-1),
    }
    non_tensors = {}
    for key in ("uid", "multi_modal_data"):
        if key in data.non_tensor_batch:
            non_tensors[key] = data.non_tensor_batch[key]
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=dict(data.meta_info),
    )


def build_plain_scoring_batch(data: DataProto) -> DataProto:
    tensor_keys = (
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "response_mask",
        "position_ids",
    )
    tensors = {key: data.batch[key] for key in tensor_keys}
    non_tensors = {
        key: data.non_tensor_batch[key]
        for key in ("uid", "multi_modal_data")
        if key in data.non_tensor_batch
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=dict(data.meta_info),
    )


def build_paired_scoring_batch(data: DataProto) -> tuple[DataProto, int]:
    """Concatenate plain and privileged contexts for one old-actor RPC."""
    plain = build_plain_scoring_batch(data)
    privileged = build_privileged_scoring_batch(data)
    return DataProto.concat([plain, privileged]), len(plain)


def opd_weight_at_step(
    step: int,
    initial_weight: float,
    hold_steps: int,
    decay_steps: int,
) -> float:
    """Return the short-lived privileged-distillation weight for this step."""
    if initial_weight <= 0:
        return 0.0
    if step <= hold_steps:
        return float(initial_weight)
    if decay_steps <= 0 or step >= hold_steps + decay_steps:
        return 0.0
    progress = (step - hold_steps) / float(decay_steps)
    return float(initial_weight) * (1.0 - progress)


def compute_topk_fkl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    token_clip: Optional[float] = None,
) -> torch.Tensor:
    """Compute non-negative teacher-to-student KL on a shared top-k support."""
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError(
            "Teacher and student top-k log-probabilities must have the same shape, "
            f"got {teacher_log_probs.shape} and {student_log_probs.shape}."
        )
    teacher_log_probs = teacher_log_probs.float()
    student_log_probs = student_log_probs.float()
    token_kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    token_kl = token_kl.clamp_min(0.0)
    if token_clip is not None:
        clip = float(token_clip)
        if clip <= 0:
            raise ValueError("OPD token KL cap must be positive when configured.")
        token_kl = token_kl.clamp_max(clip)
    return token_kl


def prepare_opd_batch(
    data: DataProto,
    tokenizer: Any,
    loss_weight: float,
    token_kl_clip: Optional[float],
    cot_open_tag: str,
    cot_close_tag: str,
    allow_legacy_cot: bool = False,
    response_starts_in_cot: bool = False,
    require_nonempty_cot: bool = False,
) -> tuple[DataProto, dict[str, float]]:
    """Build the CoT routing mask and attach detached top-k OPD targets."""
    old_log_probs = data.batch["old_log_probs"]
    response_mask = data.batch["response_mask"]
    tagged_cot_mask = build_cot_token_mask(
        responses=data.batch["responses"],
        response_mask=response_mask,
        tokenizer=tokenizer,
        open_tag=cot_open_tag,
        close_tag=cot_close_tag,
    ).to(dtype=old_log_probs.dtype)
    delimited_cot_mask = (
        build_cot_token_mask(
            responses=data.batch["responses"],
            response_mask=response_mask,
            tokenizer=tokenizer,
            open_tag=cot_open_tag,
            close_tag=cot_close_tag,
            allow_legacy_cot=True,
        ).to(dtype=old_log_probs.dtype)
        if allow_legacy_cot
        else tagged_cot_mask
    )
    cot_mask = (
        build_cot_token_mask(
            responses=data.batch["responses"],
            response_mask=response_mask,
            tokenizer=tokenizer,
            open_tag=cot_open_tag,
            close_tag=cot_close_tag,
            allow_legacy_cot=allow_legacy_cot,
            response_starts_in_cot=True,
        ).to(dtype=old_log_probs.dtype)
        if response_starts_in_cot
        else delimited_cot_mask
    )

    active_tokens = cot_mask.sum()
    if require_nonempty_cot and active_tokens.item() == 0:
        raise RuntimeError(
            f"OPD requires complete {cot_open_tag}...{cot_close_tag} spans, but this batch has none."
        )

    data.batch["opd_cot_mask"] = cot_mask
    data.meta_info["opd_loss_weight"] = float(loss_weight)
    data.meta_info["opd_token_kl_clip"] = token_kl_clip

    raw_delta = None
    if "privileged_log_probs" in data.batch:
        raw_delta = (data.batch["privileged_log_probs"] - old_log_probs).detach()
        data.batch["opd_teacher_delta"] = raw_delta * cot_mask
    if "opd_teacher_fkl" in data.batch:
        data.batch["opd_teacher_fkl"] = data.batch["opd_teacher_fkl"].detach() * cot_mask

    valid_response_tokens = response_mask.sum().clamp_min(1)
    completed_rows = (cot_mask.sum(dim=-1) > 0).float()
    tagged_rows = (tagged_cot_mask.sum(dim=-1) > 0).float()
    delimited_rows = (delimited_cot_mask.sum(dim=-1) > 0).float()
    metrics = {
        "opd/cot_row_fraction": completed_rows.mean().item(),
        "opd/tagged_cot_row_fraction": tagged_rows.mean().item(),
        "opd/legacy_cot_row_fraction": (delimited_rows - tagged_rows).clamp_min(0).mean().item(),
        "opd/prefilled_cot_row_fraction": (completed_rows - delimited_rows).clamp_min(0).mean().item(),
        "opd/cot_token_fraction": (active_tokens / valid_response_tokens).item(),
        "opd/loss_weight": float(loss_weight),
        "opd/teacher_topk_fkl_mean": 0.0,
        "opd/teacher_delta_mean": 0.0,
        "opd/teacher_delta_positive_fraction": 0.0,
    }
    active = cot_mask.bool()
    if active.any():
        if "opd_teacher_fkl" in data.batch:
            metrics["opd/teacher_topk_fkl_mean"] = data.batch["opd_teacher_fkl"][active].mean().item()
        if raw_delta is not None:
            metrics["opd/teacher_delta_mean"] = raw_delta[active].mean().item()
            metrics["opd/teacher_delta_positive_fraction"] = (raw_delta[active] > 0).float().mean().item()
    return data, metrics
