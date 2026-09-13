# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...trainer.opd import compute_topk_fkl
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        top_k: int = 0,
        distill_temperature: float = 1.0,
        special_token_ids: Optional[list[int]] = None,
        support_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if top_k > 0 and support_token_ids is not None:
            raise ValueError("Request either teacher top-k or student support scoring, not both.")
        if (top_k > 0 or support_token_ids is not None) and self.config.ulysses_size > 1:
            raise NotImplementedError("Top-k OPD currently requires actor.ulysses_size=1.")

        auxiliary = None
        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(
                logits=logits_rmpad / temperature,
                labels=input_ids_rmpad_rolled,
            )

            if top_k > 0:
                teacher_logits = logits_rmpad / distill_temperature
                if special_token_ids:
                    teacher_logits = teacher_logits.clone()
                    teacher_logits[:, special_token_ids] = -torch.inf
                topk_logits, topk_ids = torch.topk(teacher_logits, k=top_k, dim=-1)
                topk_log_probs = F.log_softmax(topk_logits.float(), dim=-1).to(logits_rmpad.dtype)
                full_topk_ids = pad_input(topk_ids, indices=indices, batch=batch_size, seqlen=seqlen)
                full_topk_log_probs = pad_input(
                    topk_log_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                auxiliary = (
                    full_topk_ids[:, -response_length - 1 : -1],
                    full_topk_log_probs[:, -response_length - 1 : -1],
                )
            elif support_token_ids is not None:
                support_size = support_token_ids.size(-1)
                support_canvas = torch.zeros(
                    (batch_size, seqlen, support_size),
                    dtype=support_token_ids.dtype,
                    device=support_token_ids.device,
                )
                support_canvas[:, -response_length - 1 : -1] = support_token_ids
                support_rmpad = index_first_axis(
                    rearrange(support_canvas, "b s k -> (b s) k"),
                    indices,
                )
                selected_logits = torch.gather(logits_rmpad, dim=-1, index=support_rmpad.long())
                selected_log_probs = F.log_softmax(
                    selected_logits.float() / distill_temperature,
                    dim=-1,
                ).to(logits_rmpad.dtype)
                full_selected_log_probs = pad_input(
                    selected_log_probs,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                auxiliary = full_selected_log_probs[:, -response_length - 1 : -1]

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits / temperature, responses)  # (bsz, response_length)
            if top_k > 0:
                teacher_logits = logits / distill_temperature
                if special_token_ids:
                    teacher_logits = teacher_logits.clone()
                    teacher_logits[..., special_token_ids] = -torch.inf
                topk_logits, topk_ids = torch.topk(teacher_logits, k=top_k, dim=-1)
                topk_log_probs = F.log_softmax(topk_logits.float(), dim=-1).to(logits.dtype)
                auxiliary = (topk_ids, topk_log_probs)
            elif support_token_ids is not None:
                selected_logits = torch.gather(logits, dim=-1, index=support_token_ids.long())
                auxiliary = F.log_softmax(
                    selected_logits.float() / distill_temperature,
                    dim=-1,
                ).to(logits.dtype)

        if top_k > 0:
            topk_ids, topk_log_probs = auxiliary
            return log_probs, topk_ids, topk_log_probs
        if support_token_ids is not None:
            return log_probs, auxiliary
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @torch.no_grad()
    def compute_teacher_topk(
        self,
        data: DataProto,
        top_k: int,
        distill_temperature: float,
        special_token_ids: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score sampled tokens and return detached teacher top-k local distributions."""
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]
        data = data.select(
            ["input_ids", "attention_mask", "position_ids", "responses"],
            ["multi_modal_inputs"],
        )
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        sampled_parts, id_parts, logp_parts = [], [], []
        for micro_batch in micro_batches:
            sampled, ids, logps = self._forward_micro_batch(
                {**micro_batch.batch, **micro_batch.non_tensor_batch},
                temperature=temperature,
                top_k=top_k,
                distill_temperature=distill_temperature,
                special_token_ids=special_token_ids,
            )
            sampled_parts.append(sampled)
            id_parts.append(ids)
            logp_parts.append(logps)

        sampled = torch.concat(sampled_parts, dim=0)
        topk_ids = torch.concat(id_parts, dim=0)
        topk_log_probs = torch.concat(logp_parts, dim=0)
        if self.config.dynamic_batching:
            sampled = restore_dynamic_batch(sampled, batch_idx_list)
            topk_ids = restore_dynamic_batch(topk_ids, batch_idx_list)
            topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)
        return sampled, topk_ids, topk_log_probs

    @torch.no_grad()
    def compute_log_prob_on_support(
        self,
        data: DataProto,
        distill_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the old student on the teacher-supported token set."""
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]
        data = data.select(
            ["input_ids", "attention_mask", "position_ids", "responses", "opd_teacher_topk_ids"],
            ["multi_modal_inputs"],
        )
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        sampled_parts, support_parts = [], []
        for micro_batch in micro_batches:
            sampled, support_logps = self._forward_micro_batch(
                {**micro_batch.batch, **micro_batch.non_tensor_batch},
                temperature=temperature,
                distill_temperature=distill_temperature,
                support_token_ids=micro_batch.batch["opd_teacher_topk_ids"],
            )
            sampled_parts.append(sampled)
            support_parts.append(support_logps)

        sampled = torch.concat(sampled_parts, dim=0)
        support_log_probs = torch.concat(support_parts, dim=0)
        if self.config.dynamic_batching:
            sampled = restore_dynamic_batch(sampled, batch_idx_list)
            support_log_probs = restore_dynamic_batch(support_log_probs, batch_idx_list)
        return sampled, support_log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        opd_loss_weight = float(data.meta_info.get("opd_loss_weight", 0.0))
        opd_active = opd_loss_weight > 0 and "opd_teacher_topk_ids" in data.batch
        if opd_active:
            select_keys.extend(["opd_teacher_topk_ids", "opd_teacher_topk_log_probs", "opd_cot_mask"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    if opd_active:
                        log_probs, student_topk_log_probs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            distill_temperature=float(data.meta_info.get("opd_temperature", 1.0)),
                            support_token_ids=model_inputs["opd_teacher_topk_ids"],
                        )
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    if opd_active:
                        cot_mask = model_inputs["opd_cot_mask"]
                        token_fkl = compute_topk_fkl(
                            teacher_log_probs=model_inputs["opd_teacher_topk_log_probs"],
                            student_log_probs=student_topk_log_probs,
                            token_clip=data.meta_info.get("opd_token_kl_clip"),
                        )
                        opd_loss = torch.sum(token_fkl * cot_mask) / torch.sum(response_mask).clamp_min(1)
                        loss = loss + opd_loss_weight * opd_loss
                        metrics["actor/opd_loss"] = opd_loss.detach().item()
                        metrics["actor/opd_weight"] = opd_loss_weight
                        metrics["actor/opd_weighted_loss"] = (opd_loss_weight * opd_loss).detach().item()

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
