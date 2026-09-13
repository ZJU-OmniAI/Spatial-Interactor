from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


EASYR1_DIR = Path(__file__).resolve().parents[1] / "EasyR1"
sys.path.insert(0, str(EASYR1_DIR))

from verl.protocol import DataProto  # noqa: E402
from verl.trainer.opd import (  # noqa: E402
    build_cot_token_mask,
    build_paired_scoring_batch,
    build_privileged_scoring_batch,
    compute_topk_fkl,
    opd_weight_at_step,
    prepare_opd_batch,
)
from verl.workers.actor.config import ActorConfig  # noqa: E402
from verl.workers.actor.dp_actor import DataParallelPPOActor  # noqa: E402


class DummyOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyActorModule(nn.Module):
    def forward(self, input_ids, **kwargs):
        del kwargs
        vocab = 16
        token_axis = torch.arange(vocab, dtype=torch.float32, device=input_ids.device)
        logits = -torch.abs(token_axis.view(1, 1, -1) - input_ids.unsqueeze(-1).float())
        return DummyOutput(logits)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "<COT>": [10, 11],
            "</COT>": [12, 13],
            "COT:": [40, 41],
            "Answer:": [42, 43],
        }[text]


class OPDCoreTest(unittest.TestCase):
    def setUp(self):
        self.responses = torch.tensor(
            [
                [10, 11, 20, 21, 12, 13, 30, 0],
                [10, 11, 22, 23, 24, 12, 13, 0],
                [20, 21, 30, 0, 0, 0, 0, 0],
            ]
        )
        self.response_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
            ]
        )

    def test_strict_cot_mask(self):
        mask = build_cot_token_mask(
            self.responses, self.response_mask, DummyTokenizer()
        )
        self.assertEqual(mask[0].tolist(), [0, 0, 1, 1, 0, 0, 0, 0])
        self.assertEqual(mask[1].tolist(), [0, 0, 1, 1, 1, 0, 0, 0])
        self.assertEqual(mask[2].sum().item(), 0)

    def test_complete_legacy_cot_mask_excludes_answer(self):
        responses = torch.tensor([[40, 41, 20, 21, 42, 43, 30, 0]])
        response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
        mask = build_cot_token_mask(
            responses,
            response_mask,
            DummyTokenizer(),
            allow_legacy_cot=True,
        )
        self.assertEqual(mask[0].tolist(), [0, 0, 1, 1, 0, 0, 0, 0])

    def test_reasoning_prefill_masks_from_response_start_to_answer(self):
        responses = torch.tensor([[20, 21, 22, 42, 43, 30, 31, 0]])
        response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
        mask = build_cot_token_mask(
            responses,
            response_mask,
            DummyTokenizer(),
            response_starts_in_cot=True,
        )
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])

    def test_privileged_batch_keeps_response(self):
        bsz, prompt_len = 3, 5
        tensors = {
            "privileged_input_ids": torch.ones((bsz, prompt_len), dtype=torch.long),
            "privileged_attention_mask": torch.ones((bsz, prompt_len), dtype=torch.long),
            "privileged_position_ids": torch.arange(prompt_len).repeat(bsz, 1),
            "responses": self.responses,
            "response_mask": self.response_mask,
        }
        data = DataProto.from_dict(tensors=tensors)
        privileged = build_privileged_scoring_batch(data)
        self.assertTrue(torch.equal(privileged.batch["responses"], self.responses))
        self.assertEqual(privileged.batch["input_ids"].shape, (bsz, prompt_len + 8))
        self.assertEqual(privileged.batch["position_ids"][0, -1].item(), prompt_len + 7)

    def test_paired_batch_scores_plain_and_privileged_in_one_rpc(self):
        bsz, prompt_len = 3, 5
        prompt_ids = torch.full((bsz, prompt_len), 2, dtype=torch.long)
        privileged_ids = torch.full((bsz, prompt_len), 3, dtype=torch.long)
        prompt_positions = torch.arange(prompt_len).repeat(bsz, 1)
        tensors = {
            "prompts": prompt_ids,
            "input_ids": torch.cat((prompt_ids, self.responses), dim=-1),
            "attention_mask": torch.cat((torch.ones_like(prompt_ids), self.response_mask), dim=-1),
            "position_ids": torch.cat(
                (prompt_positions, torch.arange(prompt_len, prompt_len + 8).repeat(bsz, 1)), dim=-1
            ),
            "privileged_input_ids": privileged_ids,
            "privileged_attention_mask": torch.ones_like(privileged_ids),
            "privileged_position_ids": prompt_positions,
            "responses": self.responses,
            "response_mask": self.response_mask,
        }
        paired, plain_size = build_paired_scoring_batch(DataProto.from_dict(tensors=tensors))
        self.assertEqual(plain_size, bsz)
        self.assertEqual(len(paired), 2 * bsz)
        self.assertTrue(torch.equal(paired.batch["prompts"][:bsz], prompt_ids))
        self.assertTrue(torch.equal(paired.batch["prompts"][bsz:], privileged_ids))
        self.assertTrue(torch.equal(paired.batch["responses"][:bsz], paired.batch["responses"][bsz:]))

    def test_prepare_batch_routes_opd_only_to_cot(self):
        shape = self.responses.shape
        tensors = {
            "responses": self.responses,
            "response_mask": self.response_mask,
            "old_log_probs": torch.zeros(shape),
            "privileged_log_probs": torch.full(shape, 0.5),
            "advantages": torch.ones(shape),
            "returns": torch.ones(shape),
            "opd_teacher_fkl": torch.full(shape, 0.25),
        }
        data = DataProto.from_dict(tensors=tensors)
        data, metrics = prepare_opd_batch(
            data=data,
            tokenizer=DummyTokenizer(),
            loss_weight=0.05,
            token_kl_clip=0.05,
            cot_open_tag="<COT>",
            cot_close_tag="</COT>",
        )
        self.assertEqual(data.batch["advantages"].tolist(), torch.ones(shape).tolist())
        self.assertEqual(data.batch["opd_cot_mask"][0].tolist(), [0, 0, 1, 1, 0, 0, 0, 0])
        self.assertEqual(data.batch["opd_teacher_fkl"][0, 2].item(), 0.25)
        self.assertEqual(data.batch["opd_teacher_fkl"][0, 6].item(), 0.0)
        self.assertEqual(data.meta_info["opd_loss_weight"], 0.05)
        self.assertAlmostEqual(metrics["opd/teacher_delta_mean"], 0.5)

    def test_topk_fkl_is_zero_for_identical_distributions(self):
        teacher = torch.log_softmax(torch.tensor([[[3.0, 2.0, 1.0]]]), dim=-1)
        result = compute_topk_fkl(teacher, teacher)
        self.assertAlmostEqual(result.item(), 0.0, places=7)

    def test_topk_fkl_optional_clip_is_applied_after_nonnegative_sum(self):
        teacher = torch.log_softmax(torch.tensor([[[3.0, 2.0, 1.0]]]), dim=-1)
        student = torch.log_softmax(torch.tensor([[[1.0, 2.0, 3.0]]]), dim=-1)
        result = compute_topk_fkl(teacher, student, token_clip=0.05)
        expected = (teacher.exp() * (teacher - student)).sum(dim=-1).clamp(min=0.0, max=0.05)
        self.assertTrue(torch.allclose(result, expected))

    def test_topk_fkl_cannot_become_negative_after_clipping(self):
        teacher = torch.tensor(
            [[[-2.3353553, -1.2728080, -8.1836729, -0.4733684]]]
        )
        student = torch.tensor(
            [[[-5.5670610, -0.0739274, -7.5135884, -2.7046528]]]
        )
        raw = compute_topk_fkl(teacher, student)
        clipped = compute_topk_fkl(teacher, student, token_clip=0.05)
        old_unsafe = (teacher.exp() * (teacher - student)).clamp(max=0.05).sum(dim=-1)
        self.assertGreater(raw.item(), 0.0)
        self.assertLess(old_unsafe.item(), 0.0)
        self.assertGreaterEqual(clipped.item(), 0.0)
        self.assertAlmostEqual(clipped.item(), 0.05, places=7)

    def test_topk_fkl_is_nonnegative_for_random_distributions(self):
        generator = torch.Generator().manual_seed(20260719)
        teacher = torch.log_softmax(torch.randn(128, 16, generator=generator), dim=-1)
        student = torch.log_softmax(torch.randn(128, 16, generator=generator), dim=-1)
        raw = compute_topk_fkl(teacher, student)
        clipped = compute_topk_fkl(teacher, student, token_clip=0.05)
        self.assertTrue(torch.all(raw >= 0))
        self.assertTrue(torch.all(clipped >= 0))
        self.assertTrue(torch.all(clipped <= 0.05))

    def test_opd_weight_schedule(self):
        self.assertEqual(opd_weight_at_step(1, 0.05, 10, 90), 0.05)
        self.assertEqual(opd_weight_at_step(10, 0.05, 10, 90), 0.05)
        self.assertAlmostEqual(opd_weight_at_step(55, 0.05, 10, 90), 0.025)
        self.assertEqual(opd_weight_at_step(100, 0.05, 10, 90), 0.0)

    def test_actor_teacher_topk_and_student_support_shapes(self):
        config = ActorConfig(
            padding_free=False,
            dynamic_batching=False,
            use_torch_compile=False,
            micro_batch_size_per_device_for_experience=1,
        )
        actor = DataParallelPPOActor(config=config, actor_module=DummyActorModule())
        actor.log_probs_from_logits = lambda logits, labels: torch.gather(
            torch.log_softmax(logits, dim=-1),
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "position_ids": torch.arange(5).view(1, -1),
                "responses": torch.tensor([[4, 5]]),
            },
            meta_info={"temperature": 0.8},
        )
        sampled, topk_ids, teacher_logps = actor.compute_teacher_topk(
            batch,
            top_k=4,
            distill_temperature=1.0,
            special_token_ids=[0],
        )
        batch.batch["opd_teacher_topk_ids"] = topk_ids
        sampled_again, student_logps = actor.compute_log_prob_on_support(batch, distill_temperature=1.0)
        self.assertEqual(sampled.shape, (1, 2))
        self.assertEqual(topk_ids.shape, (1, 2, 4))
        self.assertEqual(teacher_logps.shape, (1, 2, 4))
        self.assertTrue(torch.equal(sampled, sampled_again))
        self.assertTrue(torch.allclose(teacher_logps.exp().sum(-1), torch.ones((1, 2))))
        self.assertTrue(torch.allclose(teacher_logps, student_logps))


if __name__ == "__main__":
    unittest.main()
