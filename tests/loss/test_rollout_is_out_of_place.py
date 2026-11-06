"""Rollout importance-sampling masks must not mutate the full source labels."""

from __future__ import annotations

import unittest

import torch

from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling


class TestRolloutISOutOfPlace(unittest.TestCase):
    def test_masked_tokens_do_not_mutate_source_labels(self):
        labels = torch.tensor([[2, 3, 4]])
        source = labels.clone()
        ctx = GRPOLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            rollout_is=RolloutImportanceSampling(
                rollout_is_mode="mask",
                rollout_is_threshold=(2.0, 0.5),
                rollout_is_mask_threshold=(2.0, 0.5),
            ),
        ).build(
            data={"shifted_labels": labels, "advantages": torch.zeros_like(labels, dtype=torch.float32)},
            device="cpu",
        )
        assert ctx is not None
        # Token 0 has ratio exp(1 - (-1)) > 2 and must be rejected; token 1 stays.
        ctx.loss_kwargs.old_logprobs = torch.tensor([[1.0, 0.0, 0.0]])
        ctx.loss_kwargs.rollout_logprobs = torch.tensor([[-1.0, 0.0, 0.0]])

        ctx.compute_rollout_is(None, torch.tensor([3]))  # type: ignore[arg-type]

        torch.testing.assert_close(labels, source)
        self.assertEqual(ctx.loss_kwargs.shifted_labels.tolist(), [[-100, 3, 4]])
        self.assertIsNotNone(ctx.loss_kwargs.is_weights)
