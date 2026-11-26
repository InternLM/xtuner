"""Actor snapshot binding on the RL loss context (S3/S4 seam)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from xtuner.v1.rl.loss import GRPOLossConfig


class TestBindActorSnapshot(unittest.TestCase):
    def test_binds_frozen_stage_outputs_without_splitting(self):
        labels = torch.tensor([[-100, 2, 3]])
        ctx = GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}).build(
            data={"shifted_labels": labels, "advantages": torch.zeros_like(labels, dtype=torch.float32)},
            device="cpu",
        )
        assert ctx is not None
        snapshot = SimpleNamespace(
            shifted_labels=torch.tensor([[-100, -100, 3]]),
            old_logprobs=torch.tensor([[-0.1, -0.2, -0.3]]),
            is_weights=torch.tensor([[1.0, 1.0, 1.5]]),
            ref_logprobs=None,
            teacher_logprobs=None,
            target_token_ids=None,
        )

        ctx.bind_actor_snapshot(snapshot)

        torch.testing.assert_close(ctx.loss_kwargs.shifted_labels, snapshot.shifted_labels)
        torch.testing.assert_close(ctx.loss_kwargs.old_logprobs, snapshot.old_logprobs)
        torch.testing.assert_close(ctx.loss_kwargs.is_weights, snapshot.is_weights)

    def test_rejects_snapshot_with_different_local_layout(self):
        labels = torch.tensor([[-100, 2, 3]])
        ctx = GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}).build(
            data={"shifted_labels": labels, "advantages": torch.zeros_like(labels, dtype=torch.float32)},
            device="cpu",
        )
        assert ctx is not None
        snapshot = SimpleNamespace(
            shifted_labels=torch.tensor([[-100, 3]]),
            old_logprobs=torch.tensor([[-0.1, -0.2]]),
            is_weights=None,
            ref_logprobs=None,
            teacher_logprobs=None,
            target_token_ids=None,
        )

        with self.assertRaisesRegex(ValueError, "different local layout"):
            ctx.bind_actor_snapshot(snapshot)
