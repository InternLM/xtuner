"""Tests for the PPO critic value loss."""

from unittest import TestCase

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.rl.loss import (
    ValueLossConfig,
    ValueLossContext,
    explained_variance,
    value_loss,
)


class TestValueLossFunction(TestCase):
    def test_mse_is_half_squared_error(self) -> None:
        values = torch.tensor([[1.0, 2.0]])
        returns = torch.tensor([[0.0, 0.0]])
        weights = torch.ones_like(values)

        loss = value_loss(values, returns, weights, loss_type="mse")

        # 0.5 * (1^2 + 2^2)
        self.assertAlmostEqual(loss.item(), 2.5, places=6)

    def test_weights_scale_the_loss(self) -> None:
        values = torch.tensor([[2.0, 2.0]])
        returns = torch.zeros_like(values)
        # Zero weight must exclude a token entirely.
        weights = torch.tensor([[1.0, 0.0]])

        loss = value_loss(values, returns, weights, loss_type="mse")

        self.assertAlmostEqual(loss.item(), 2.0, places=6)

    def test_clipped_equals_mse_inside_the_trust_region(self) -> None:
        # |values - old_values| = 0.1 < value_clip, so clipping is inactive and
        # the clipped error equals the unclipped one.
        values = torch.tensor([[1.1]])
        old_values = torch.tensor([[1.0]])
        returns = torch.tensor([[0.0]])
        weights = torch.ones_like(values)

        clipped = value_loss(values, returns, weights, loss_type="clipped", old_values=old_values, value_clip=0.2)
        mse = value_loss(values, returns, weights, loss_type="mse")

        torch.testing.assert_close(clipped, mse)

    def test_clipped_penalizes_moving_outside_the_trust_region(self) -> None:
        # The prediction jumped 1.0 from old_values while the target is further
        # still, so the clipped branch yields the larger error and is selected.
        values = torch.tensor([[2.0]])
        old_values = torch.tensor([[1.0]])
        returns = torch.tensor([[5.0]])
        weights = torch.ones_like(values)

        loss = value_loss(values, returns, weights, loss_type="clipped", old_values=old_values, value_clip=0.2)

        # clipped prediction = 1.0 + clamp(1.0, -0.2, 0.2) = 1.2
        # max((2-5)^2, (1.2-5)^2) = max(9, 14.44) = 14.44
        self.assertAlmostEqual(loss.item(), 0.5 * 14.44, places=5)

    def test_clipped_never_below_mse(self) -> None:
        """The max() makes clipped loss a pessimistic bound on the MSE."""
        torch.manual_seed(0)
        values = torch.randn(1, 64)
        old_values = values + torch.randn(1, 64)
        returns = torch.randn(1, 64)
        weights = torch.ones_like(values)

        clipped = value_loss(values, returns, weights, loss_type="clipped", old_values=old_values, value_clip=0.2)
        mse = value_loss(values, returns, weights, loss_type="mse")

        self.assertGreaterEqual(clipped.item() + 1e-6, mse.item())

    def test_clipped_requires_old_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "old_values is required"):
            value_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), loss_type="clipped")

    def test_gradient_flows_to_values_only(self) -> None:
        values = torch.tensor([[1.0, 2.0]], requires_grad=True)
        returns = torch.tensor([[0.5, 0.5]], requires_grad=True)

        value_loss(values, returns, torch.ones(1, 2), loss_type="mse").backward()

        self.assertIsNotNone(values.grad)
        # Targets are frozen; a gradient path into them would train the critic
        # towards its own predictions.
        self.assertIsNone(returns.grad)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            value_loss(torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 3), loss_type="mse")

    def test_negative_clip_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_clip must be non-negative"):
            value_loss(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.ones(1, 2),
                loss_type="clipped",
                old_values=torch.zeros(1, 2),
                value_clip=-0.1,
            )


class TestValueLossConfig(TestCase):
    def test_build_requires_returns_and_mask(self) -> None:
        cfg = ValueLossConfig(loss_type="mse")
        self.assertIsNone(cfg.build(data={}))
        self.assertIsNone(cfg.build(data={"returns": torch.zeros(1, 2)}))

    def test_build_produces_context(self) -> None:
        cfg = ValueLossConfig(loss_type="mse")
        ctx = cfg.build(data={"returns": torch.zeros(1, 4), "value_mask": torch.ones(1, 4, dtype=torch.bool)})
        self.assertIsInstance(ctx, ValueLossContext)

    def test_clipped_build_requires_old_values(self) -> None:
        cfg = ValueLossConfig(loss_type="clipped")
        with self.assertRaisesRegex(ValueError, "old_values is required"):
            cfg.build(data={"returns": torch.zeros(1, 4), "value_mask": torch.ones(1, 4, dtype=torch.bool)})

    def test_chunk_mode_is_rejected(self) -> None:
        # Chunking exists to avoid materializing [tokens, vocab] logits, which
        # a scalar head never does.
        with self.assertRaisesRegex(ValueError, "only supports mode='eager'"):
            ValueLossConfig(mode="chunk", chunk_size=128)

    def test_negative_value_clip_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "value_clip must be non-negative"):
            ValueLossConfig(value_clip=-1.0)


class TestValueLossContext(TestCase):
    @staticmethod
    def _context(returns: torch.Tensor, mask: torch.Tensor) -> ValueLossContext:
        cfg = ValueLossConfig(loss_type="mse")
        ctx = cfg.build(data={"returns": returns, "value_mask": mask})
        assert ctx is not None
        return ctx

    def test_build_batches_normalizes_by_valid_token_count(self) -> None:
        mask = torch.tensor([[True, True, False, False]])
        ctx = self._context(torch.zeros(1, 4), mask)

        ValueLossContext.build_batches([ctx])

        weight = ctx.loss_kwargs.loss_weight
        assert weight is not None
        # Two valid tokens -> weight 1/2 on them, 0 elsewhere.
        torch.testing.assert_close(weight, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))

    def test_build_batches_shares_one_denominator_across_micro_batches(self) -> None:
        """Gradient accumulation must not change the effective loss scale."""
        first = self._context(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool))
        second = self._context(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool))

        ValueLossContext.build_batches([first, second])

        # 8 valid tokens across both micro-batches, so every weight is 1/8.
        for ctx in (first, second):
            weight = ctx.loss_kwargs.loss_weight
            assert weight is not None
            torch.testing.assert_close(weight, torch.full((1, 4), 0.125))

    def test_build_batches_handles_all_masked(self) -> None:
        ctx = self._context(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
        ValueLossContext.build_batches([ctx])
        weight = ctx.loss_kwargs.loss_weight
        assert weight is not None
        # Must not divide by zero.
        self.assertTrue(bool(torch.isfinite(weight).all()))
        self.assertEqual(float(weight.sum()), 0.0)

    def test_build_batches_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ValueLossContext.build_batches([])

    def test_forward_before_build_batches_raises(self) -> None:
        ctx = self._context(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool))
        with self.assertRaisesRegex(RuntimeError, "build_batches must be called"):
            ctx.loss_fn(torch.randn(1, 4, 8), torch.randn(1, 8), None, ctx.loss_kwargs)

    def test_loss_fn_averages_over_valid_tokens(self) -> None:
        """A calibrated loss equals the mean over valid tokens."""
        returns = torch.zeros(1, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        ctx = self._context(returns, mask)
        ValueLossContext.build_batches([ctx])

        # Force every predicted value to exactly 1.0.
        hidden_states = torch.ones(1, 4, 2)
        head_weight = torch.tensor([[0.5, 0.5]])

        loss, (values, _) = ctx.loss_fn(hidden_states, head_weight, None, ctx.loss_kwargs)

        torch.testing.assert_close(values, torch.ones(1, 4))
        # Mean of 0.5 * (1 - 0)^2 over 4 tokens.
        self.assertAlmostEqual(loss.item(), 0.5, places=6)

    def test_non_scalar_head_raises(self) -> None:
        ctx = self._context(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool))
        ValueLossContext.build_batches([ctx])
        # A vocabulary-shaped head would silently broadcast; catch it instead.
        with self.assertRaisesRegex(ValueError, "one scalar per token"):
            ctx.loss_fn(torch.randn(1, 4, 2), torch.randn(3, 2), None, ctx.loss_kwargs)

    def test_metrics_report_reducible_sums(self) -> None:
        ctx = self._context(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool))
        ValueLossContext.build_batches([ctx])

        _, (_, metrics) = ctx.loss_fn(torch.ones(1, 4, 2), torch.tensor([[0.5, 0.5]]), None, ctx.loss_kwargs)

        self.assertEqual(float(metrics["reduced_critic_valid_count"]), 4.0)
        self.assertAlmostEqual(float(metrics["reduced_critic_value_sum"]), 4.0, places=5)
        self.assertAlmostEqual(float(metrics["reduced_critic_error_square_sum"]), 4.0, places=5)


class TestExplainedVariance(TestCase):
    def test_perfect_prediction_is_one(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ev = explained_variance(
            return_square_sum=float(returns.square().sum()),
            return_sum=float(returns.sum()),
            error_square_sum=0.0,
            count=float(returns.numel()),
        )
        assert ev is not None
        self.assertAlmostEqual(ev, 1.0, places=6)

    def test_predicting_the_mean_is_zero(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        errors = returns - returns.mean()
        ev = explained_variance(
            return_square_sum=float(returns.square().sum()),
            return_sum=float(returns.sum()),
            error_square_sum=float(errors.square().sum()),
            count=float(returns.numel()),
        )
        assert ev is not None
        self.assertAlmostEqual(ev, 0.0, places=6)

    def test_worse_than_the_mean_is_negative(self) -> None:
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ev = explained_variance(
            return_square_sum=float(returns.square().sum()),
            return_sum=float(returns.sum()),
            error_square_sum=1000.0,
            count=float(returns.numel()),
        )
        assert ev is not None
        self.assertLess(ev, 0.0)

    def test_constant_returns_yield_none(self) -> None:
        # Zero return variance makes the ratio undefined rather than infinite.
        self.assertIsNone(explained_variance(return_square_sum=4.0, return_sum=4.0, error_square_sum=0.0, count=4.0))

    def test_too_few_samples_yield_none(self) -> None:
        self.assertIsNone(explained_variance(return_square_sum=1.0, return_sum=1.0, error_square_sum=0.0, count=1.0))


class TestValueLossDistributed(DistributedTestBase):
    """Verify the loss is calibrated across ranks and accumulation steps."""

    @property
    def world_size(self) -> int:
        return 2

    def test_loss_matches_single_rank_equivalent(self) -> None:
        """Two ranks x grad_acc must equal one rank over the concatenated data.

        This is the contract in `xtuner.v1.loss.base_loss_ctx`: the calibrated
        loss is invariant to how the batch is split over ranks and accumulation
        steps.
        """
        self.create_pg("cpu")
        try:
            grad_acc = 2
            tokens = 4
            # Rank-dependent but deterministic data.
            torch.manual_seed(0)
            all_returns = torch.arange(self.world_size * grad_acc * tokens, dtype=torch.float32).reshape(
                self.world_size, grad_acc, 1, tokens
            )
            all_values = all_returns * 0.5 + 1.0

            cfg = ValueLossConfig(loss_type="mse")
            contexts = []
            for step in range(grad_acc):
                ctx = cfg.build(
                    data={
                        "returns": all_returns[self.rank, step],
                        "value_mask": torch.ones(1, tokens, dtype=torch.bool),
                    }
                )
                assert ctx is not None
                contexts.append(ctx)
            ValueLossContext.build_batches(contexts)

            # Every rank must see the same global denominator.
            total_tokens = self.world_size * grad_acc * tokens
            for ctx in contexts:
                weight = ctx.loss_kwargs.loss_weight
                assert weight is not None
                torch.testing.assert_close(weight, torch.full((1, tokens), 1.0 / total_tokens))

            # Sum this rank's accumulated loss, then reduce over ranks.
            local_loss = torch.zeros(())
            for step, ctx in enumerate(contexts):
                local_loss = local_loss + value_loss(
                    all_values[self.rank, step],
                    ctx.loss_kwargs.returns,
                    ctx.loss_kwargs.loss_weight,
                    loss_type="mse",
                )
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

            # Reference: one rank, no accumulation, over all the data.
            flat_values = all_values.reshape(1, -1)
            flat_returns = all_returns.reshape(1, -1)
            expected = value_loss(
                flat_values,
                flat_returns,
                torch.full_like(flat_values, 1.0 / total_tokens),
                loss_type="mse",
            )

            torch.testing.assert_close(local_loss, expected, atol=1e-5, rtol=1e-5)
        finally:
            dist.destroy_process_group()

    def test_masked_tokens_excluded_from_global_count(self) -> None:
        """Ranks with differing valid-token counts share one denominator."""
        self.create_pg("cpu")
        try:
            # Rank 0 has 3 valid tokens, rank 1 has 1.
            valid = 3 if self.rank == 0 else 1
            mask = torch.zeros(1, 4, dtype=torch.bool)
            mask[0, :valid] = True

            cfg = ValueLossConfig(loss_type="mse")
            ctx = cfg.build(data={"returns": torch.zeros(1, 4), "value_mask": mask})
            assert ctx is not None
            ValueLossContext.build_batches([ctx])

            self.assertEqual(float(ctx.loss_kwargs.global_valid_count), 4.0)
            weight = ctx.loss_kwargs.loss_weight
            assert weight is not None
            self.assertAlmostEqual(float(weight.sum()), valid / 4.0, places=6)
        finally:
            dist.destroy_process_group()


class TestValueLossSequenceParallel(DistributedTestBase):
    """Sequence parallelism splits tokens; the loss must be unchanged."""

    @property
    def world_size(self) -> int:
        return 2

    def test_sp_split_preserves_total_loss(self) -> None:
        self.create_pg("cpu")
        try:
            from torch.distributed.device_mesh import init_device_mesh

            sp_mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("sp",))["sp"]

            tokens = 8
            returns = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens)
            values = returns * 0.5 + 1.0
            mask = torch.ones(1, tokens, dtype=torch.bool)

            cfg = ValueLossConfig(loss_type="mse")
            ctx = cfg.build(data={"returns": returns.clone(), "value_mask": mask.clone()}, sp_mesh=sp_mesh)
            assert ctx is not None
            # Each rank holds half the sequence.
            self.assertEqual(ctx.loss_kwargs.returns.shape, (1, tokens // self.world_size))

            ValueLossContext.build_batches([ctx])
            shard = tokens // self.world_size
            local_values = values[:, self.rank * shard : (self.rank + 1) * shard]
            local_loss = value_loss(
                local_values, ctx.loss_kwargs.returns, ctx.loss_kwargs.loss_weight, loss_type="mse"
            )
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

            expected = value_loss(values, returns, torch.full_like(values, 1.0 / tokens), loss_type="mse")
            torch.testing.assert_close(local_loss, expected, atol=1e-5, rtol=1e-5)
        finally:
            dist.destroy_process_group()
