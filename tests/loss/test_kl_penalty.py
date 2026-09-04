"""Tests for the per-token KL estimator shared by loss- and reward-side KL."""

import pytest
import torch

from xtuner.v1.rl.loss import kl_divergence_per_token, kl_penalty


KL_TYPES = ["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"]


@pytest.fixture
def logprob_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logprobs = torch.randn(2, 8, dtype=torch.float32)
    ref_logprobs = torch.randn(2, 8, dtype=torch.float32)
    return logprobs, ref_logprobs


class TestKLDivergencePerToken:
    @pytest.mark.parametrize("kl_type", KL_TYPES)
    def test_shape_is_preserved(self, kl_type: str, logprob_pair) -> None:
        logprobs, ref_logprobs = logprob_pair
        kl = kl_divergence_per_token(logprobs, ref_logprobs, kl_type)
        assert kl.shape == logprobs.shape

    @pytest.mark.parametrize("kl_type", KL_TYPES)
    def test_matches_weighted_sum_of_kl_penalty(self, kl_type: str, logprob_pair) -> None:
        """``kl_penalty`` must stay exactly the weighted sum of the per-token KL.

        This pins the refactor: the loss-side KL path is unchanged while the
        reward-side path consumes the same unreduced estimate.
        """
        logprobs, ref_logprobs = logprob_pair
        weights = torch.rand_like(logprobs)

        per_token = kl_divergence_per_token(logprobs, ref_logprobs, kl_type)
        expected = (per_token * weights).sum()
        actual = kl_penalty(logprobs, ref_logprobs, weights, kl_type)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("kl_type", KL_TYPES)
    def test_identical_policies_give_zero_kl(self, kl_type: str, logprob_pair) -> None:
        logprobs, _ = logprob_pair
        kl = kl_divergence_per_token(logprobs, logprobs.clone(), kl_type)
        torch.testing.assert_close(kl, torch.zeros_like(kl))

    @pytest.mark.parametrize("kl_type", ["abs", "mse", "k2", "low_var_kl", "k3"])
    def test_symmetric_estimators_are_non_negative(self, kl_type: str, logprob_pair) -> None:
        logprobs, ref_logprobs = logprob_pair
        kl = kl_divergence_per_token(logprobs, ref_logprobs, kl_type)
        assert bool((kl >= 0).all())

    def test_k1_is_signed(self, logprob_pair) -> None:
        # k1 is an unbiased but signed estimator; callers rely on that sign.
        logprobs, ref_logprobs = logprob_pair
        kl = kl_divergence_per_token(logprobs, ref_logprobs, "k1")
        torch.testing.assert_close(kl, logprobs - ref_logprobs)

    def test_low_var_kl_is_clamped(self) -> None:
        # Extreme gaps must stay finite so a KL-in-reward term cannot blow up GAE.
        logprobs = torch.tensor([[-100.0, 100.0]])
        ref_logprobs = torch.tensor([[100.0, -100.0]])
        kl = kl_divergence_per_token(logprobs, ref_logprobs, "low_var_kl")
        assert bool(torch.isfinite(kl).all())
        assert bool((kl <= 10.0).all())

    def test_unknown_estimator_raises(self, logprob_pair) -> None:
        logprobs, ref_logprobs = logprob_pair
        with pytest.raises(NotImplementedError):
            kl_divergence_per_token(logprobs, ref_logprobs, "not_a_kl_type")
