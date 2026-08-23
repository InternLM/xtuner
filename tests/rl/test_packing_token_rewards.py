"""Tests for packing per-token PPO tensors alongside labels and advantages."""

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.trainer.controller import TOKEN_TENSOR_KEYS, TrainingController


def _sample(seq_len: int, reward_at_last: float | None = None) -> dict:
    """One trajectory as `_packing` expects it, optionally with token rewards."""
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    sample = {
        "seq_ctx": SequenceContext.from_input_ids((input_ids,), device="cpu"),
        "shifted_labels": input_ids.clone(),
        "advantage": [0.0] * seq_len,
        "rollout_logprobs": None,
    }
    if reward_at_last is not None:
        token_rewards = torch.zeros(1, seq_len, dtype=torch.float32)
        token_rewards[0, -1] = reward_at_last
        sample["token_rewards"] = token_rewards
    return sample


class TestPackingTokenRewards:
    def test_token_rewards_are_packed_and_zero_padded(self) -> None:
        controller = TrainingController.__new__(TrainingController)
        pack_max_length = 64
        batches = [_sample(8, reward_at_last=1.5), _sample(4, reward_at_last=-2.0)]

        packed = controller._packing(batches, pack_max_length, language_cfg=None)

        assert len(packed) == 1
        token_rewards = packed[0]["token_rewards"]
        assert token_rewards.shape == (1, pack_max_length)
        # Each trajectory keeps its terminal reward at its own last real token.
        assert token_rewards[0, 7].item() == 1.5
        assert token_rewards[0, 11].item() == -2.0
        # Padding is zero, the neutral reward.
        assert bool((token_rewards[0, 12:] == 0).all())
        # Total reward mass is preserved by packing.
        torch.testing.assert_close(token_rewards.sum(), torch.tensor(-0.5))

    def test_token_rewards_align_with_labels(self) -> None:
        controller = TrainingController.__new__(TrainingController)
        batches = [_sample(6, reward_at_last=1.0), _sample(6, reward_at_last=1.0)]

        packed = controller._packing(batches, 32, language_cfg=None)

        labels = packed[0]["shifted_labels"]
        token_rewards = packed[0]["token_rewards"]
        assert token_rewards.shape == labels.shape
        # Rewards must land on real tokens, never on padding.
        assert bool((token_rewards[labels == -100] == 0).all())

    def test_absent_token_rewards_are_not_fabricated(self) -> None:
        """Group-baseline algorithms must see no PPO tensors."""
        controller = TrainingController.__new__(TrainingController)
        batches = [_sample(8), _sample(8)]

        packed = controller._packing(batches, 32, language_cfg=None)

        for key in TOKEN_TENSOR_KEYS:
            assert key not in packed[0]

    def test_multiple_packs_each_carry_token_rewards(self) -> None:
        controller = TrainingController.__new__(TrainingController)
        # pack_max_length forces two packs.
        batches = [_sample(8, reward_at_last=1.0) for _ in range(4)]

        packed = controller._packing(batches, 16, language_cfg=None)

        assert len(packed) == 2
        for pack in packed:
            assert pack["token_rewards"].shape == (1, 16)
            torch.testing.assert_close(pack["token_rewards"].sum(), torch.tensor(2.0))
