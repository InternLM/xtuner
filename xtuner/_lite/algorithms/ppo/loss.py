# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F

from xtuner._lite import get_logger

logger = get_logger()


def gather_logprobs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


@torch.no_grad()
def compute_kl_rewards(logprobs, ref_logprobs, reward_score, kl_coef=0.01):
    assert logprobs.ndim == 1
    last_mask = torch.zeros_like(logprobs, dtype=torch.int)
    last_mask[-1] = 1

    kl = ref_logprobs - logprobs
    kl_reward = kl_coef * kl * (1 - last_mask)

    last_reward = reward_score * last_mask

    rewards = kl_reward + last_reward

    return rewards


@torch.no_grad()
def compute_advantages_and_returns(values, rewards, gamma=1.0, gae_lambda=0.99):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134  # noqa: E501
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
    """
    lastgaelam = 0
    advantages_reversed = []

    assert values.numel() == rewards.numel(), f"{values.numel()}, {rewards.numel()}"
    length = rewards.numel()

    for t in reversed(range(0, length)):
        nextvalues = values[t + 1] if t < length - 1 else 0.0
        # Since old_rewards and old_values are masked with action_mask,
        # i.e. they have 0's at pad tokens,
        # delta will be 0 if current t is at a pad token,
        # so will lastgaelam
        delta = rewards[t] + gamma * nextvalues - values[t]
        lastgaelam = delta + gamma * gae_lambda * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=0)
    returns = advantages + values
    return advantages.detach(), returns


class CriticLoss(torch.nn.Module):
    """Loss function for critic model."""

    def __init__(self, cliprange_value: float = 0.5, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.loss_type = loss_type

        assert self.loss_type in ["per_token", "per_seq"]

    def critic_loss_fn(self, values, old_values, returns, loss_factor=None):
        values_clipped = old_values + (values - old_values).clamp(
            -self.cliprange_value, self.cliprange_value
        )
        vf_loss1 = (values_clipped - returns) ** 2
        vf_loss2 = (values - returns) ** 2
        if self.loss_type == "per_seq":
            vf_loss = torch.max(vf_loss1, vf_loss2).mean(-1)
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            vf_loss = torch.sum(torch.max(vf_loss1, vf_loss2) * loss_factor)
        return 0.5 * vf_loss

    def forward(self, values: torch.Tensor, old_values, returns, loss_factor=None):
        loss = self.critic_loss_fn(
            values=values,
            old_values=old_values,
            returns=returns,
            loss_factor=loss_factor,
        )
        return loss


class PPOPolicyLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, cliprange: float = 0.2, loss_type: str = "per_seq"):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type in ["per_token", "per_seq"]

    def forward(self, logprobs, old_logprobs, advantages, loss_factor=None):
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange, 1 + self.cliprange) * advantages
        if self.loss_type == "per_seq":
            pg_loss = torch.max(pg_loss1, pg_loss2).mean(dim=-1)
        elif self.loss_type == "per_token":
            assert loss_factor is not None
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2)) * loss_factor
        return pg_loss
