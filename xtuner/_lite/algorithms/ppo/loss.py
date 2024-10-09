import torch
from torch.nn import functional as F

from xtuner._lite import get_logger

logger = get_logger()


def gather_logprobs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


@torch.no_grad()
def compute_rewards(logprobs, ref_logprobs, reward_score):

    kl_div_estimate = (logprobs - ref_logprobs)

    rewards = kl_div_estimate + reward_score

    return rewards


@torch.no_grad()
def compute_advantages_and_returns(values, rewards):

    gamma = 0.1
    lam = 0.1
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]

    for t in reversed(range(0, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


class CriticLoss(torch.nn.Module):
    """Loss function for critic model."""

    def __init__(self,
                 cliprange_value: float = 0.5,
                 loss_type: str = 'per_seq'):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.loss_type = loss_type

        assert self.loss_type in ['per_token', 'per_seq']

    def critic_loss_fn(self, values, old_values, returns, loss_factor=None):
        values_clipped = old_values + (values - old_values).clamp(
            -self.cliprange_value, self.cliprange_value)
        vf_loss1 = (values_clipped - returns)**2
        vf_loss2 = (values - returns)**2
        if self.loss_type == 'per_seq':
            vf_loss = torch.max(vf_loss1, vf_loss2).mean(-1)
        elif self.loss_type == 'per_token':
            assert loss_factor is not None
            vf_loss = torch.sum(torch.max(vf_loss1, vf_loss2) * loss_factor)
        return 0.5 * vf_loss

    def forward(self,
                values: torch.Tensor,
                old_values,
                returns,
                loss_factor=None):
        assert values.ndim == 2

        loss = self.critic_loss_fn(
            values=values,
            old_values=old_values,
            returns=returns,
            loss_factor=loss_factor)
        return loss


class PPOPolicyLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, cliprange: float = 0.2, loss_type: str = 'per_seq'):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type in ['per_token', 'per_seq']

    def forward(self, logprobs, old_logprobs, advantages, loss_factor=None):
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange,
                                1 + self.cliprange) * advantages
        if self.loss_type == 'per_seq':
            pg_loss = torch.max(pg_loss1, pg_loss2).mean(dim=-1)
        elif self.loss_type == 'per_token':
            assert loss_factor is not None
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2)) * loss_factor
        return pg_loss
