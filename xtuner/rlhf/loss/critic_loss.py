from typing import Any

import torch


class CriticLoss(torch.nn.Module):
    """Loss function for critic model."""

    def __init__(self, cliprange_value: float = 0.5):
        super().__init__()
        self.cliprange_value = cliprange_value

    def critic_loss_fn(self, values, old_values, returns, mask):
        values_clipped = old_values + (values - old_values).clamp(
            -self.cliprange_value, self.cliprange_value)
        vf_loss1 = (values_clipped - returns)**2
        vf_loss2 = (values - returns)**2
        vf_loss = (torch.max(vf_loss1, vf_loss2) * mask).sum() / mask.sum()
        return 0.5 * vf_loss.mean()

    def forward(self, values: torch.Tensor, labels: dict[str, Any]):
        assert values.ndim == 2
        mask = labels['mask']
        num_actions = mask.size(1)
        values = values[:, -num_actions:]

        old_values = labels['old_values']
        returns = labels['returns']
        loss = self.critic_loss_fn(
            values=values, old_values=old_values, returns=returns, mask=mask)
        return loss
