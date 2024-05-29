from typing import Any

import torch


class CriticLoss(torch.nn.Module):
    """Loss function for critic model."""

    def __init__(self,
                 cliprange_value: float = 100,
                 loss_type: str = 'per_seq'):
        super().__init__()
        self.cliprange_value = cliprange_value
        self.loss_type = loss_type
        assert self.loss_type in ['per_token', 'per_seq']

    def critic_loss_fn(self, values, old_values, returns, mask, loss_factor):
        values_clipped = old_values + (values - old_values).clamp(
            -self.cliprange_value, self.cliprange_value)
        vf_loss1 = (values_clipped - returns)**2
        vf_loss2 = (values - returns)**2

        if self.loss_type == 'per_seq':
            vf_loss = (torch.max(vf_loss1, vf_loss2) * mask).sum() / mask.sum()
        elif self.loss_type == 'per_token':
            vf_loss = torch.sum(
                torch.max(vf_loss1, vf_loss2) * mask * loss_factor)
        else:
            raise RuntimeError(
                f"CriticLoss.loss_type must be ['per_seq', 'per_token'], got {self.loss_type}"  # noqa: E501
            )
        return 0.5 * vf_loss.mean()

    def forward(self, values: torch.Tensor, labels: dict[str, Any]):
        """Forward function of CriticLoss.

        Args:
            values (Tensor): Forward result of the model. Its shape may be varied.  # noqa: E501
                For packed forward: (micro_bsz * seqlen, 1), where micro_bsz = 1  # noqa: E501
                For non packed forward: (micro_bsz, seqlen, 1)

            labels (Tuple[dict]): Label values which are split by pipeline
                schedule into pieces. The length of the list is micro_bsz. Each
                element is a dict, representing labels to a batch.

        Note:
            The parameter `labels` seems strange because of pj-colossalai's
            pipeline schedule mechanism. Labels are delivered to colosslai.Engine  # noqa: E501
            in List format, so pipeline schedule split it into micro_bsz pieces,  # noqa: E501
            and deliver them to loss_fn by `*args`.

        Returns:
            Tensor: Return the final loss
        """
        assert values.ndim == 2
        mask = labels['mask']  # (micro_bsz, seqlen)
        num_actions = mask.size(1)
        values = values[:, -num_actions:]

        old_values = labels['old_values']  # (micro_bsz, seqlen)
        returns = labels['returns']  # (micro_bsz, seqlen)
        loss_factor = labels['loss_factor']
        loss = self.critic_loss_fn(
            values=values,
            old_values=old_values,
            returns=returns,
            mask=mask,
            loss_factor=loss_factor,
        )
        return loss
