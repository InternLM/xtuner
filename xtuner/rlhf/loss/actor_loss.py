from typing import Any

import torch

from ..policy_output import logprobs_from_logits


class ActorLoss(torch.nn.Module):
    """Loss function for actor model."""

    def __init__(self, cliprange: float = 0.2, loss_type: str = 'per_seq'):
        super().__init__()
        self.cliprange = cliprange
        self.loss_type = loss_type
        assert self.loss_type in ['per_token', 'per_seq']

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask,
                      loss_factor):
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange,
                                1 + self.cliprange) * advantages
        if self.loss_type == 'per_seq':
            pg_loss = (torch.max(pg_loss1, pg_loss2) * mask).sum() / mask.sum()
        elif self.loss_type == 'per_token':
            pg_loss = torch.sum(
                torch.max(pg_loss1, pg_loss2) * mask) * loss_factor
        else:
            raise RuntimeError(
                f"ActorLoss.loss_type must be ['per_seq', 'per_token'], got {self.loss_type}"  # noqa: E501
            )
        return pg_loss.mean()

    def forward(self, logits: torch.Tensor, labels: dict[str, Any]):
        """Forward function of ActorLoss.

        Args:
            logits (Tensor): Forward result of the model. Its shape may be varied.  # noqa: E501
                For packed forward: (micro_bsz * seqlen, 1), where micro_bsz = 1  # noqa: E501
                For non packed forward: (micro_bsz, seqlen, 1)

            labels (tuple[dict]): Label values which are split by pipeline
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
        assert logits.ndim == 3
        mask = labels['mask']  # (micro_bsz, seqlen)

        assert logits.shape[0] == labels['input_ids'].shape[0]
        input_ids = labels['input_ids']  # (micro_bsz, seqlen)
        old_logprobs = labels['old_logprobs']  # (micro_bsz, seqlen)
        advantages = labels['advantages']  # (micro_bsz, seqlen)
        loss_factor = labels['loss_factor']

        logpy = logprobs_from_logits(
            logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        num_actions = mask.size(1)
        logprobs = logpy[:, -num_actions:]

        loss = self.actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            loss_factor=loss_factor,
        )
        return loss
