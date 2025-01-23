from typing import Any

import torch
from loguru import logger

from ..policy_output import logprobs_from_logits


class PretrainLoss(torch.nn.Module):
    """Loss function for flash GPT Language Model."""

    def __init__(self, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None and label_smoothing != 0:
            logger.warning(f'Use label_smoothing: {label_smoothing}')
        self.label_smoothing = label_smoothing

        # the output will gather output is set in the model,
        # so use ordinary loss
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction='mean', label_smoothing=label_smoothing)

    def forward(self, *args):
        if len(args) == 3:
            # residual is to match prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(
                f'The number of criterion inputs are:{len(args)}')
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)
        loss = self.loss_fn(shift_logits, shift_labels)
        # There is no need to consider the ignore_index problem here,
        # because the loss calculation will be calculated through the calculation range,  # noqa: E501
        # and -100 must be outside this range,
        # so there is no problem

        return loss


class PPOPolicyLoss(torch.nn.Module):
    """Loss function for policy model."""

    def __init__(self, cliprange: float = 0.2):
        super().__init__()
        self.cliprange = cliprange

    def policy_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ratio = (logprobs - old_logprobs).exp()
        pg_loss1 = -ratio * advantages
        pg_loss2 = -ratio.clamp(1 - self.cliprange,
                                1 + self.cliprange) * advantages
        pg_loss = (torch.max(pg_loss1, pg_loss2) * mask).sum() / mask.sum()
        return pg_loss.mean()

    def forward(self, logits: torch.Tensor, labels: dict[str, Any]):
        assert logits.ndim == 3
        mask = labels['mask']

        assert logits.shape[0] == labels['input_ids'].shape[0]
        input_ids = labels['input_ids']
        old_logprobs = labels['old_logprobs']
        advantages = labels['advantages']

        logpy = logprobs_from_logits(
            logits=logits[:, :-1, :], labels=input_ids[:, 1:], gather=True)
        num_actions = mask.size(1)
        logprobs = logpy[:, -num_actions:]

        loss = self.policy_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask)
        return loss
