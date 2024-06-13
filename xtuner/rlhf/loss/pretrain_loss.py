import torch
from loguru import logger


class PretrainLoss(torch.nn.Module):
    """Loss function for flash GPT Language Model."""

    def __init__(self, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None and label_smoothing != 0:
            logger.warning(f'Use label_smoothing: {label_smoothing}')
        self.label_smoothing = label_smoothing

        # Here, the output will gather output is set in the model, so use ordinary loss  # noqa: E501
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
        # There is no need to consider the ignore_index problem here, because the loss calculation will be  # noqa: E501
        # calculated through the calculation range, and -100 must be outside this range, so there is no problem  # noqa: E501

        return loss

