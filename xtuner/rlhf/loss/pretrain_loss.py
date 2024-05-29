import torch
from loguru import logger

try:
    from flash_attn.losses.cross_entropy import \
        CrossEntropyLoss as FlashCrossEntropyLoss
except ImportError:
    pass


# Adapted from: https://gitlab.pjlab.org.cn/openmmlab/bigmodel/rl3m/-/blob/main/rl3m/layers/loss.py#L37  # noqa: E501
class FlashGPTLMLoss(torch.nn.Module):
    """Loss function for flash GPT Language Model."""

    def __init__(self, parallel_output=True, label_smoothing=0):
        super().__init__()

        if label_smoothing is not None and label_smoothing != 0:
            logger.warning(f'Use label_smoothing: {label_smoothing}')
        self.label_smoothing = label_smoothing

        if parallel_output:
            # The loss in this place is bound to the gather_output initialized by VocabParallelClassifier1D  # noqa: E501
            self.loss_fn = FlashCrossEntropyLoss(
                reduction='mean',
                inplace_backward=True,
                process_group=None,
                label_smoothing=label_smoothing,
            )
        else:
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


# Adapted from: https://gitlab.pjlab.org.cn/openmmlab/bigmodel/rl3m/-/blob/main/rl3m/layers/loss.py#L37  # noqa: E501
class PretrainLoss(FlashGPTLMLoss):
    """Modified from pretrain/sft loss, but with a loss factor term to balance
    with ppo policy loss."""

    def __init__(self, *args, loss_factor=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_factor = loss_factor

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        return loss * self.loss_factor
