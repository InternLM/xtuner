from typing import TypedDict

import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext


class RLTrainItem(TypedDict):
    """Worker-side trainable projection of one rollout sample.

    The training worker converts each rollout state into one ``RLTrainItem`` before
    packing. ``loss_inputs`` is a plain CPU data dictionary consumed directly by
    ``loss_cfg.build()``; it is not a runtime ``BaseRLLossContext``. The stable keys are
    ``shifted_labels``, ``advantages`` and ``rollout_logprobs``, plus the optional
    distillation keys ``teacher_logprobs``, ``target_token_ids`` and ``teacher_indices``.
    Every position-wise tensor must satisfy ``tensor.shape[1] ==
    seq_ctx.input_ids.shape[1]``.
    """

    seq_ctx: SequenceContext
    loss_inputs: dict[str, torch.Tensor | None]
