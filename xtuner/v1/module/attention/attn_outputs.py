from typing import TypedDict

import torch


class AttnOutputs(TypedDict, total=False):
    projected_output: torch.Tensor
    raw_output: torch.Tensor
    softmax_lse: torch.Tensor | None
    attn_logits: torch.Tensor | None
