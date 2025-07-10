# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/distributed_utils.py

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import DTensor

from xtuner.v1.float8.float8_tensor import Float8Tensor


def tensor_already_casted_to_fp8(tensor: torch.Tensor) -> bool:
    """Check if the tensor is already casted to fp8, works if the local tensor
    is wrapped in DTensor."""
    if isinstance(tensor, Float8Tensor):
        return True
    elif isinstance(tensor, DTensor):
        # TODO: shall we stick to public API and directly use tensor.to_local() here?
        return tensor_already_casted_to_fp8(tensor._local_tensor)
    elif isinstance(tensor, funcol.AsyncCollectiveTensor):
        return tensor_already_casted_to_fp8(tensor.elem)

    return False
