from typing import Any

import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel


def sp_split(
    tensor,
    sp_mesh: DeviceMesh,
    split_dim: int,
    padding_value: Any,
):
    tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
    tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
    return tensor


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs
