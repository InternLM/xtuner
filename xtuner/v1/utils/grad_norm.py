from typing import List

import torch
from torch.distributed.tensor import DTensor

from .dtensor import cal_total_norm, group_tensors_by_device_mesh_and_placements


def cal_grad_norm(grads: List[DTensor], dtype=torch.float32):
    grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    total_norms = []
    for grads in grouped_grads.values():
        total_norm = cal_total_norm(grads, norm_type=2.0, foreach=True, dtype=dtype)
        total_norms.append(total_norm)
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
    grad_norm = grad_norm.to(grads[0].dtype)
    return grad_norm, grouped_grads
