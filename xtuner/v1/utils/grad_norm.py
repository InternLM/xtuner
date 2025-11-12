from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _has_foreach_support,
)


def group_tensors_by_device_mesh_and_placements(tensors: List[DTensor]):
    grouped_tensors: Dict[Tuple[DeviceMesh, Tuple[Placement, ...]], List[DTensor]] = {}
    for tensor in tensors:
        assert isinstance(tensor, DTensor)
        key = (tensor.device_mesh, tensor.placements)
        if key in grouped_tensors:
            grouped_tensors[key].append(tensor)
        else:
            grouped_tensors[key] = [tensor]
    return grouped_tensors


def cal_total_norm(
    tensors: List[DTensor], norm_type: float = 2.0, foreach: Optional[bool] = None, dtype=torch.float32
):
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    device_mesh: DeviceMesh = tensors[
        0
    ].device_mesh  # For eg: DeviceMesh('cuda', [0, 1], mesh_dim_names=('default.fsdp',))
    placements = tensors[0].placements  # For eg: (Shard(dim=0),)
    device = tensors[0].device  # For eg: device(type='cuda', index=0)
    norms: Tuple[DTensor, ...]
    if (foreach is None and _has_foreach_support(tensors, device)) or (  # type: ignore
        foreach and _device_has_foreach_support(device)
    ):
        norms = torch._foreach_norm(tensors, norm_type, dtype=dtype)  # type: ignore
        # element of norms is dtensor with placement of _NormPartial
        # For example: norms[0] = DTensor(local_tensor=0.04525977373123169,
        #                                 device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('default.fsdp',)),
        #                                 placements=(_NormPartial(reduce_op='sum', norm_type=2.0),))
    elif foreach:
        raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
    else:
        norms = tuple(torch.linalg.vector_norm(g, norm_type, dtype=dtype) for g in tensors)

    local_norm = torch.linalg.vector_norm(torch.stack([norm.to_local() for norm in norms]), norm_type, dtype=dtype)
    if norm_type == 2:
        local_norm_squared = local_norm**2
        for i, placement in enumerate(placements):
            if isinstance(placement, Shard):
                # When using ep + fsdp, the placement corresponding to fsdp mesh is _StridedShard
                # isinstance(_StridedShard, Shard) is True
                dist.all_reduce(local_norm_squared, group=device_mesh.get_group(i))
            elif isinstance(placement, Replicate):
                pass
            else:
                raise ValueError(f"Unsupported placement type {placement} in clip_grad_norm")
        global_norm = local_norm_squared**0.5
    else:
        raise NotImplementedError
    return global_norm


def cal_grad_norm(grads: List[DTensor], dtype=torch.float32):
    grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    # print(f"clip_grad_norm dtype: {dtype}")
    total_norms = []
    for grads in grouped_grads.values():
        total_norm = cal_total_norm(grads, norm_type=2.0, foreach=True, dtype=dtype)
        total_norms.append(total_norm)
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
    grad_norm = grad_norm.to(grads[0].dtype)
    return grad_norm, grouped_grads
