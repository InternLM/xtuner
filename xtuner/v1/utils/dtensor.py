from typing import cast

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard


def is_evenly_distributed(dtensor: DTensor) -> bool:
    """Check if a DTensor is evenly distributed across the device mesh."""
    global_shape = dtensor.shape

    mesh = dtensor.device_mesh
    placements = dtensor.placements

    tensor_dim_to_mesh_dims: dict[int, list[int]] = {}

    for dim_idx, placement in enumerate(placements):
        if hasattr(placement, "is_shard") and placement.is_shard():
            mesh_dim = cast(Shard, placement).dim
            if dim_idx not in tensor_dim_to_mesh_dims:
                tensor_dim_to_mesh_dims[dim_idx] = []
            tensor_dim_to_mesh_dims[dim_idx].append(mesh_dim)

    for tensor_dim, mesh_dims in tensor_dim_to_mesh_dims.items():
        total_devices = 1
        for mesh_dim in mesh_dims:
            total_devices *= mesh.size(mesh_dim)

        if global_shape[tensor_dim] % total_devices != 0:
            return False

    return True
