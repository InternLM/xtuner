from typing import cast

import torch
import torch.distributed as dist


# mypy: allow-untyped-decorators
@torch.no_grad()
def foreach_all_gather(
    params: list[torch.Tensor],
    group: dist.ProcessGroup | None,
) -> list[list[torch.Tensor]]:
    if group is None:
        group = dist.group.WORLD

    param0 = params[0]
    assert all(param.dtype == param0.dtype for param in params)

    input_tensor_numels = [param.numel() for param in params]
    input_tensor_shapes = [param.shape for param in params]

    flatten_copyin_tensor = torch.empty((sum(input_tensor_numels),), dtype=param0.dtype, device=param0.device)
    splits_copyin_tensor = torch.split(flatten_copyin_tensor, input_tensor_numels)
    torch._foreach_copy_(splits_copyin_tensor, [p.flatten() for p in params])

    input_tensor_numels_tensor = torch.tensor(input_tensor_numels, dtype=torch.int64, device=param0.device)
    global_input_tensor_numels = [
        torch.zeros_like(input_tensor_numels_tensor) for _ in range(dist.get_world_size(group))
    ]

    dist.all_gather(global_input_tensor_numels, input_tensor_numels_tensor, group=group)
    copyout_size = int(sum(sum(i) for i in global_input_tensor_numels))
    flatten_copyout_tensor = torch.empty((copyout_size,), dtype=param0.dtype, device=param0.device)

    dist.all_gather_into_tensor(flatten_copyout_tensor, flatten_copyin_tensor, group=group)
    copyout_split_size: list[int] = sum([i.tolist() for i in global_input_tensor_numels], [])
    splits_copyout_tensor = torch.split(flatten_copyout_tensor, copyout_split_size)

    _global_input_tensor_shapes: list[None] | list[list[tuple]] = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(_global_input_tensor_shapes, input_tensor_shapes, group=group)
    _global_input_tensor_shapes = cast(list[list[tuple]], _global_input_tensor_shapes)
    global_input_tensor_shapes: list[tuple] = sum(_global_input_tensor_shapes, [])

    gathered_params: list[list[torch.Tensor]] = []
    for i in range(len(params)):
        single_gathered_params: list[torch.Tensor] = []
        for rank in range(dist.get_world_size(group)):
            offset = len(params) * rank
            origin_shape: tuple = global_input_tensor_shapes[offset + i]
            single_gathered_params.append(splits_copyout_tensor[offset + i].view(origin_shape))
        gathered_params.append(single_gathered_params)

    return gathered_params
