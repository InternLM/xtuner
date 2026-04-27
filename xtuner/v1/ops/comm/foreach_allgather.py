import torch
import torch.distributed as dist


# mypy: allow-untyped-decorators
@torch.no_grad()
def foreach_all_gather(
    params: list[torch.Tensor],
    group: dist.ProcessGroup | None,
) -> list[list[torch.Tensor]]:
    """Perform a fused all-gather on a list of tensors.

    All ranks must contribute tensors with identical numels and shapes.
    """
    if group is None:
        group = dist.group.WORLD

    param0 = params[0]
    assert all(param.dtype == param0.dtype for param in params)

    input_tensor_numels = [param.numel() for param in params]
    input_tensor_shapes = [param.shape for param in params]
    world_size = dist.get_world_size(group)
    local_tensor_size = sum(input_tensor_numels)
    global_tensor_size = local_tensor_size * world_size

    # prepare flatten tensor
    flatten_copyin_tensor = torch.empty((local_tensor_size,), dtype=param0.dtype, device=param0.device)
    splits_copyin_tensor = torch.split(flatten_copyin_tensor, input_tensor_numels)
    torch._foreach_copy_(splits_copyin_tensor, [p.flatten() for p in params])
    flatten_copyout_tensor = torch.empty((global_tensor_size,), dtype=param0.dtype, device=param0.device)

    # allgather global flatten tensor
    dist.all_gather_into_tensor(flatten_copyout_tensor, flatten_copyin_tensor, group=group)
    copyout_split_size: list[int] = input_tensor_numels * world_size
    splits_copyout_tensor = torch.split(flatten_copyout_tensor, copyout_split_size)
    global_input_tensor_shapes = input_tensor_shapes * world_size

    # gathered_params: [[params1/p, params1/p,...], [params2/p, params2/p,...], ...]
    gathered_params: list[list[torch.Tensor]] = []
    for i in range(len(params)):
        single_gathered_params: list[torch.Tensor] = []
        for rank in range(dist.get_world_size(group)):
            offset = len(params) * rank
            origin_shape: tuple = global_input_tensor_shapes[offset + i]
            single_gathered_params.append(splits_copyout_tensor[offset + i].view(origin_shape))
        gathered_params.append(single_gathered_params)

    return gathered_params
