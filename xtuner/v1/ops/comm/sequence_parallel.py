import torch
import torch.distributed as dist


def split_for_sequence_parallel(input, dim: int, sp_mesh):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    sp_group = sp_mesh.get_group()
    sp_size = sp_mesh.size()
    if sp_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % sp_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of "
        f"sp size ({sp_size}), cannot split tensor evenly"
    )

    tensor_list = torch.split(input, dim_size // sp_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output