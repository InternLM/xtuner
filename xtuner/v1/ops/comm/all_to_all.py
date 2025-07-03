import torch
from torch.distributed._functional_collectives import all_to_all_single_autograd as _all_to_all_single_autograd
from torch.distributed.device_mesh import DeviceMesh


def ulysses_all_to_all(
    input: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    mesh: DeviceMesh,
) -> torch.Tensor:
    """Performs a collective all-to-all operation on a tensor across a device
    mesh.

    This function redistributes data from the scatter dimension to the gather dimension
    across all processes in the device mesh.

    Args:
        input (torch.Tensor): The input tensor to redistribute.
        scatter_dim (int): The dimension along which to scatter the input tensor.
        gather_dim (int): The dimension along which to gather the output tensor.
        mesh (DeviceMesh): The device mesh defining the process group for communication.

    Returns:
        torch.Tensor: The redistributed tensor after the all-to-all operation.

    Note:
        The scatter dimension must be evenly divisible by the group size.
    """
    world_size = mesh.size()
    split_size = input.size(scatter_dim) // world_size
    input_split_sizes = [split_size] * world_size
    output_split_sizes = input_split_sizes

    input = input.contiguous()
    # TODO: here ``movedim` is replaced by `transpose`, need to check if this is correct.
    input = input.transpose(scatter_dim, 0)

    output = _all_to_all_single_autograd(
        input,
        group=mesh.get_group(),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    output = output.transpose(0, scatter_dim)

    output_list = torch.tensor_split(output, world_size, scatter_dim)
    output = torch.cat(output_list, dim=gather_dim).contiguous()
    return output


all_to_all_single_autograd = _all_to_all_single_autograd
