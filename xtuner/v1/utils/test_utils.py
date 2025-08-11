import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


# from liger-kernel
def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5):
    """Assert that two tensors are element-wise equal within a tolerance,
    providing detailed information about mismatches.

    Parameters:
    tensor1 (torch.Tensor): First tensor to compare.
    tensor2 (torch.Tensor): Second tensor to compare.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        raise AssertionError("Input tensors must have the same shape.")

    # Calculate the difference between the tensors
    diff = torch.abs(tensor1 - tensor2)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(tensor2)

    # Find tolerance mismatched elements
    tol_mismatched = diff > tolerance

    # Find nan mismatched elements
    nan_mismatched = torch.logical_xor(torch.isnan(tensor1), torch.isnan(tensor2))

    # Find +inf mismatched elements
    posinf_mismatched = torch.logical_xor(torch.isposinf(tensor1), torch.isposinf(tensor2))
    # Find -inf mismatched elements
    neginf_mismatched = torch.logical_xor(torch.isneginf(tensor1), torch.isneginf(tensor2))

    # Find all mismatched elements
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Check if all elements are close
    all_close = num_mismatched == 0

    # Raise AssertionError with detailed information if there are mismatches
    if not all_close and num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        print_count = min(max_print, num_mismatched)
        for index in mismatched_indices[:print_count]:
            i = tuple(index.tolist())
            mismatch_details.append(f"Mismatch at index {i}: tensor1[{i}] = {tensor1[i]}, tensor2[{i}] = {tensor2[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched elements.")

        raise AssertionError("\n".join(mismatch_details))


def init_data_mesh(device, sp_size):
    world_size = dist.get_world_size()
    dp_size = world_size // sp_size
    data_mesh = init_device_mesh(
        device,
        (dp_size, sp_size),
        mesh_dim_names=("dp", "sp"),
    )
    return data_mesh
