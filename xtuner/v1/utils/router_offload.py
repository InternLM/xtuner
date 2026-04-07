from dataclasses import dataclass

import torch


@dataclass
class AsyncOffloadedTensor:
    """Holds a CPU copy scheduled from device memory.

    Args:
        tensor_cpu (torch.Tensor): The pinned CPU tensor that receives device data.
        event (torch.cuda.Event | None): Completion event for the async D2H copy.
    """

    tensor_cpu: torch.Tensor
    event: torch.cuda.Event | None


def async_offload_to_cpu(tensor: torch.Tensor, stream: torch.cuda.Stream) -> AsyncOffloadedTensor:
    """Detach and asynchronously copy a device tensor to pinned CPU memory.

    Args:
        tensor (torch.Tensor): Source tensor on device or CPU.
        stream (torch.cuda.Stream): Stream used for the D2H copy.

    Returns:
        AsyncOffloadedTensor: CPU tensor plus the completion event.
    """

    detached_tensor = tensor.detach()
    if detached_tensor.device.type == "cpu":
        return AsyncOffloadedTensor(tensor_cpu=detached_tensor, event=None)

    tensor_cpu = torch.empty(detached_tensor.shape, dtype=detached_tensor.dtype, device="cpu", pin_memory=True)
    event = torch.cuda.Event()
    current_stream = torch.cuda.current_stream()
    is_slice_tensor = detached_tensor.storage().size() != detached_tensor.numel()

    with torch.no_grad():
        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
            if is_slice_tensor:
                tensor_cpu.copy_(detached_tensor, non_blocking=True)
            else:
                tensor_cpu.storage().copy_(detached_tensor.storage(), non_blocking=True)
            detached_tensor.record_stream(stream)
            event.record(stream)

    return AsyncOffloadedTensor(tensor_cpu=tensor_cpu, event=event)


def wait_async_offload(offloaded_tensor: AsyncOffloadedTensor) -> torch.Tensor:
    """Wait for an async D2H copy and return the CPU tensor.

    Args:
        offloaded_tensor (AsyncOffloadedTensor): Tensor and event pair created by async_offload_to_cpu.

    Returns:
        torch.Tensor: The completed CPU tensor.
    """

    if offloaded_tensor.event is not None:
        offloaded_tensor.event.synchronize()
    return offloaded_tensor.tensor_cpu
