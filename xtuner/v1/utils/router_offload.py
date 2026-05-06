from typing import Any

import torch
from torch.utils._pytree import tree_map_only


# Names of view-like aten ops whose results share storage with their input. For these ops the
# wrapper can stay lazy (no sync) because no real data movement happened. Non-aten or unknown
# ops fall through to the safe path that triggers a wait.
_VIEW_OP_NAMES = frozenset(
    {
        "view",
        "_unsafe_view",
        "reshape",
        "unsqueeze",
        "squeeze",
        "transpose",
        "permute",
        "select",
        "slice",
        "detach",
        "alias",
        "expand",
        "as_strided",
        "t",
    }
)


def _is_view_op(func: Any) -> bool:
    """Check whether an aten op is view-like and can stay lazy.

    Matches the op base name (e.g. ``"view"`` for ``aten.view.default``) exactly against
    ``_VIEW_OP_NAMES``; substring matching would over-match (``"select"`` would match
    ``"select_backward"``, ``"t"`` would match every op whose name ends in ``t``).
    """
    packet = getattr(func, "overloadpacket", None)
    if packet is None:
        return False
    return getattr(packet, "__name__", "") in _VIEW_OP_NAMES


class AsyncOffloadedTensor(torch.Tensor):
    """Tensor wrapper that waits for async D2H completion on first real use.

    This follows the same high-level pattern as PyTorch's async collective tensor
    wrappers: callers can treat the object like a tensor, and synchronization is
    triggered lazily when the value is actually consumed.
    """

    elem: torch.Tensor
    event: torch.cuda.Event | None
    completed: bool

    __slots__ = ["elem", "event", "completed"]

    @staticmethod
    def __new__(
        cls,
        elem: torch.Tensor,
        event: torch.cuda.Event | None = None,
        *,
        completed: bool | None = None,
    ) -> "AsyncOffloadedTensor":
        wrapped = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        wrapped.elem = elem
        wrapped.event = event
        wrapped.completed = event is None if completed is None else completed
        return wrapped

    def __repr__(self) -> str:  # type: ignore[override]
        return f"AsyncOffloadedTensor({self.trigger_wait()})"

    def __tensor_flatten__(self):
        return ["elem"], self.completed

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        meta: Any,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "AsyncOffloadedTensor":
        del outer_size, outer_stride
        return AsyncOffloadedTensor(inner_tensors["elem"], completed=bool(meta))

    def trigger_wait(self) -> torch.Tensor:
        """Synchronize the D2H event at most once and return the ready
        tensor."""
        if not self.completed and self.event is not None:
            self.event.synchronize()
            self.completed = True
        return self.elem

    def wait(self) -> torch.Tensor:
        """Explicitly wait for completion and return the underlying tensor."""
        return self.trigger_wait()

    def tolist(self) -> list[Any]:
        return self.trigger_wait().tolist()

    def numpy(self):
        return self.trigger_wait().numpy()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        del types
        kwargs = kwargs or {}
        is_view_op = _is_view_op(func)
        first_wrapper: AsyncOffloadedTensor | None = None

        def unwrap(e: "AsyncOffloadedTensor") -> torch.Tensor:
            nonlocal first_wrapper
            if first_wrapper is None:
                first_wrapper = e
            if is_view_op:
                return e.elem
            return e.trigger_wait()

        def wrap(e: torch.Tensor) -> "AsyncOffloadedTensor":
            if isinstance(e, AsyncOffloadedTensor):
                raise AssertionError("Cannot wrap AsyncOffloadedTensor inside another AsyncOffloadedTensor")
            assert first_wrapper is not None
            return AsyncOffloadedTensor(e, first_wrapper.event, completed=first_wrapper.completed)

        unwrapped_args = tree_map_only(AsyncOffloadedTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncOffloadedTensor, unwrap, kwargs)
        out = func(*unwrapped_args, **unwrapped_kwargs)

        if is_view_op and first_wrapper is not None:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out


def async_offload_to_cpu(tensor: torch.Tensor, stream: torch.cuda.Stream) -> AsyncOffloadedTensor:
    """Detach and asynchronously copy a device tensor to pinned CPU memory.

    Args:
        tensor (torch.Tensor): Source tensor on device or CPU.
        stream (torch.cuda.Stream): Stream used for the D2H copy.

    Returns:
        AsyncOffloadedTensor: A tensor wrapper whose underlying value becomes
        ready once the async D2H event completes.
    """

    detached_tensor = tensor.detach()
    if detached_tensor.device.type == "cpu":
        return AsyncOffloadedTensor(detached_tensor, None)

    tensor_cpu = torch.empty(detached_tensor.shape, dtype=detached_tensor.dtype, device="cpu", pin_memory=True)
    event = torch.cuda.Event()
    current_stream = torch.cuda.current_stream()
    # Storage-level memcpy is faster than the strided element-wise path, but only correct when the
    # source tensor is contiguous and owns its full storage at offset 0; otherwise the raw storage
    # bytes do not correspond to a logical row-major layout and tensor_cpu would receive garbage.
    use_storage_copy = (
        detached_tensor.is_contiguous()
        and detached_tensor.storage_offset() == 0
        and detached_tensor.untyped_storage().nbytes() == detached_tensor.numel() * detached_tensor.element_size()
    )

    with torch.no_grad():
        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
            if use_storage_copy:
                tensor_cpu.untyped_storage().copy_(detached_tensor.untyped_storage(), non_blocking=True)
            else:
                tensor_cpu.copy_(detached_tensor, non_blocking=True)
            detached_tensor.record_stream(stream)
            event.record(stream)

    return AsyncOffloadedTensor(tensor_cpu, event)


def wait_async_offload(offloaded_tensor: torch.Tensor) -> torch.Tensor:
    """Wait for an async D2H copy and return the ready CPU tensor."""
    if isinstance(offloaded_tensor, AsyncOffloadedTensor):
        return offloaded_tensor.trigger_wait()
    return offloaded_tensor
