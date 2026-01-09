import importlib
import os
import sys
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.distributed_c10d import ReduceOp, _resolve_process_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_gradient_divide_factors,
    _get_param_all_gather_inputs,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    compiled_autograd_enabled,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor

from xtuner.v1.ops.comm import AllGatherManager, ReduceScatterManager, SymmBufferManager
from xtuner.v1.utils.device import get_device, get_torch_device_module


is_comm_opt_available = True

try:
    import ib_wrapper
    from ib_wrapper import ibgdaAllgather, ibReduceScatter
except ImportError:
    ib_wrapper = None
    ibReduceScatter = None
    ibgdaAllgather = None
    is_comm_opt_available = False


ag_event: torch.Event | None = None
rs_event: torch.Event | None = None


USE_CUSTOM_AG = int(os.getenv("XTUNER_USE_CUSTOM_AG_IN_FSDP", 0)) == 1
USE_CUSTOM_RS = int(os.getenv("XTUNER_USE_CUSTOM_RS_IN_FSDP", 0)) == 1


NUM_AG_BUFFERS = 2 if USE_CUSTOM_AG else 0
NUM_RS_BUFFERS = 1 if USE_CUSTOM_RS else 0


SELECT_COMM_SM_IN_FSDP = int(os.getenv("XTUNER_SELECT_COMM_SM_IN_FSDP", 1)) == 1


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=NUM_AG_BUFFERS)
rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=NUM_RS_BUFFERS)

ag_manager = AllGatherManager(num_buffers=NUM_AG_BUFFERS, select_comm_sm=SELECT_COMM_SM_IN_FSDP)
rs_manager = ReduceScatterManager(num_buffers=NUM_RS_BUFFERS, select_comm_sm=SELECT_COMM_SM_IN_FSDP)


def allocate_memory(
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
    from_process_group: bool,
) -> torch.Tensor:
    if from_process_group:
        backend = group._get_backend(device)
        if backend.supports_tensor_alloc(device):  # type: ignore
            return backend.allocate_tensor(size, dtype=dtype, device=device)  # type: ignore
    return torch.empty((size,), dtype=dtype, device=device)


lib = torch.library.Library("fsdp", "FRAGMENT")  # noqa: TOR901


lib.define(
    """
    all_gather_copy_in_customed(
        Tensor[] all_gather_inputs,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt world_size,
        SymInt rank,
        ScalarType dtype,
        Device device,
        str group_name,
        bool allocate_memory_from_process_group
    ) -> (Tensor, Tensor)
    """
)


@torch.library.impl(lib, "all_gather_copy_in_customed", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
    allocate_memory_from_process_group: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = torch.empty((all_gather_input_numel * world_size,), dtype=dtype, device="meta")
    all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
    return all_gather_input, all_gather_output


@torch.library.impl(lib, "all_gather_copy_in_customed", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in_customed", "CPU")
def all_gather_copy_in_cuda(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    group_name: str,
    allocate_memory_from_process_group: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ---------- diff --------- #
    global ag_symm
    if (USE_CUSTOM_AG or USE_CUSTOM_RS) and world_size == dist.get_world_size():
        recv_bytes = all_gather_input_numel * world_size * all_gather_inputs[0].element_size()
        send_bytes = recv_bytes // world_size
        recv_bytes_aligned = (send_bytes + 127) // 128 * 128 * world_size
        symm_buf = ag_symm.get_buffer(bytes=recv_bytes_aligned, device=device)
        all_gather_output = symm_buf.view(dtype)[: all_gather_input_numel * world_size]
    else:
        all_gather_output = allocate_memory(
            all_gather_input_numel * world_size,
            dtype=dtype,
            device=device,
            group=_resolve_process_group(group_name),
            from_process_group=allocate_memory_from_process_group,
        )

    # ---------- end --------- #

    all_gather_input = all_gather_output.narrow(0, all_gather_input_numel * rank, all_gather_input_numel)
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.Stream,
    all_gather_stream: torch.Stream,
    device: torch.device,
    allocate_memory_from_process_group: bool = False,
) -> Optional[AllGatherResult]:
    world_size, rank = group.size(), group.rank()
    device_handle = _get_device_handle(device.type)

    with device_handle.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = _get_param_all_gather_inputs(fsdp_params)
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        if dtype == torch.uint8:
            all_gather_inputs = [t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts]
        else:
            all_gather_inputs = [t for ts in param_all_gather_inputs for t in ts]
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        send_bytes = all_gather_input_numel * all_gather_inputs[0].element_size()

        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in_customed(
            all_gather_inputs,
            inp_split_sizes,
            all_gather_input_numel,
            world_size,
            rank,
            dtype,
            device,
            group.group_name,
            allocate_memory_from_process_group,
        )

        del param_all_gather_inputs

    # ---------- diff --------- #
    # Use the global dictionary for caching ibgdaAllgather objects
    use_custom_ag = USE_CUSTOM_AG and group.size() == dist.get_world_size()
    global ag_manager
    if use_custom_ag:
        ag_manager.prepare_allgather_objects(
            send_bytes=send_bytes,
            process_group=group,
            all_gather_stream=all_gather_stream,
            barrier_all=True,
        )
    # Ensure all_gather_stream waits for copy operations to complete
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    global rs_event
    if rs_event is not None:
        all_gather_stream.wait_event(rs_event)
    # Execute operations on the all_gather_stream
    with device_handle.stream(all_gather_stream):
        if use_custom_ag:
            ag_manager.execute_allgather(
                send_bytes=send_bytes,
                all_gather_output=all_gather_output,
                all_gather_input=all_gather_input,
                process_group=group,
            )
            # Set work handle to None since we're using custom implementation
            all_gather_work = None
        else:
            all_gather_work = dist.all_gather_into_tensor(
                output_tensor=all_gather_output,
                input_tensor=all_gather_input,
                group=group,
                async_op=async_op,
            )

        # Record event for synchronization
        all_gather_event = all_gather_stream.record_event()
        global ag_event
        ag_event = all_gather_event

        # ---------- end --------- #

        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


@torch.no_grad()
def foreach_reduce_pt26(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    reduce_scatter_reduce_op: Optional[Union[dist.ReduceOp, dist.ReduceOp.RedOpType]],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
) -> Tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients."""
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}")
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    predivide_factor, postdivide_factor = _get_gradient_divide_factors(
        reduce_scatter_group, all_reduce_group, reduce_dtype
    )
    world_size = reduce_scatter_group.size()
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
            continue
        assert unsharded_grad.size(shard_dim) % world_size == 0, (
            f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
        )
        chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size

    # ---------- diff --------- #
    use_custom_rs = USE_CUSTOM_RS and reduce_scatter_group.size() == dist.get_world_size()
    # global symm_buf
    send_bytes = reduce_scatter_input_numel * unsharded_grads[0].element_size()
    recv_bytes = send_bytes // world_size
    recv_bytes_aligned = (recv_bytes + 127) // 128 * 128
    send_bytes_aligned = recv_bytes_aligned * world_size

    global rs_manager, rs_symm
    # Get reduce-scatter objects
    if use_custom_rs:
        rs_manager.prepare_reducescatter_objects(
            recv_bytes=recv_bytes,
            process_group=reduce_scatter_group,
            reduce_scatter_stream=reduce_scatter_stream,
            barrier_all=True,
        )

    if use_custom_rs:
        symm_buf = rs_symm.get_buffer(bytes=send_bytes_aligned, device=device)
        reduce_scatter_input = symm_buf.view(reduce_dtype)[:reduce_scatter_input_numel]
    else:
        reduce_scatter_input = torch.empty((reduce_scatter_input_numel,), dtype=reduce_dtype, device=device)
    # ---------- end --------- #

    device_handle = _get_device_handle(device.type)
    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None
    # ---------- diff --------- #
    global ag_event, rs_event
    if ag_event is not None:
        reduce_scatter_stream.wait_event(ag_event)
    # ---------- end --------- #
    with device_handle.stream(reduce_scatter_stream):
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if reduce_scatter_reduce_op is None:
            if predivide_factor is None:
                reduce_scatter_reduce_op = ReduceOp.AVG
            else:
                reduce_scatter_reduce_op = ReduceOp.SUM
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        # ---------- diff --------- #
        if use_custom_rs:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes=recv_bytes,
                reduce_scatter_output=reduce_output,
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_reduce_op=reduce_scatter_reduce_op,
            )
        else:
            dist.reduce_scatter_tensor(
                output=reduce_output,
                input=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_reduce_op,
            )
        # ---------- end --------- #
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        rs_event = reduce_scatter_event
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
        rs_event = post_reduce_event
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


@torch.no_grad()
def foreach_reduce_pt28(
    fsdp_params: List[FSDPParam],
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    reduce_dtype: Optional[torch.dtype],
    device: torch.device,
    gradient_divide_factor: Optional[float],
    all_reduce_group: Optional[dist.ProcessGroup],  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Optional[torch.Tensor],  # only used for HSDP
    all_reduce_hook: Optional[Callable[[torch.Tensor], None]],
    allocate_memory_from_process_group: bool = False,
    force_sum_reduction_for_comms: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Event,
    torch.Event,
    Optional[torch.Tensor],
    Optional[torch.Event],
    Optional[torch.Tensor],
]:
    """``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients."""
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}")
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = _get_gradient_divide_factors(  # type: ignore
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device.type,
        gradient_divide_factor,
        force_sum_reduction_for_comms,
    )
    world_size = reduce_scatter_group.size()
    for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads)):
        if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
            continue
        assert unsharded_grad.size(shard_dim) % world_size == 0, (
            f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
        )
        chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
        unsharded_grads[i] = torch.cat(chunks, dim=0)
    padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size

    # ---------- diff --------- #
    use_custom_rs = USE_CUSTOM_RS and reduce_scatter_group.size() == dist.get_world_size()
    # global symm_buf
    send_bytes = reduce_scatter_input_numel * unsharded_grads[0].element_size()
    recv_bytes = send_bytes // world_size
    recv_bytes_aligned = (recv_bytes + 127) // 128 * 128
    send_bytes_aligned = recv_bytes_aligned * world_size

    global rs_manager, rs_symm
    # Get reduce-scatter objects
    if use_custom_rs:
        rs_manager.prepare_reducescatter_objects(
            recv_bytes=recv_bytes,
            process_group=reduce_scatter_group,
            reduce_scatter_stream=reduce_scatter_stream,
            barrier_all=True,
        )

    if use_custom_rs:
        symm_buf = rs_symm.get_buffer(bytes=send_bytes_aligned, device=device)
        reduce_scatter_input = symm_buf.view(reduce_dtype)[:reduce_scatter_input_numel]
    else:
        reduce_scatter_input = allocate_memory(
            reduce_scatter_input_numel,
            dtype=reduce_dtype,
            device=device,
            group=reduce_scatter_group,
            from_process_group=allocate_memory_from_process_group,
        )
    # ---------- end --------- #

    device_handle = _get_device_handle(device.type)

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    current_stream = device_handle.current_stream()
    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)

    all_reduce_input = None
    all_reduce_event = None
    # ---------- diff --------- #
    global ag_event, rs_event
    if ag_event is not None:
        reduce_scatter_stream.wait_event(ag_event)
    # ---------- end --------- #

    with device_handle.stream(reduce_scatter_stream):
        _div_if_needed(reduce_scatter_input, predivide_factor)

        reduce_output = allocate_memory(
            reduce_scatter_output_numel,
            dtype=reduce_dtype,
            device=device,
            group=reduce_scatter_group,
            from_process_group=allocate_memory_from_process_group,
        )
        # ---------- diff --------- #
        if use_custom_rs:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes=recv_bytes,
                reduce_scatter_output=reduce_output,
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_reduce_op=reduce_scatter_op,
            )
        else:
            dist.reduce_scatter_tensor(
                output=reduce_output,
                input=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        # ---------- end --------- #

        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        rs_event = reduce_scatter_event
        if all_reduce_group is not None:  # HSDP
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) if not
                # accumulating gradients since the CPU add kernel depends on
                # the copy result and we cannot run the add as a callback
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            if to_accumulate_grad:
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            if not compiled_autograd_enabled():
                for hook in (getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                    hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
        rs_event = post_reduce_event
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def patch_fsdp_agrs() -> None:
    if not (USE_CUSTOM_AG or USE_CUSTOM_RS):
        return

    if (USE_CUSTOM_AG or USE_CUSTOM_RS) and not is_comm_opt_available:
        raise ImportError("XTUNER_USE_CUSTOM_{AG,RS}_IN_FSDP is set but ib_wrapper is not available.")

    if not (torch.__version__.startswith("2.6") or torch.__version__.startswith("2.8")):
        raise ImportError("XTUNER_USE_CUSTOM_{AG,RS}_IN_FSDP is only supported in PyTorch 2.6 and 2.8.")

    collectives = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_collectives")

    # 1) Patch the source module attributes
    collectives.foreach_all_gather = foreach_all_gather  # type: ignore
    collectives.foreach_reduce = foreach_reduce_pt28 if torch.__version__.startswith("2.8") else foreach_reduce_pt26  # type: ignore

    # 2) Patch any already-imported modules that did
    #    `from ._fsdp_collectives import foreach_all_gather/foreach_reduce`
    patched_modules = []
    prefix = "torch.distributed.fsdp._fully_shard."
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(prefix):
            continue
        changed = False
        if getattr(mod, "foreach_all_gather", None) is not None:
            mod.foreach_all_gather = foreach_all_gather  # type: ignore
            changed = True
        if getattr(mod, "foreach_reduce", None) is not None:
            mod.foreach_reduce = foreach_reduce_pt28 if torch.__version__.startswith("2.8") else foreach_reduce_pt26  # type: ignore
            changed = True
        if changed:
            patched_modules.append(name)

    print("[xtuner] patched fsdp collectives: foreach_all_gather/foreach_reduce")
