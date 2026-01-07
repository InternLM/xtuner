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


class SymmBufferManager:
    """A manager class for symmetric buffer allocation and lifecycle
    management.

    Optimizes buffer reuse and handles dynamic resizing based on communication requirements. Implements n buffering for
    concurrent operations with contiguous memory.
    """

    def __init__(self, default_size=0, alignment=128, num_buffers=3):
        """Initialize the symmetric buffer manager with n buffering in
        contiguous memory.

        Args:
            default_size (int): Default buffer size in bytes
            alignment (int): Memory alignment requirement for the buffer
            num_buffers (int): Number of buffers for n-buffering
        """
        self.symm_buf_contiguous = None  # Contiguous memory block for all buffers
        self.symm_buf_ptrs = [None] * num_buffers  # Pointers to individual buffers within contiguous block
        self.current_buffer_idx = 0  # Index of current active buffer
        self.symm_buf_size = default_size  # Current configured buffer size
        self.alignment = alignment  # Memory alignment for optimal performance
        self._creation_count = 0  # Track how many times buffers have been created
        self.num_buffers = num_buffers  # Number of buffers for n-buffering

    def get_buffer(self, bytes, device):
        """Get or create a symmetric buffer of appropriate size for the
        communication. Implements n buffering by cycling through multiple
        buffers in contiguous memory.

        Args:
            bytes (int): Number of bytes required for the current operation
            device: The device (GPU) where the buffer should be allocated

        Returns:
            The symmetric buffer object ready for use
        """
        # Calculate required size - use the larger of configured size or actual need
        required_size = max(self.symm_buf_size, bytes * self.num_buffers)

        # Case 1: No contiguous buffer exists - create initial contiguous buffer
        if self.symm_buf_contiguous is None:
            self._create_contiguous_buffer(required_size, device, "initial creation")

        # Case 2: Existing contiguous buffer is too small - recreate with larger size
        elif self.symm_buf_contiguous.numel() < required_size:
            print(f"Buffer resize required: {self.symm_buf_contiguous.numel()} -> {required_size}")
            del self.symm_buf_contiguous
            self.symm_buf_ptrs = [None] * self.num_buffers
            self._create_contiguous_buffer(required_size, device, "resize due to insufficient size")

        # Case 3: Buffer exists and is large enough - reuse existing buffer
        # No action needed

        # Get the current buffer pointer
        current_buffer_ptr = self.symm_buf_ptrs[self.current_buffer_idx]

        # Shift to the next buffer for next call
        self._shift_buffer()

        return current_buffer_ptr

    def _shift_buffer(self):
        """Shift to the next buffer in the n buffer system using round-
        robin."""
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.num_buffers

    def _create_contiguous_buffer(self, size, device, reason):
        """Internal method to create a contiguous memory block for all buffers.

        Args:
            size (int): Total size of the contiguous buffer
            device: Target device for buffer allocation
            reason (str): Description of why the buffer is being created (for debugging)
        """
        print(f"{reason = }, {size / 2**30:.1f} GB")
        self.symm_buf_contiguous = ib_wrapper.create_symm_buffer(
            size, alignment=self.alignment, local_rank=device.index
        )

        # Calculate size per buffer with alignment
        size_per_buf = ((size // self.num_buffers) + self.alignment - 1) // self.alignment * self.alignment

        # Create pointers to each individual buffer
        for i in range(self.num_buffers):
            start_idx = i * size_per_buf
            end_idx = (i + 1) * size_per_buf
            # Ensure we don't exceed the total buffer size
            if end_idx > size:
                end_idx = size
            self.symm_buf_ptrs[i] = self.symm_buf_contiguous[start_idx:end_idx]

        self._creation_count += 1

    def resize(self, new_size, device):
        """Explicitly resize all buffers to a new size.

        Args:
            new_size (int): New buffer size in bytes
            device: Target device for the resized buffer
        """
        if self.symm_buf_contiguous is not None:
            del self.symm_buf_contiguous
            self.symm_buf_contiguous = None
            self.symm_buf_ptrs = [None] * self.num_buffers

        self.symm_buf_size = new_size

        # Create new contiguous buffer with new size
        self._create_contiguous_buffer(new_size, device, "explicit resize")

    def release(self):
        """Explicitly release all buffer resources."""
        if self.symm_buf_contiguous is not None:
            del self.symm_buf_contiguous
            self.symm_buf_contiguous = None
            self.symm_buf_ptrs = [None] * self.num_buffers

    def get_current_buffer_index(self):
        """Get the index of the current active buffer.

        Returns:
            int: Current buffer index (0 to num_buffers-1)
        """
        return self.current_buffer_idx

    def get_stats(self):
        """Get statistics about buffer usage.

        Returns:
            dict: Buffer statistics including current sizes and creation count
        """
        if self.symm_buf_contiguous is not None:
            # Calculate individual buffer sizes
            size_per_buf = (
                ((self.symm_buf_contiguous.numel() // self.num_buffers) + self.alignment - 1)
                // self.alignment
                * self.alignment
            )
            current_sizes = [size_per_buf] * self.num_buffers
            # Adjust the last buffer size if needed
            total_allocated = size_per_buf * self.num_buffers
            if total_allocated > self.symm_buf_contiguous.numel():
                current_sizes[-1] = self.symm_buf_contiguous.numel() - size_per_buf * (self.num_buffers - 1)
        else:
            current_sizes = [0] * self.num_buffers

        return {
            "current_sizes": current_sizes,
            "contiguous_size": self.symm_buf_contiguous.numel() if self.symm_buf_contiguous is not None else 0,
            "current_buffer_index": self.current_buffer_idx,
            "configured_size": self.symm_buf_size,
            "creation_count": self._creation_count,
            "alignment": self.alignment,
            "num_buffers": self.num_buffers,
            "is_contiguous": self.symm_buf_contiguous is not None,
        }

    def __del__(self):
        """Destructor to ensure proper resource cleanup."""
        self.release()


class AllGatherIBManager:
    """Manager for ibgdaAllgather objects with double buffering support.

    Handles creation, caching, and rotation of communication buffers.
    """

    def __init__(self, num_buffers: int = 2, use_custom_ag: bool = False, select_sm: bool = False):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.use_custom_ag = use_custom_ag
        self.ag_ib_dict: dict[int, list] = {}
        self.select_sm = int(os.getenv("SELECT_COMM_SM_IN_FSDP", 0))

        self.comm_sm_list = None

    def prepare_allgather_objects(
        self, send_bytes: int, group_size: int, world_size: int, device_count: int, all_gather_stream, mode: int = 0
    ):
        """Get or create ibgdaAllgather objects for the given byte size and
        group configuration.

        Args:
            send_bytes: Size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            device_count: Number of CUDA devices per node
            all_gather_stream: CUDA stream for all-gather operations
            mode: Operation mode for ibgdaAllgather
        """
        if send_bytes not in self.ag_ib_dict and self.use_custom_ag:
            torch.cuda.synchronize()
            if self.comm_sm_list is None:
                self.comm_sm_list, _ = ib_wrapper.init_comm_sm()
            # Determine if this is a full world or EP group
            is_ep_group = group_size == world_size // device_count

            vertical_group_ag = is_ep_group  # Use local rank for EP groups

            # Create double buffered all-gather objects
            AGs = []
            if vertical_group_ag:
                for _ in range(self.num_buffers):
                    AGs.append(
                        ibgdaAllgather(
                            send_bytes,
                            torch.distributed.group.WORLD,
                            all_gather_stream,
                            mode=1,
                            barrier_all=True,
                            vertical_group_ag=vertical_group_ag,
                        )
                    )

            else:
                for _ in range(self.num_buffers):
                    AGs.append(
                        ibgdaAllgather(
                            send_bytes,
                            torch.distributed.group.WORLD,
                            all_gather_stream,
                            mode=0,
                            barrier_all=True,
                            vertical_group_ag=vertical_group_ag,
                            comm_sm_list=self.comm_sm_list,
                            select_sm=self.select_sm,
                        )
                    )
            torch.cuda.synchronize()
            self.ag_ib_dict[send_bytes] = AGs

    def execute_allgather(
        self, send_bytes: int, group_size: int, world_size: int, all_gather_output, all_gather_input, group
    ):
        """Execute all-gather operation using cached ibgdaAllgather objects.

        Args:
            send_bytes: Size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            all_gather_output: Output tensor for all-gather
            all_gather_input: Input tensor for all-gather
            group: Process group for the operation
        """
        if self.use_custom_ag:
            if group_size == world_size or group_size == world_size // torch.cuda.device_count():
                # Use cached ibgdaAllgather objects
                self.ag_ib_dict[send_bytes][self.comm_buf_iter].all_gather_into_tensor(
                    all_gather_output, all_gather_input, group=group
                )
                # Rotate buffer iterator
                self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers

    def clear_cache(self):
        """Clear all cached all-gather objects."""
        self.ag_ib_dict.clear()
        self.comm_buf_iter = 0


class ReduceScatterIBManager:
    """Manager for ibReduceScatter objects with double buffering support.

    Handles creation, caching, and execution of reduce-scatter operations.
    """

    def __init__(self, num_buffers: int = 2, use_custom_rs: bool = False):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.use_custom_rs = use_custom_rs
        self.rs_ib_dict: dict[int, list] = {}
        # self.rdc_scale: dict[int, torch.Tensor] = {}
        self.copy_event_prev: torch.Event | None = None
        self.copy_event: torch.Event | None = None
        self.select_sm = int(os.getenv("SELECT_COMM_SM_IN_FSDP", 0))

        self.comm_sm_list = None

    def prepare_reducescatter_objects(
        self, recv_bytes_aligned: int, group_size: int, world_size: int, device_count: int, reduce_scatter_stream
    ):
        """Get or create ibReduceScatter objects for the given byte size and
        group configuration.

        Args:
            recv_bytes_aligned: Aligned size of the tensor in bytes
            group_size: Size of the process group
            world_size: Total world size
            device_count: Number of CUDA devices per node
            reduce_scatter_stream: CUDA stream for reduce-scatter operations
        """
        if recv_bytes_aligned not in self.rs_ib_dict and self.use_custom_rs:
            torch.cuda.synchronize()
            if self.comm_sm_list is None:
                self.comm_sm_list, _ = ib_wrapper.init_comm_sm()

            # Determine group type and configuration
            is_full_world = group_size == world_size
            is_ep_group = group_size == world_size // device_count

            RSs = []
            if is_full_world:
                for _ in range(self.num_buffers):
                    RSs.append(
                        ibReduceScatter(
                            recv_bytes_aligned,
                            torch.distributed.group.WORLD,
                            reduce_scatter_stream,
                            mode=0,
                            barrier_all=True,
                            vertical_group_rs=False,
                            select_sm=self.select_sm,
                            comm_sm_list=self.comm_sm_list,
                        )
                    )

            elif is_ep_group:
                for _ in range(self.num_buffers):
                    RSs.append(
                        ibReduceScatter(
                            recv_bytes_aligned,
                            torch.distributed.group.WORLD,
                            reduce_scatter_stream,
                            mode=1,
                            barrier_all=True,
                            vertical_group_rs=True,
                        )
                    )

            torch.cuda.synchronize()
            self.rs_ib_dict[recv_bytes_aligned] = RSs
            # self.rdc_scale[recv_bytes_aligned] = None

    def execute_reducescatter(
        self,
        recv_bytes_aligned: int,
        group_size: int,
        world_size: int,
        reduce_scatter_input,
        reduce_scatter_input_aligned,
        reduce_scatter_group,
        reduce_scatter_stream,
        reduce_scatter_reduce_op,
        recv_bytes: int,
        reduce_scatter_output_numel: int,
        device,
    ):
        """Execute reduce-scatter operation using cached ibReduceScatter
        objects.

        Args:
            recv_bytes_aligned: Aligned size in bytes
            group_size: Process group size
            world_size: Total world size
            reduce_scatter_input: Input tensor
            reduce_scatter_input_aligned: Aligned input tensor
            reduce_scatter_group: Process group
            reduce_scatter_stream: CUDA stream
            reduce_scatter_reduce_op: Reduction operation
            recv_bytes: Original receive bytes
            reduce_scatter_output_numel: Output tensor element count
            device: Target device
        """
        with torch.cuda.stream(reduce_scatter_stream):
            if self.use_custom_rs and (
                group_size == world_size or group_size == world_size // torch.cuda.device_count()
            ):
                # Use cached ibReduceScatter objects with buffer swapping
                reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
                # Execute reduce-scatter
                self.rs_ib_dict[recv_bytes_aligned][self.comm_buf_iter].reduce_scatter_tensor(
                    output=reduce_output,
                    input=reduce_scatter_input_aligned,
                    group=reduce_scatter_group,
                    op=reduce_scatter_reduce_op,
                )
                self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers

        return reduce_output

    def clear_cache(self):
        """Clear all cached reduce-scatter objects."""
        self.rs_ib_dict.clear()
        self.comm_buf_iter = 0


ag_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=NUM_AG_BUFFERS)
rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=NUM_RS_BUFFERS)

ag_manager = AllGatherIBManager(num_buffers=NUM_AG_BUFFERS, use_custom_ag=USE_CUSTOM_AG)
rs_manager = ReduceScatterIBManager(num_buffers=NUM_RS_BUFFERS, use_custom_rs=USE_CUSTOM_RS)


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
            group_size=group.size(),
            world_size=dist.get_world_size(),
            device_count=torch.cuda.device_count(),
            all_gather_stream=all_gather_stream,
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
                group_size=group.size(),
                world_size=dist.get_world_size(),
                all_gather_output=all_gather_output,
                all_gather_input=all_gather_input,
                group=group,
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
            recv_bytes_aligned=recv_bytes,
            group_size=reduce_scatter_group.size(),
            world_size=dist.get_world_size(),
            device_count=torch.cuda.device_count(),
            reduce_scatter_stream=reduce_scatter_stream,
        )

    if use_custom_rs:
        symm_buf = rs_symm.get_buffer(bytes=send_bytes_aligned, device=device)
        reduce_scatter_input = symm_buf.view(reduce_dtype)[:reduce_scatter_input_numel]
        reduce_scatter_input_aligned = reduce_scatter_input
    else:
        reduce_scatter_input = torch.empty((reduce_scatter_input_numel,), dtype=reduce_dtype, device=device)
        reduce_scatter_input_aligned = reduce_scatter_input
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
        # ---------- diff --------- #
        if use_custom_rs:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes_aligned=recv_bytes,
                group_size=reduce_scatter_group.size(),
                world_size=dist.get_world_size(),
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_input_aligned=reduce_scatter_input_aligned,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_reduce_op=reduce_scatter_reduce_op,
                recv_bytes=recv_bytes,
                reduce_scatter_output_numel=reduce_scatter_output_numel,
                device=device,
            )
        else:
            reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
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
            recv_bytes_aligned=recv_bytes,
            group_size=reduce_scatter_group.size(),
            world_size=dist.get_world_size(),
            device_count=torch.cuda.device_count(),
            reduce_scatter_stream=reduce_scatter_stream,
        )

    if use_custom_rs:
        symm_buf = rs_symm.get_buffer(bytes=send_bytes_aligned, device=device)
        reduce_scatter_input = symm_buf.view(reduce_dtype)[:reduce_scatter_input_numel]
        reduce_scatter_input_aligned = reduce_scatter_input
    else:
        reduce_scatter_input = allocate_memory(
            reduce_scatter_input_numel,
            dtype=reduce_dtype,
            device=device,
            group=reduce_scatter_group,
            from_process_group=allocate_memory_from_process_group,
        )
        # reduce_scatter_input = torch.empty(
        #     (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
        # )
        reduce_scatter_input_aligned = reduce_scatter_input
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

        # ---------- diff --------- #
        if use_custom_rs:
            reduce_output = rs_manager.execute_reducescatter(
                recv_bytes_aligned=recv_bytes,
                group_size=reduce_scatter_group.size(),
                world_size=dist.get_world_size(),
                reduce_scatter_input=reduce_scatter_input,
                reduce_scatter_input_aligned=reduce_scatter_input_aligned,
                reduce_scatter_group=reduce_scatter_group,
                reduce_scatter_stream=reduce_scatter_stream,
                reduce_scatter_reduce_op=reduce_scatter_op,
                recv_bytes=recv_bytes,
                reduce_scatter_output_numel=reduce_scatter_output_numel,
                device=device,
            )
        else:
            reduce_output = allocate_memory(
                reduce_scatter_output_numel,
                dtype=reduce_dtype,
                device=device,
                group=reduce_scatter_group,
                from_process_group=allocate_memory_from_process_group,
            )
            # reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
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
