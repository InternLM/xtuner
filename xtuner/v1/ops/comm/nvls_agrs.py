import torch
import torch.distributed as dist

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


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


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
        DEVICE_MODULE.synchronize()
        self.symm_buf_contiguous = ib_wrapper.create_symm_buffer(
            size, alignment=self.alignment, local_rank=device.index
        )

        self.symm_buf_contiguous.requires_grad_(False)

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
        DEVICE_MODULE.synchronize()

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


class AllGatherManager:
    """Manager for ibgdaAllgather objects with double buffering support.

    Handles creation, caching, and rotation of communication buffers.
    """

    def __init__(self, num_buffers: int = 2, select_comm_sm: bool = True):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.ag_ib_dict: dict[int, list] = {}
        # We implement fine-grained control over the SM IDs allocated to communication kernels,
        # enabling intelligent avoidance of conflicts with persistent computation kernels and
        # minimizing the impact of communication on computational tasks.
        self.select_comm_sm = select_comm_sm
        self.comm_sm_list = None

    def prepare_allgather_objects(
        self,
        send_bytes: int,
        process_group: dist.ProcessGroup,
        all_gather_stream: torch.Stream,
        barrier_all: bool,
    ):
        """Get or create ibgdaAllgather objects for the given byte size and
        group configuration.

        Args:
            send_bytes: Size of the tensor in bytes
            process_group: Process group for the all-gather operations
            all_gather_stream: CUDA stream for all-gather operations
            barrier_all: Whether to use barrier synchronization among all nodes
        """
        if send_bytes in self.ag_ib_dict:
            # Reuse existing ibgdaAllgather objects
            return

        device_count = DEVICE_MODULE.device_count()
        is_full_world = process_group.size() == dist.get_world_size()
        is_horizontal = process_group.size() == device_count
        ranks = dist.get_process_group_ranks(process_group)
        for i, rank in enumerate(ranks):
            if rank % device_count != i:
                is_horizontal = False
                break

        if not (is_full_world or is_horizontal):
            raise NotImplementedError("ibReduceScatter only supports full world or horizontal process groups.")

        DEVICE_MODULE.synchronize()
        if self.comm_sm_list is None:
            self.comm_sm_list, _ = ib_wrapper.init_comm_sm()

        AGs = [
            ibgdaAllgather(
                send_bytes,
                process_group,
                all_gather_stream,
                mode=0,
                barrier_all=barrier_all,
                vertical_group_ag=False,
                comm_sm_list=self.comm_sm_list,
                select_sm=self.select_comm_sm,
            )
            for _ in range(self.num_buffers)
        ]

        DEVICE_MODULE.synchronize()
        self.ag_ib_dict[send_bytes] = AGs

    def execute_allgather(
        self,
        send_bytes: int,
        all_gather_output,
        all_gather_input,
        process_group,
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
        # Use cached ibgdaAllgather objects
        self.ag_ib_dict[send_bytes][self.comm_buf_iter].all_gather_into_tensor(
            all_gather_output, all_gather_input, group=process_group
        )
        # Rotate buffer iterator
        self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers

    def clear_cache(self):
        """Clear all cached all-gather objects."""
        self.ag_ib_dict.clear()
        self.comm_buf_iter = 0


class ReduceScatterManager:
    """Manager for ibReduceScatter objects with double buffering support.

    Handles creation, caching, and execution of reduce-scatter operations.
    """

    def __init__(self, num_buffers: int = 2, select_comm_sm: bool = True):
        self.num_buffers = num_buffers
        self.comm_buf_iter = 0
        self.rs_ib_dict: dict[int, list] = {}
        # We implement fine-grained control over the SM IDs allocated to communication kernels,
        # enabling intelligent avoidance of conflicts with persistent computation kernels and
        # minimizing the impact of communication on computational tasks.
        self.select_comm_sm = select_comm_sm
        self.comm_sm_list = None

    def prepare_reducescatter_objects(
        self,
        recv_bytes: int,
        process_group: dist.ProcessGroup,
        reduce_scatter_stream: torch.Stream,
        barrier_all: bool,
    ):
        """Create ibReduceScatter objects for the given byte size and group
        configuration if necessary.

        Args:
            recv_bytes: Size of the receive buffer in bytes
            process_group: Process group for the reduce-scatter operations
            reduce_scatter_stream: CUDA stream for reduce-scatter operations
            barrier_all: Whether to use barrier synchronization among all nodes
        """
        if recv_bytes in self.rs_ib_dict:
            return

        device_count = DEVICE_MODULE.device_count()
        is_full_world = process_group.size() == dist.get_world_size()
        is_horizontal = process_group.size() == device_count
        ranks = dist.get_process_group_ranks(process_group)
        for i, rank in enumerate(ranks):
            if rank % device_count != i:
                is_horizontal = False
                break

        if not (is_full_world or is_horizontal):
            raise NotImplementedError("ibReduceScatter only supports full world or horizontal process groups.")

        DEVICE_MODULE.synchronize()
        if self.comm_sm_list is None:
            self.comm_sm_list, _ = ib_wrapper.init_comm_sm()

        RSs = [
            ibReduceScatter(
                recv_bytes,
                process_group,
                reduce_scatter_stream,
                mode=0,
                barrier_all=barrier_all,
                vertical_group_rs=False,
                select_sm=self.select_comm_sm,
                comm_sm_list=self.comm_sm_list,
            )
            for _ in range(self.num_buffers)
        ]

        DEVICE_MODULE.synchronize()
        self.rs_ib_dict[recv_bytes] = RSs

    def execute_reducescatter(
        self,
        recv_bytes: int,
        reduce_scatter_output,
        reduce_scatter_input,
        reduce_scatter_group,
        reduce_scatter_reduce_op,
    ):
        """Execute reduce-scatter operation using cached ibReduceScatter
        objects.

        Args:
            recv_bytes: Size of the receive buffer in bytes
            reduce_scatter_input: Input tensor
            reduce_scatter_group: Process group
            reduce_scatter_stream: CUDA stream
            reduce_scatter_reduce_op: Reduction operation
            reduce_scatter_output_numel: Output tensor element count
        """
        # Use cached ibReduceScatter objects with buffer swapping
        self.rs_ib_dict[recv_bytes][self.comm_buf_iter].reduce_scatter_tensor(
            output=reduce_scatter_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=reduce_scatter_reduce_op,
        )
        self.comm_buf_iter = (self.comm_buf_iter + 1) % self.num_buffers

        return reduce_scatter_output

    def clear_cache(self):
        """Clear all cached reduce-scatter objects."""
        self.rs_ib_dict.clear()
        self.comm_buf_iter = 0
