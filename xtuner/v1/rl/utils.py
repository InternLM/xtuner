import atexit
import heapq
import signal
import subprocess
from typing import Any

import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from xtuner.v1.utils.logger import get_logger


def sp_split(
    tensor,
    sp_mesh: DeviceMesh,
    split_dim: int,
    padding_value: Any,
):
    tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
    tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
    return tensor


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs


logger = get_logger()


def close_ray():
    """Clean up the ray resource."""
    import ray

    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup():
    """Register cleanup handlers for Ray on exit and signals."""
    _cleaned = False

    def cleanup_once():
        nonlocal _cleaned
        if not _cleaned:
            _cleaned = True
            close_ray()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once()
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup_once)


# Adapted from https://github.com/volcengine/verl/blob/eb6991a622e15c494ee8403e2289708b2a3b278f/verl/utils/seqlen_balancing.py#L37
def calculate_workload(seqlen_list: list) -> list:
    """Calculate approximate computational workload for transformer attention.

    Estimates FLOPs for dense transformer blocks based on sequence length using
    the formula: FLOPs ≈ 12 * hidden_size² * seqlen + 2 * hidden_size * seqlen²

    The constants are calibrated for a 7B model (hidden_size=4096), yielding:
    workload ∝ 24576 * seqlen + seqlen²

    Args:
        seqlen_list: Sequence lengths as a tensor.

    Returns:
        torch.Tensor: Estimated workload values proportional to actual FLOPs.

    Note:
        The returned values are relative workloads, not actual FLOP counts.
        Useful for balancing computation across data parallel ranks.
    """
    workload_list = []
    for seqlen in seqlen_list:
        workload = 24576 * seqlen + seqlen * seqlen
        workload_list.append(workload)
    return workload_list


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """Calculates partitions of indices from seqlen_list such that the sum of
    sequence lengths in each partition is balanced. Uses the Karmarkar-Karp
    differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                        Requires len(seqlen_list) to be divisible by k_partitions.
                        If False, partitions can have varying numbers of items, focusing
                        only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                        original indices of the items assigned to that partition. The indices
                        within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)
