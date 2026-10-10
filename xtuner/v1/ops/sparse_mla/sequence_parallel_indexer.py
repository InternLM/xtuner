# Copyright (c) OpenMMLab. All rights reserved.
"""Balance causal indexer work while preserving the attention sequence shards."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.data_proto import SequenceContext


@dataclass(frozen=True)
class IndexerSPPartition:
    """Global query boundaries for attention and balanced indexer ownership."""

    attention: tuple[int, ...]
    indexer: tuple[int, ...]
    candidate_work: tuple[int, ...]
    estimated_work: tuple[int, ...]


def causal_work_prefix(cu_seq_lens: tuple[int, ...], position: int, pool_size: int) -> int:
    """Count visible complete pools scored by queries before a global position.

    Args:
        cu_seq_lens (tuple[int, ...]): Global packed document boundaries.
        position (int): Exclusive global query endpoint.
        pool_size (int): Number of tokens in a candidate pool; use one for token indexers.

    Returns:
        int: Sum of per-query visible candidate counts, restarting at document boundaries.
    """
    total = 0
    for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:]):
        count = min(max(position - start, 0), end - start)
        complete, remainder = divmod(count, pool_size)
        total += pool_size * complete * (complete - 1) // 2 + complete * (remainder + 1)
        if position <= end:
            break
    return total


def balanced_query_partition(
    cu_seq_lens: tuple[int, ...], sp_size: int, pool_size: int, query_alignment: int = 1, per_query_work: int = 0
) -> IndexerSPPartition:
    """Partition queries into contiguous intervals with approximately equal causal work.

    Args:
        cu_seq_lens (tuple[int, ...]): Global packed document boundaries.
        sp_size (int): Number of sequence parallel ranks.
        pool_size (int): Number of tokens in a candidate pool.
        query_alignment (int): Prefer boundaries aligned to the selector's query tile.
        per_query_work (int): Additional fixed work per query, in candidate-scoring units.
    Returns:
        IndexerSPPartition: Uniform attention boundaries, balanced query boundaries, and work per rank.
    """
    if (
        sp_size < 1
        or pool_size < 1
        or query_alignment < 1
        or per_query_work < 0
        or len(cu_seq_lens) < 2
        or cu_seq_lens[0] != 0
    ):
        raise ValueError("Expected positive SP/pool sizes and packed document boundaries starting at zero")
    if any(end < start for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:])):
        raise ValueError("Document boundaries must be nondecreasing")
    length = cu_seq_lens[-1]
    if length % sp_size:
        raise ValueError("Attention sequence length must be divisible by SP size")
    attention = tuple(rank * (length // sp_size) for rank in range(sp_size + 1))
    # Prefix sums are evaluated using document arithmetic rather than an O(S) tensor or a
    # CUDA synchronization per query. The small packed-document metadata is cached below.
    doc_work = [0]
    for start, end in zip(cu_seq_lens[:-1], cu_seq_lens[1:]):
        doc_work.append(doc_work[-1] + causal_work_prefix((0, end - start), end - start, pool_size))

    def prefix(position: int) -> int:
        doc = min(bisect_right(cu_seq_lens, position) - 1, len(cu_seq_lens) - 2)
        return (
            position * per_query_work
            + doc_work[doc]
            + causal_work_prefix((0, cu_seq_lens[doc + 1] - cu_seq_lens[doc]), position - cu_seq_lens[doc], pool_size)
        )

    total_work = doc_work[-1] + length * per_query_work
    boundaries = [0]
    for rank in range(1, sp_size):
        if total_work == 0:
            boundaries.append(attention[rank])
            continue
        target = total_work * rank
        low, high = boundaries[-1], length
        while low < high:
            middle = (low + high) // 2
            if prefix(middle) * sp_size < target:
                low = middle + 1
            else:
                high = middle
        # Tile-aligned boundaries avoid extra padded scoring and tiny transfers over an
        # additional neighbor when a balanced boundary nearly matches an attention seam.
        aligned = max(0, (low - 1) // query_alignment * query_alignment)
        candidates = (max(boundaries[-1], aligned), min(length, aligned + query_alignment))
        endpoint = min(candidates, key=lambda value: abs(prefix(value) * sp_size - target))
        # Keep a nonempty query shard whenever the uniform attention partition has one.
        if length >= sp_size:
            endpoint = min(max(endpoint, boundaries[-1] + 1), length - (sp_size - rank))
        boundaries.append(endpoint)
    boundaries.append(length)
    estimated_work = tuple(prefix(end) - prefix(start) for start, end in zip(boundaries[:-1], boundaries[1:]))
    candidate_work = tuple(
        work - (end - start) * per_query_work
        for start, end, work in zip(boundaries[:-1], boundaries[1:], estimated_work)
    )
    return IndexerSPPartition(attention, tuple(boundaries), candidate_work, estimated_work)


@torch.compiler.disable
def get_indexer_sp_partition(
    seq_ctx: SequenceContext,
    pool_size: int,
    query_alignment: int = 1,
    per_query_work: int = 0,
    query_len: int | None = None,
) -> IndexerSPPartition | None:
    """Cache the work partition once per packed sequence context.

    Args:
        seq_ctx (SequenceContext): Uniform attention query context.
        pool_size (int): Number of tokens in a candidate pool.
        query_alignment (int): Prefer boundaries aligned to the selector's query tile.
        per_query_work (int): Additional fixed work per query, in candidate-scoring units.
        query_len (int | None): Local attention query count, when known. Ownership
            must stay unchanged for the lifetime of an eligible context.

    Returns:
        IndexerSPPartition | None: Query ownership, or None for unsupported ownership
            or when redistribution cannot lower estimated maximum rank work.
    """
    mesh = seq_ctx.sequence_parallel_mesh
    if mesh is None or mesh.size() == 1 or mesh.ndim != 1:
        return None
    cache = getattr(seq_ctx, "_indexer_sp_partitions", None)
    if cache is None:
        cache = {}
        setattr(seq_ctx, "_indexer_sp_partitions", cache)
    key = (mesh.size(), pool_size, query_alignment, per_query_work)
    if key not in cache:
        cu = tuple(seq_ctx.cu_seq_lens_q.detach().cpu().tolist())
        if cu[-1] % mesh.size():
            cache[key] = None
        else:
            partition = balanced_query_partition(cu, mesh.size(), pool_size, query_alignment, per_query_work)
            uniform_work = tuple(
                causal_work_prefix(cu, end, pool_size)
                - causal_work_prefix(cu, start, pool_size)
                + (end - start) * per_query_work
                for start, end in zip(partition.attention[:-1], partition.attention[1:])
            )
            cache[key] = partition if max(partition.estimated_work) < max(uniform_work) else None
    partition = cache[key]
    if partition is None:
        return None
    # Every rank must make the same fallback decision before entering P2P. Cache
    # this tiny agreement check once per context/shape, alongside the work plan.
    # Cache progression uses only global partition parameters. A local ownership
    # change must never make one rank enter a new collective while its peers hit
    # their caches. Context ownership is immutable after a successful agreement;
    # an ineligible context keeps its uniform fallback for its whole lifetime.
    local_signature = (seq_ctx.shard_start, query_len)
    eligibility = getattr(seq_ctx, "_indexer_sp_eligibility", None)
    if eligibility is None:
        eligibility = {}
        setattr(seq_ctx, "_indexer_sp_eligibility", eligibility)
    if key not in eligibility:
        local_valid = seq_ctx.shard_start == partition.attention[mesh.get_local_rank()]
        local_valid &= query_len is None or query_len == partition.attention[1]
        group = mesh.get_group()
        device = torch.device("cpu")
        if dist.get_backend(group) == "nccl":
            device = seq_ctx.cu_seq_lens_q.device
            if device.type != "cuda":
                device = torch.device("cuda", torch.cuda.current_device())
        valid = torch.tensor(int(local_valid), dtype=torch.int32, device=device)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=group)
        eligibility[key] = (local_signature, bool(valid.item()))
    signature, globally_valid = eligibility[key]
    if not globally_valid:
        return None
    if signature != local_signature:
        raise RuntimeError("Balanced indexer requires immutable SequenceContext query ownership; create a new context")
    return partition


@torch.compiler.disable
@torch.no_grad()
def redistribute_indexer_queries(
    tensors: tuple[Tensor, ...],
    source: tuple[int, ...],
    target: tuple[int, ...],
    mesh: DeviceMesh,
) -> tuple[Tensor, ...]:
    """Move contiguous query rows between ownership schemes using only neighbor P2P.

    Each source/target intersection follows the shortest path along the SP ranks. A
    boundary crossing more than one attention shard therefore uses multiple neighbor hops.
    The deterministic metadata gives matching send/receive shapes without size exchanges.

    Args:
        tensors (tuple[Tensor, ...]): Local source tensors with query rows in dimension zero.
        source (tuple[int, ...]): Global source query boundaries.
        target (tuple[int, ...]): Global destination query boundaries.
        mesh (DeviceMesh): One dimensional SP mesh.

    Returns:
        tuple[Tensor, ...]: Tensors ordered by global query position on the destination rank.
    """
    if source == target:
        return tensors
    rank, size, group = mesh.get_local_rank(), mesh.size(), mesh.get_group()
    if len(source) != size + 1 or len(target) != size + 1 or source[-1] != target[-1]:
        raise ValueError("Source and destination boundaries must span the same SP sequence")
    if any(tensor.shape[0] != source[rank + 1] - source[rank] for tensor in tensors):
        raise ValueError("Tensor query count does not match its source shard")
    result = tuple(tensor.new_empty((target[rank + 1] - target[rank], *tensor.shape[1:])) for tensor in tensors)
    routes = []
    pieces: dict[tuple[int, int], tuple[Tensor, ...]] = {}
    for src in range(size):
        first_dst = max(0, bisect_right(target, source[src]) - 1)
        last_dst = min(size, bisect_left(target, source[src + 1]) + 1)
        for dst in range(first_dst, last_dst):
            start, end = max(source[src], target[dst]), min(source[src + 1], target[dst + 1])
            if start >= end:
                continue
            if src == dst:
                if src == rank:
                    for output, tensor in zip(result, tensors):
                        output[start - target[rank] : end - target[rank]].copy_(
                            tensor[start - source[rank] : end - source[rank]]
                        )
                continue
            routes.append((src, dst, start, end))
            if src == rank:
                pieces[src, dst] = tuple(
                    tensor[start - source[src] : end - source[src]].contiguous() for tensor in tensors
                )
    max_hops = max((abs(dst - src) for src, dst, _, _ in routes), default=0)
    for hop in range(max_hops):
        operations = []
        sent = []
        received = {}
        for src, dst, start, end in routes:
            if hop >= abs(dst - src):
                continue
            step = 1 if dst > src else -1
            sender, receiver = src + hop * step, src + (hop + 1) * step
            if sender == rank:
                for tensor in pieces[src, dst]:
                    operations.append(dist.P2POp(dist.isend, tensor, dist.get_global_rank(group, receiver), group))
                sent.append((src, dst))
            elif receiver == rank:
                if receiver == dst:
                    buffers = tuple(output[start - target[rank] : end - target[rank]] for output in result)
                else:
                    buffers = tuple(tensor.new_empty((end - start, *tensor.shape[1:])) for tensor in tensors)
                for buffer in buffers:
                    operations.append(dist.P2POp(dist.irecv, buffer, dist.get_global_rank(group, sender), group))
                received[src, dst] = buffers
        if operations:
            for request in dist.batch_isend_irecv(operations):
                request.wait()
        for key in sent:
            del pieces[key]
        pieces.update(received)
    return result
