"""Causal work arithmetic and neighbor redistribution of packed indexer queries."""

import importlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla.sequence_parallel_indexer import (
    balanced_query_partition,
    causal_work_prefix,
    get_indexer_sp_partition,
    redistribute_indexer_queries,
)


class TestIndexerSPPartition:
    @pytest.mark.parametrize("documents", [(131072,), (5, 11), (1, 3, 9, 51), (31, 1, 32)])
    @pytest.mark.parametrize("sp_size", [2, 4])
    def test_partitions_balance_packed_causal_work(self, documents: tuple[int, ...], sp_size: int) -> None:
        cu = [0]
        for length in documents:
            cu.append(cu[-1] + length)
        if cu[-1] % sp_size:
            return
        partition = balanced_query_partition(tuple(cu), sp_size, 4)
        assert partition.indexer[0] == 0 and partition.indexer[-1] == cu[-1]
        assert all(end > start for start, end in zip(partition.indexer[:-1], partition.indexer[1:]))
        expected_total = sum(sum(position // 4 for position in range(1, length + 1)) for length in documents)
        assert sum(partition.candidate_work) == expected_total
        # A boundary can differ by at most one query's indivisible candidate work.
        assert max(partition.candidate_work) - min(partition.candidate_work) <= 2 * max(documents) // 4

    @pytest.mark.parametrize("pool_size", [1, 2, 4, 7])
    def test_prefix_restarts_at_each_document(self, pool_size: int) -> None:
        cu = (0, 5, 18, 19, 64)
        costs = [position // pool_size for length in (5, 13, 1, 45) for position in range(1, length + 1)]
        for endpoint in range(65):
            assert causal_work_prefix(cu, endpoint, pool_size) == sum(costs[:endpoint])

    def test_poolless_documents_keep_uniform_shards(self) -> None:
        partition = balanced_query_partition((0, 1, 2, 3, 4), 4, 4)
        assert partition.indexer == partition.attention == (0, 1, 2, 3, 4)
        assert partition.candidate_work == (0, 0, 0, 0)

    def test_tile_alignment_avoids_an_extra_neighbor_hop(self) -> None:
        partition = balanced_query_partition((0, 131072), 4, 4, query_alignment=4)
        assert partition.indexer == (0, 65536, 92684, 113512, 131072)
        assert max(partition.candidate_work) - min(partition.candidate_work) < 4 * 32768

    @pytest.mark.parametrize("cu", [(0, 131072), (0, 5, 18, 19, 64)])
    def test_includes_global_width_work_for_every_packed_query(self, cu: tuple[int, ...]) -> None:
        per_query_work = sum((end - start + 3) // 4 for start, end in zip(cu[:-1], cu[1:])) // 2
        partition = balanced_query_partition(cu, 4, 4, query_alignment=4, per_query_work=per_query_work)
        for start, end, causal, estimated in zip(
            partition.indexer[:-1], partition.indexer[1:], partition.candidate_work, partition.estimated_work
        ):
            assert estimated == causal + (end - start) * per_query_work
        max_query_work = max((end - start) // 4 for start, end in zip(cu[:-1], cu[1:])) + per_query_work
        assert max(partition.estimated_work) - min(partition.estimated_work) <= 8 * max_query_work

    def test_rejects_nonuniform_attention_length(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            balanced_query_partition((0, 15), 4, 4)


def _cpu_selector(q, k, weights, starts, ends, topk, query_chunk_size=None):
    scores = torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu()
    scores = torch.einsum("qhk,qh->qk", scores, weights)
    ids = torch.arange(k.shape[0])
    scores.masked_fill_((ids[None] < starts[:, None]) | (ids[None] >= ends[:, None]), float("-inf"))
    values, indices = scores.topk(min(topk, k.shape[0]))
    return indices.masked_fill(values.isneginf(), -1).to(torch.int32).unsqueeze(1)


def _cpu_cooperative_selector(q, k, weights, cu, shard_start, pool_size, topk, chunk_size):
    # Substitute only the GPU scorer, preserving the production redistribution
    # and packed pool/global query coordinate conversions around it.
    ctx = SequenceContext(
        input_ids=torch.zeros(1, q.shape[0], dtype=torch.long),
        cu_seq_lens_q=cu,
        cu_seq_lens_k=cu,
        max_length_q=int((cu[1:] - cu[:-1]).max()),
        max_length_k=int((cu[1:] - cu[:-1]).max()),
        device="cpu",
        shard_start=shard_start,
        shard_size=q.shape[0],
    )
    from xtuner.v1.ops.sparse_mla.kpool import build_pool_index, pool_causal_ranges

    pool_index = build_pool_index(ctx, int(cu[-1]), pool_size, q.device)
    starts, ends = pool_causal_ranges(ctx, pool_index, q.shape[0], q.device)
    return _cpu_selector(q, k, weights, starts, ends, topk)


def _distributed_cpu_worker(rank, world_size, init_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_path}")
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        # Causal-only partition shifts a boundary beyond one attention neighbor.
        partition = balanced_query_partition((0, 64), world_size, 4, query_alignment=4)
        rows = torch.arange(partition.attention[rank], partition.attention[rank + 1])
        payload = torch.stack((rows, rows + 100), dim=-1)
        balanced_rows, balanced_payload = redistribute_indexer_queries(
            (rows, payload), partition.attention, partition.indexer, mesh
        )
        torch.testing.assert_close(balanced_rows, torch.arange(partition.indexer[rank], partition.indexer[rank + 1]))
        torch.testing.assert_close(balanced_payload, torch.stack((balanced_rows, balanced_rows + 100), dim=-1))
        restored = redistribute_indexer_queries(
            (balanced_rows, balanced_payload), partition.indexer, partition.attention, mesh
        )
        torch.testing.assert_close(restored[0], rows)
        torch.testing.assert_close(restored[1], payload)

        cooperative = importlib.import_module("xtuner.v1.ops.sparse_mla.cooperative_kpool")
        kpool = importlib.import_module("xtuner.v1.ops.sparse_mla.kpool")

        kpool.tilelang_indexer_topk_from_ranges = _cpu_selector
        cooperative.cooperative_kpool_topk = _cpu_cooperative_selector
        for cu_values in ((0, 64), (0, 5, 16, 64), (0, 1, 4, 17, 64)):
            torch.manual_seed(87)
            q = torch.randn(1, 64, 2, 8)
            k = torch.randn(1, 64, 8)
            gates = torch.randn_like(k)
            weights = torch.randn(1, 64, 2)
            ape = torch.randn(4, 8)
            cu = torch.tensor(cu_values, dtype=torch.int32)
            full_ctx = SequenceContext(
                input_ids=torch.zeros(1, 64, dtype=torch.long),
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=64,
                max_length_k=64,
                device="cpu",
            )
            local_ctx = full_ctx.copy(sequence_parallel_mesh=mesh, shard_start=rank * 16, shard_size=16)
            expected = kpool.torch_kpool_topk_indices(
                q, k, gates, weights, ape, full_ctx, index_head_dim=8, index_topk=8
            )
            for scorer in ("original", "cooperative"):
                for ratio in (0, 0.5):
                    actual = kpool.kpool_topk_indices(
                        q[:, rank * 16 : (rank + 1) * 16],
                        k[:, rank * 16 : (rank + 1) * 16],
                        gates[:, rank * 16 : (rank + 1) * 16],
                        weights[:, rank * 16 : (rank + 1) * 16],
                        ape,
                        local_ctx,
                        index_head_dim=8,
                        index_topk=8,
                        scorer=scorer,
                        balance_sp=True,
                        sp_full_pool_work_ratio=ratio,
                    )
                    torch.testing.assert_close(actual, expected[rank * 16 : (rank + 1) * 16], atol=0, rtol=0)

        # One rank's invalid local ownership must cause a collective fallback.
        invalid_ctx = full_ctx.copy(sequence_parallel_mesh=mesh, shard_start=rank * 16 + (rank == 1), shard_size=16)
        assert get_indexer_sp_partition(invalid_ctx, 4, query_len=16) is None
        assert get_indexer_sp_partition(invalid_ctx, 4, query_len=16) is None  # cached agreement
        invalid_ctx._shard_start = rank * 16
        assert get_indexer_sp_partition(invalid_ctx, 4, query_len=16) is None  # lifetime fallback, no new collective
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires the CPU Gloo backend")
def test_neighbor_redistribution_and_packed_kpool_preserve_attention_ownership(tmp_path):
    mp.spawn(_distributed_cpu_worker, args=(4, str(tmp_path / "gloo_init")), nprocs=4, join=True)
