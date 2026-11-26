# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-GPU KPool gathering, causal ranges and local query ownership."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla import get_kpool_topk_indices
from xtuner.v1.ops.sparse_mla.lmdeploy_kpool import ensure_lmdeploy_kpool_runtime_available


def _sp_worker(rank: int, init_path: str, backend: str) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"file://{init_path}")
    try:
        select = get_kpool_topk_indices(backend)
        mesh = init_device_mesh("cuda", (2,))
        # First case: pool [4, 5, 6, 7] straddles the shard seam at token 6.
        # Other cases place a document boundary at/before the seam and exercise
        # short documents, tails and a context exceeding the Top-K budget.
        for lengths in ((12,), (6, 6), (5, 7), (1, 3, 8), (2051, 2053)):
            torch.manual_seed(84)
            tokens = sum(lengths)
            q = torch.randn(1, tokens, 32, 128, dtype=torch.bfloat16).to(device)
            k = torch.randn(1, tokens, 128, dtype=torch.bfloat16).to(device)
            gates = torch.randn_like(k)
            weights = torch.randn(1, tokens, 32, device=device)
            ape = torch.randn(4, 128, device=device)
            full_ctx = SequenceContext.from_input_ids(
                tuple(torch.zeros(1, n, dtype=torch.long) for n in lengths), device=device
            )
            shard = tokens // 2
            local_ctx = full_ctx.copy(sequence_parallel_mesh=mesh, shard_start=rank * shard, shard_size=shard)
            local = slice(rank * shard, (rank + 1) * shard)
            arguments = {"index_head_dim": 128, "index_topk": 2048, "query_chunk_size": 257}
            expected = select(q, k, gates, weights, ape, full_ctx, **arguments)
            actual = select(
                q[:, local].contiguous(),
                k[:, local].contiguous(),
                gates[:, local].contiguous(),
                weights[:, local].contiguous(),
                ape,
                local_ctx,
                **arguments,
            )
            torch.testing.assert_close(actual, expected[local], atol=0, rtol=0)
            assert actual.shape == (shard, 1, 2560)
            # Independent token-level oracle: these short cases select every
            # causal token, including incomplete tails, without crossing docs.
            if tokens <= 12:
                boundaries = [0]
                for length in lengths:
                    boundaries.append(boundaries[-1] + length)
                for row, global_query in enumerate(range(rank * shard, (rank + 1) * shard)):
                    doc_start = max(start for start in boundaries[:-1] if start <= global_query)
                    valid = actual[row, 0][actual[row, 0] >= 0].sort().values
                    wanted = torch.arange(doc_start, global_query + 1, device=device, dtype=torch.int32)
                    torch.testing.assert_close(valid, wanted, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


class TestLMDeployKPoolSequenceParallel:
    @pytest.mark.gpu
    @pytest.mark.parametrize("backend", ["lmdeploy_fp8", "lmdeploy_bf16"])
    def test_two_gpu_gathering_and_packed_ownership(self, tmp_path: Path, backend: str) -> None:
        if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
            pytest.skip("Two CUDA GPUs and NCCL required")
        if any(torch.cuda.get_device_capability(i)[0] < 9 for i in range(2)):
            pytest.skip("SM90+ required")
        try:
            ensure_lmdeploy_kpool_runtime_available(backend.removeprefix("lmdeploy_"))
        except RuntimeError as error:
            pytest.skip(str(error))
        mp.spawn(_sp_worker, args=(str(tmp_path / "nccl_init"), backend), nprocs=2, join=True)
