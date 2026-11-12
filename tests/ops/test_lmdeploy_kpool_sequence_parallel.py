# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-GPU KPool gathering, causal ranges and local query ownership."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.ops.sparse_mla.kpool import _lmdeploy_kpool_is_available, kpool_topk_indices


def _sp_worker(rank: int, init_path: str) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"file://{init_path}")
    try:
        assert _lmdeploy_kpool_is_available(device) or _lmdeploy_kpool_is_available(device, bf16=True), (
            "Requires compatible LMDeploy KPool on SM90+"
        )
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
            expected = kpool_topk_indices(q, k, gates, weights, ape, full_ctx, **arguments)
            actual = kpool_topk_indices(
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
    @pytest.mark.parametrize("bf16", ["0", "1"])
    def test_two_gpu_gathering_and_packed_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bf16: str
    ) -> None:
        monkeypatch.setenv("XTUNER_GLM53_KPOOL_BF16", bf16)
        _lmdeploy_kpool_is_available.cache_clear()
        if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
            pytest.skip("Two CUDA GPUs and NCCL required")
        pytest.importorskip("lmdeploy.pytorch.backends.cuda.kpool")
        if not _lmdeploy_kpool_is_available(torch.device("cuda", 0), bf16=bf16 == "1"):
            pytest.skip("Requires compatible LMDeploy/DeepGEMM and SM90+")
        mp.spawn(_sp_worker, args=(str(tmp_path / "nccl_init"),), nprocs=2, join=True)
