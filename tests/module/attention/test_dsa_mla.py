"""DSA MLA 的算子、attention、checkpoint 与序列并行行为测试。

TestTorchSparseMLA
    test_padded_indices_support_int32_and_backward: PyTorch 后端处理 padding、int32 和反向传播。
TestDSAAttention
    test_packed_inputs_respect_causal_boundaries_and_backward: packed attention 遵守分段因果边界并可反传。
    test_shared_layers_reuse_topk_without_cross_context_leak: shared layer 复用当前样本 top-k 且不跨样本泄漏。
    test_reentrant_checkpoint_reuses_and_releases_topk: checkpoint 重算复用并最终释放 top-k。
TestAcceleratedSparseMLA
    test_tilelang_forward_backward_matches_torch: TileLang 前反向数值与 PyTorch 后端一致。
    test_compiled_cudnn_backward_matches_tilelang: 编译后的 cuDNN DSA 前反向与 TileLang 一致。
TestDSASequenceParallel
    test_packed_attention_matches_full_sequence: SP2 的输出、top-k 和输入梯度与完整序列一致。
    test_tilelang_indexer_matches_torch: SP2 query shard 的 TileLang indexer 与 PyTorch 一致。
    test_cudnn_local_query_global_kv_matches_full_sequence: SP2 cuDNN DSA 与完整序列数值一致。
"""

import math
import subprocess
import sys
from functools import cache

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.utils import checkpoint_wrapper
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.attention.dsa_topk_sharing import register_dsa_topk_decoder_lifecycle_hooks
from xtuner.v1.ops.sparse_mla import dsa_topk_indices, sparse_mla
from xtuner.v1.utils.test_utils import init_data_mesh


BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
DKV_ATOL = 1e-1
DKV_RTOL = 1e-1
CUDNN_DQ_ATOL = 5e-2
CUDNN_DQ_RTOL = 5e-2


@cache
def _tilelang_sparse_mla_available() -> bool:
    if not torch.cuda.is_available():
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import tilelang"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


@cache
def _cudnn_dsa_sparse_mla_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return False
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cudnn.deepseek_sparse_attention.sparse_attention_backward import sparse_attention_backward_wrapper",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _sparse_indices(seq_len: int, topk: int) -> torch.Tensor:
    indices = torch.full((seq_len, 1, topk), -1, device="cuda", dtype=torch.int64)
    for token_idx in range(seq_len):
        valid = min(token_idx + 1, topk)
        indices[token_idx, 0, :valid] = torch.arange(token_idx + 1 - valid, token_idx + 1, device="cuda")
    return indices


def _tilelang_sparse_mla_inputs():
    torch.manual_seed(0)
    seq_len = 64
    q = torch.randn(seq_len, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    return q, kv, _sparse_indices(seq_len, topk=64)


def _cudnn_dsa_sparse_mla_inputs():
    torch.manual_seed(0)
    seq_len = 64
    q = torch.randn(seq_len, 64, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    return q, kv, _sparse_indices(seq_len, topk=64)


def _tiny_dsa_attention(
    indexer_types: list[str] | None = None,
    layer_idx: int = 0,
):
    return DSAMLAConfig(
        num_attention_heads=2,
        head_dim=2,
        kv_lora_rank=3,
        q_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        indexer_types=indexer_types,
        sparse_mla_backend="torch",
    ).build(hidden_size=4, layer_idx=layer_idx)


class _TinyDsaDecoderBlock(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.self_attn = attention
        register_dsa_topk_decoder_lifecycle_hooks(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        return self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )["projected_output"]


class TestTorchSparseMLA:
    def test_padded_indices_support_int32_and_backward(self):
        # 验证 PyTorch SparseMLA 忽略 -1 padding、接受 int32 索引并产生有限梯度。
        torch.manual_seed(0)
        q = torch.randn(4, 2, 6, requires_grad=True)
        kv = torch.randn(5, 1, 6, requires_grad=True)
        indices = torch.tensor(
            [
                [[0, -1, -1]],
                [[1, 0, -1]],
                [[2, 1, -1]],
                [[4, 3, 1]],
            ],
            dtype=torch.int64,
        )

        expected = sparse_mla(q.detach(), kv.detach(), indices, scaling=0.5, value_dim=4, backend="torch")
        actual = sparse_mla(q, kv, indices.to(torch.int32), scaling=0.5, value_dim=4, backend="torch")
        (actual.raw_output.square().mean() + actual.softmax_lse.mean()).backward()

        torch.testing.assert_close(actual.raw_output, expected.raw_output)
        torch.testing.assert_close(actual.softmax_lse, expected.softmax_lse)
        assert actual.raw_output.shape == (4, 2, 4)
        assert actual.softmax_lse.shape == (4, 2)
        assert torch.isfinite(q.grad).all()
        assert torch.isfinite(kv.grad).all()


class TestDSAAttention:
    def test_packed_inputs_respect_causal_boundaries_and_backward(self):
        # 验证 packed attention 不跨子序列取 key，并能对真实输入完成有限反向传播。
        torch.manual_seed(0)
        attention = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
        hidden_states = torch.randn(1, 5, 4, requires_grad=True)
        seq_ctx = SequenceContext.from_input_ids(
            (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])),
            device="cpu",
        )
        position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

        outputs = attention(hidden_states, position_embeddings, seq_ctx)
        outputs["projected_output"].square().mean().backward()

        assert outputs["projected_output"].shape == (1, 5, 4)
        assert outputs["raw_output"].shape == (1, 5, 6)
        assert torch.isfinite(outputs["projected_output"]).all()
        assert torch.isfinite(hidden_states.grad).all()
        topk = seq_ctx.dsa_topk_cache.indices[0]
        for token_idx, seq_start in [(0, 0), (1, 0), (2, 2), (3, 2), (4, 2)]:
            valid_indices = topk[token_idx, 0][topk[token_idx, 0] != -1]
            assert valid_indices.numel() == token_idx - seq_start + 1
            assert valid_indices.min().item() >= seq_start
            assert valid_indices.max().item() <= token_idx

    def test_shared_layers_reuse_topk_without_cross_context_leak(self):
        # 验证 shared attention 复用同一 SequenceContext 的 source top-k，其他 context 保持独立。
        torch.manual_seed(0)
        source_attention = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0)
        shared_attention = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
        position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))
        hidden_states = torch.randn(1, 4, 4)
        seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")

        source_attention(hidden_states, position_embeddings, seq_ctx)
        source_topk = seq_ctx.dsa_topk_cache.indices[0]
        shared_output = shared_attention(hidden_states, position_embeddings, seq_ctx)["projected_output"]

        other_seq_ctx = SequenceContext.from_input_ids((torch.tensor([[5, 6, 7, 8]]),), device="cpu")
        source_attention(torch.randn(1, 4, 4), position_embeddings, other_seq_ctx)

        assert torch.isfinite(shared_output).all()
        assert seq_ctx.dsa_topk_cache.indices[0] is source_topk
        assert other_seq_ctx.dsa_topk_cache.indices[0] is not source_topk

    def test_reentrant_checkpoint_reuses_and_releases_topk(self):
        # 验证真实 source/shared decoder 经 reentrant checkpoint 重算后梯度有限且缓存释放。
        torch.manual_seed(0)
        source_block = checkpoint_wrapper(
            _TinyDsaDecoderBlock(_tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0)),
            checkpoint_impl=CheckpointImpl.REENTRANT,
        )
        shared_block = checkpoint_wrapper(
            _TinyDsaDecoderBlock(_tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)),
            checkpoint_impl=CheckpointImpl.REENTRANT,
        )
        hidden_states = torch.randn(1, 4, 4, requires_grad=True)
        position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))
        seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")

        output = source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        output = shared_block(output, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        output.square().mean().backward()

        assert torch.isfinite(hidden_states.grad).all()
        assert seq_ctx.dsa_topk_cache.indices == {}
        assert seq_ctx.dsa_topk_cache.offloaded == {}


class TestAcceleratedSparseMLA:
    @pytest.mark.skipif(
        not _tilelang_sparse_mla_available(),
        reason="requires CUDA and importable TileLang runtime",
    )
    def test_tilelang_forward_backward_matches_torch(self):
        # 验证 TileLang SparseMLA 的输出、LSE、dQ 和 dKV 与 PyTorch oracle 一致。
        q, kv, indices = _tilelang_sparse_mla_inputs()
        scaling = 1 / math.sqrt(q.shape[-1])
        q_ref = q.detach().clone().requires_grad_()
        kv_ref = kv.detach().clone().requires_grad_()
        q_tilelang = q.detach().clone().requires_grad_()
        kv_tilelang = kv.detach().clone().requires_grad_()

        expected = sparse_mla(q_ref, kv_ref, indices, scaling=scaling, value_dim=512, backend="torch")
        actual = sparse_mla(
            q_tilelang,
            kv_tilelang,
            indices.to(torch.int32),
            scaling=scaling,
            value_dim=512,
            backend="tilelang",
        )
        grad_output = torch.randn_like(expected.raw_output)
        expected.raw_output.backward(grad_output)
        actual.raw_output.backward(grad_output)

        torch.testing.assert_close(actual.raw_output, expected.raw_output, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(actual.softmax_lse, expected.softmax_lse, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(q_tilelang.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(kv_tilelang.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)

    @pytest.mark.skipif(
        not (_tilelang_sparse_mla_available() and _cudnn_dsa_sparse_mla_available()),
        reason="requires CUDA, TileLang, and cuDNN DSA sparse attention backward",
    )
    def test_compiled_cudnn_backward_matches_tilelang(self):
        # 验证 torch.compile 下 cuDNN DSA 的输出与梯度仍和 TileLang oracle 一致。
        q, kv, indices = _cudnn_dsa_sparse_mla_inputs()
        scaling = 1 / math.sqrt(q.shape[-1])

        def compiled_sparse_mla(q: torch.Tensor, kv: torch.Tensor, backend: str) -> torch.Tensor:
            return sparse_mla(
                q,
                kv,
                indices,
                scaling=scaling,
                value_dim=512,
                backend=backend,
            ).raw_output

        compiled_sparse_mla = torch.compile(compiled_sparse_mla, fullgraph=False)
        q_tilelang = q.detach().clone().requires_grad_()
        kv_tilelang = kv.detach().clone().requires_grad_()
        q_cudnn = q.detach().clone().requires_grad_()
        kv_cudnn = kv.detach().clone().requires_grad_()

        expected = compiled_sparse_mla(q_tilelang, kv_tilelang, "tilelang")
        actual = compiled_sparse_mla(q_cudnn, kv_cudnn, "cudnn_dsa")
        grad_output = torch.randn_like(expected)
        expected.backward(grad_output)
        actual.backward(grad_output)

        torch.testing.assert_close(actual, expected, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(q_cudnn.grad, q_tilelang.grad, atol=CUDNN_DQ_ATOL, rtol=CUDNN_DQ_RTOL)
        torch.testing.assert_close(kv_cudnn.grad, kv_tilelang.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


class TestDSASequenceParallel(DeterministicDDPTestCase):
    def test_packed_attention_matches_full_sequence(self):
        # 验证 SP2 packed attention 的输出、top-k 与输入梯度拼回后等同完整序列。
        self.create_pg("cuda")
        torch.manual_seed(7)
        attention = _tiny_dsa_attention().cuda()
        packed_input_ids = (
            torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
            torch.tensor([[6, 7, 8]], device="cuda"),
        )
        full_hidden_states = torch.randn(1, 8, 4, device="cuda", requires_grad=True)
        full_position_embeddings = (
            torch.randn(1, 8, 2, device="cuda"),
            torch.randn(1, 8, 2, device="cuda"),
        )
        full_output_grad = torch.randn(1, 8, 4, device="cuda")

        full_seq_ctx = SequenceContext.from_input_ids(packed_input_ids, device="cuda")
        expected_output = attention(
            full_hidden_states,
            position_embeddings=full_position_embeddings,
            seq_ctx=full_seq_ctx,
        )["projected_output"]
        expected_topk = full_seq_ctx.dsa_topk_cache.indices[0].clone()
        expected_output.backward(full_output_grad)
        expected_input_grad = full_hidden_states.grad.clone()
        attention.zero_grad(set_to_none=True)

        sp_mesh = init_data_mesh("cuda", sp_size=2)["sp"]
        sp_seq_ctx = SequenceContext.from_input_ids(packed_input_ids, device="cuda").split(sp_mesh)
        shard_size = 4
        shard_start = sp_seq_ctx.sp_rank * shard_size
        shard_end = shard_start + shard_size
        local_hidden_states = full_hidden_states.detach()[:, shard_start:shard_end].clone().requires_grad_()
        local_output = attention(
            local_hidden_states,
            position_embeddings=tuple(x[:, shard_start:shard_end] for x in full_position_embeddings),
            seq_ctx=sp_seq_ctx,
        )["projected_output"]
        local_topk = sp_seq_ctx.dsa_topk_cache.indices[0]
        local_output.backward(full_output_grad[:, shard_start:shard_end])

        gathered_output = [torch.empty_like(local_output) for _ in range(2)]
        gathered_topk = [torch.empty_like(local_topk) for _ in range(2)]
        gathered_input_grad = [torch.empty_like(local_hidden_states.grad) for _ in range(2)]
        sp_group = sp_mesh.get_group()
        dist.all_gather(gathered_output, local_output, group=sp_group)
        dist.all_gather(gathered_topk, local_topk, group=sp_group)
        dist.all_gather(gathered_input_grad, local_hidden_states.grad, group=sp_group)

        torch.testing.assert_close(torch.cat(gathered_output, dim=1), expected_output)
        torch.testing.assert_close(torch.cat(gathered_topk, dim=0), expected_topk)
        torch.testing.assert_close(torch.cat(gathered_input_grad, dim=1), expected_input_grad)

    @pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and TileLang")
    def test_tilelang_indexer_matches_torch(self):
        # 验证 SP2 本地 query 配合全局 key 时，TileLang indexer 与 PyTorch oracle 一致。
        self.create_pg("cuda")
        torch.manual_seed(11)
        packed_input_ids = (
            torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
            torch.tensor([[6, 7, 8]], device="cuda"),
        )
        full_q = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)
        full_k = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
        full_weights = torch.randn(1, 8, 4, device="cuda", dtype=torch.float32)
        sp_mesh = init_data_mesh("cuda", sp_size=2)["sp"]
        seq_ctx = SequenceContext.from_input_ids(packed_input_ids, device="cuda").split(sp_mesh)
        shard_start = seq_ctx.sp_rank * 4
        shard_end = shard_start + 4

        expected = dsa_topk_indices(
            full_q[:, shard_start:shard_end],
            full_k,
            full_weights[:, shard_start:shard_end],
            seq_ctx,
            index_head_dim=128,
            index_topk=4,
            backend="torch",
        )
        actual = dsa_topk_indices(
            full_q[:, shard_start:shard_end],
            full_k,
            full_weights[:, shard_start:shard_end],
            seq_ctx,
            index_head_dim=128,
            index_topk=4,
            backend="tilelang",
        )

        torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(
        not (_tilelang_sparse_mla_available() and _cudnn_dsa_sparse_mla_available()),
        reason="requires CUDA, TileLang, and cuDNN DSA",
    )
    def test_cudnn_local_query_global_kv_matches_full_sequence(self):
        # 验证 SP2 cuDNN DSA 的本地 query 输出、dQ 及聚合 dKV 与完整序列 oracle 一致。
        self.create_pg("cuda")
        q, kv, indices = _cudnn_dsa_sparse_mla_inputs()
        full_q = q.detach().clone().requires_grad_()
        full_kv = kv.detach().clone().requires_grad_()
        expected = sparse_mla(full_q, full_kv, indices, scaling=0.2, backend="tilelang").raw_output
        output_grad = torch.randn_like(expected)
        expected.backward(output_grad)

        sp_mesh = init_data_mesh("cuda", sp_size=2)["sp"]
        shard_start = sp_mesh.get_local_rank() * 32
        shard_end = shard_start + 32
        local_q = q[shard_start:shard_end].detach().clone().requires_grad_()
        global_kv = kv.detach().clone().requires_grad_()
        actual = sparse_mla(
            local_q,
            global_kv,
            indices[shard_start:shard_end],
            scaling=0.2,
            backend="cudnn_dsa",
        ).raw_output
        actual.backward(output_grad[shard_start:shard_end])
        dist.all_reduce(global_kv.grad, group=sp_mesh.get_group())

        torch.testing.assert_close(actual, expected[shard_start:shard_end], atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(
            local_q.grad,
            full_q.grad[shard_start:shard_end],
            atol=CUDNN_DQ_ATOL,
            rtol=CUDNN_DQ_RTOL,
        )
        actual_dkv = global_kv.grad.float()
        expected_dkv = full_kv.grad.float()
        relative_error = (actual_dkv - expected_dkv).norm() / expected_dkv.norm().clamp_min(1e-12)
        cosine_similarity = torch.nn.functional.cosine_similarity(
            actual_dkv.flatten(),
            expected_dkv.flatten(),
            dim=0,
        )
        self.assertLess(float(relative_error), 1e-2)
        self.assertGreater(float(cosine_similarity), 0.9999)

    @property
    def world_size(self) -> int:
        return 2
