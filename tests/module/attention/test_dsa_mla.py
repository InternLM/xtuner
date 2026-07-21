import math
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.utils import checkpoint_wrapper
from xtuner.v1.module.attention import DSAMLAConfig, dsa_mla
from xtuner.v1.module.attention.dsa_topk_sharing import (
    before_dsa_topk_decoder_forward,
    get_dsa_topk_sharing_runtime,
    register_dsa_topk_decoder_lifecycle_hooks,
)
from xtuner.v1.ops.sparse_mla import dsa_topk_indices, sparse_mla, torch_sparse_mla
from xtuner.v1.utils.test_utils import init_data_mesh


BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
DKV_ATOL = 1e-1
DKV_RTOL = 1e-1
CUDNN_DQ_ATOL = 5e-2
CUDNN_DQ_RTOL = 5e-2


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


def _cudnn_dsa_sparse_mla_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
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


def _tilelang_sparse_mla_inputs():
    torch.manual_seed(0)
    seq_len = 64
    topk = 64
    q = torch.randn(seq_len, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((seq_len, 1, topk), -1, device="cuda", dtype=torch.int64)
    for token_idx in range(seq_len):
        indices[token_idx, 0, : token_idx + 1] = torch.arange(token_idx + 1, device="cuda")
    return q, kv, indices


def _cudnn_dsa_sparse_mla_inputs():
    torch.manual_seed(0)
    seq_len = 64
    topk = 64
    q = torch.randn(seq_len, 64, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.full((seq_len, 1, topk), -1, device="cuda", dtype=torch.int64)
    for token_idx in range(seq_len):
        indices[token_idx, 0, : token_idx + 1] = torch.arange(token_idx + 1, device="cuda")
    return q, kv, indices


def _cat_microbatch_topk_indices(seq_len: int, topk: int, device: str):
    local_rows = torch.arange(seq_len, device=device, dtype=torch.int64).view(seq_len, 1)
    topk_offsets = torch.arange(topk, device=device, dtype=torch.int64).view(1, topk)
    local_topk = local_rows - topk_offsets
    first = local_topk.clamp_min(-1)
    second = torch.where(local_topk >= 0, local_topk + seq_len, torch.full_like(local_topk, -1))
    return torch.cat([first, second], dim=0).unsqueeze(1).contiguous()


def _tiny_dsa_attention(
    indexer_types: list[str] | None = None,
    layer_idx: int = 0,
    sparse_mla_backend: str = "torch",
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
        sparse_mla_backend=sparse_mla_backend,
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
        outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        return outputs["projected_output"]


class TestDSASequenceParallel(DeterministicDDPTestCase):
    def test_packed_forward_topk_and_input_gradient_match_full_sequence(self):
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

        for sp_size in (2, 4, 8):
            with self.subTest(sp_size=sp_size):
                sp_mesh = init_data_mesh("cuda", sp_size=sp_size)["sp"]
                sp_seq_ctx = SequenceContext.from_input_ids(packed_input_ids, device="cuda").split(sp_mesh)
                assert sp_seq_ctx.input_ids is not None
                shard_size = sp_seq_ctx.input_ids.shape[1]
                shard_start = sp_seq_ctx.sp_rank * shard_size
                shard_end = shard_start + shard_size
                local_hidden_states = full_hidden_states.detach()[:, shard_start:shard_end].clone().requires_grad_()
                local_position_embeddings = tuple(x[:, shard_start:shard_end] for x in full_position_embeddings)
                local_output_grad = full_output_grad[:, shard_start:shard_end]

                local_output = attention(
                    local_hidden_states,
                    position_embeddings=local_position_embeddings,
                    seq_ctx=sp_seq_ctx,
                )["projected_output"]
                local_topk = sp_seq_ctx.dsa_topk_cache.indices[0]
                local_output.backward(local_output_grad)

                gathered_output = [torch.empty_like(local_output) for _ in range(sp_size)]
                gathered_topk = [torch.empty_like(local_topk) for _ in range(sp_size)]
                gathered_input_grad = [torch.empty_like(local_hidden_states.grad) for _ in range(sp_size)]
                sp_group = sp_mesh.get_group()
                dist.all_gather(gathered_output, local_output, group=sp_group)
                dist.all_gather(gathered_topk, local_topk, group=sp_group)
                dist.all_gather(gathered_input_grad, local_hidden_states.grad, group=sp_group)

                torch.testing.assert_close(torch.cat(gathered_output, dim=1), expected_output)
                torch.testing.assert_close(torch.cat(gathered_topk, dim=0), expected_topk)
                torch.testing.assert_close(torch.cat(gathered_input_grad, dim=1), expected_input_grad)
                attention.zero_grad(set_to_none=True)

    @pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and TileLang")
    def test_tilelang_indexer_matches_torch_for_packed_query_shards(self):
        self.create_pg("cuda")
        torch.manual_seed(11)

        packed_input_ids = (
            torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
            torch.tensor([[6, 7, 8]], device="cuda"),
        )
        full_q = torch.randn(1, 8, 4, 128, device="cuda", dtype=torch.bfloat16)
        full_k = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
        full_weights = torch.randn(1, 8, 4, device="cuda", dtype=torch.float32)

        for sp_size in (2, 4, 8):
            with self.subTest(sp_size=sp_size):
                sp_mesh = init_data_mesh("cuda", sp_size=sp_size)["sp"]
                seq_ctx = SequenceContext.from_input_ids(packed_input_ids, device="cuda").split(sp_mesh)
                assert seq_ctx.input_ids is not None
                shard_size = seq_ctx.input_ids.shape[1]
                shard_start = seq_ctx.sp_rank * shard_size
                shard_end = shard_start + shard_size
                local_q = full_q[:, shard_start:shard_end]
                local_weights = full_weights[:, shard_start:shard_end]

                expected = dsa_topk_indices(
                    local_q,
                    full_k,
                    local_weights,
                    seq_ctx,
                    index_head_dim=128,
                    index_topk=4,
                    backend="torch",
                )
                actual = dsa_topk_indices(
                    local_q,
                    full_k,
                    local_weights,
                    seq_ctx,
                    index_head_dim=128,
                    index_topk=4,
                    backend="tilelang",
                )

                torch.testing.assert_close(actual, expected)

    @pytest.mark.skipif(not _cudnn_dsa_sparse_mla_available(), reason="requires CUDA and cuDNN DSA")
    def test_cudnn_dsa_local_query_global_kv_matches_full_sequence(self):
        self.create_pg("cuda")
        q, kv, indices = _cudnn_dsa_sparse_mla_inputs()

        full_q = q.detach().clone().requires_grad_()
        full_kv = kv.detach().clone().requires_grad_()
        expected = sparse_mla(full_q, full_kv, indices, scaling=0.2, backend="tilelang").raw_output
        output_grad = torch.randn_like(expected)
        expected.backward(output_grad)

        for sp_size in (2, 4, 8):
            with self.subTest(sp_size=sp_size):
                sp_mesh = init_data_mesh("cuda", sp_size=sp_size)["sp"]
                shard_size = q.shape[0] // sp_size
                shard_start = sp_mesh.get_local_rank() * shard_size
                shard_end = shard_start + shard_size
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
                # The full TileLang kernel and the sharded cuDNN kernels followed
                # by a BF16 NCCL sum accumulate dKV in different orders.
                actual_dkv = global_kv.grad.float()
                expected_dkv = full_kv.grad.float()
                relative_error = (actual_dkv - expected_dkv).norm() / expected_dkv.norm().clamp_min(1e-12)
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    actual_dkv.flatten(), expected_dkv.flatten(), dim=0
                )
                self.assertLess(float(relative_error), 1e-2)
                self.assertGreater(float(cosine_similarity), 0.9999)

    @property
    def world_size(self) -> int:
        return 8


def test_torch_sparse_mla_handles_invalid_indices_and_backward():
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

    out, lse = torch_sparse_mla(q, kv, indices, scaling=0.5, value_dim=4)
    loss = out.square().mean() + lse.mean()
    loss.backward()

    assert out.shape == (4, 2, 4)
    assert lse.shape == (4, 2)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    assert q.grad is not None
    assert kv.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(kv.grad).all()


def test_torch_sparse_mla_accepts_int32_indices_like_int64():
    torch.manual_seed(0)
    q = torch.randn(4, 2, 6)
    kv = torch.randn(5, 1, 6)
    indices = torch.tensor(
        [
            [[0, -1, -1]],
            [[1, 0, -1]],
            [[2, 1, -1]],
            [[4, 3, 1]],
        ],
        dtype=torch.int64,
    )

    int64_outputs = torch_sparse_mla(q, kv, indices, scaling=0.5, value_dim=4)
    int32_outputs = torch_sparse_mla(q, kv, indices.to(torch.int32), scaling=0.5, value_dim=4)

    torch.testing.assert_close(int32_outputs.raw_output, int64_outputs.raw_output)
    torch.testing.assert_close(int32_outputs.softmax_lse, int64_outputs.softmax_lse)


def test_sparse_mla_selects_torch_backend_explicitly():
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

    outputs = sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="torch")
    out = outputs.raw_output
    lse = outputs.softmax_lse
    loss = out.square().mean() + lse.mean()
    loss.backward()

    assert out.shape == (4, 2, 4)
    assert lse.shape == (4, 2)
    assert q.grad is not None
    assert kv.grad is not None


def test_sparse_mla_tilelang_backend_reports_unsupported_device_clearly():
    q = torch.randn(4, 2, 6)
    kv = torch.randn(5, 1, 6)
    indices = torch.zeros(4, 1, 4, dtype=torch.int64)

    with pytest.raises(RuntimeError, match="requires q, kv, and indices to be CUDA tensors"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="tilelang")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_mla_tilelang_backend_reports_unsupported_topk_clearly():
    q = torch.randn(64, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(64, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(64, 1, 63, device="cuda", dtype=torch.int64)

    with pytest.raises(RuntimeError, match="requires topk to be divisible by 64"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=512, backend="tilelang")


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_sparse_mla_tilelang_forward_matches_torch_sparse_mla():
    q, kv, indices = _tilelang_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])

    ref_out, ref_lse = sparse_mla(q, kv, indices, scaling=scaling, value_dim=512, backend="torch")
    tilelang_out, tilelang_lse = sparse_mla(
        q,
        kv,
        indices.to(torch.int32),
        scaling=scaling,
        value_dim=512,
        backend="tilelang",
    )

    torch.testing.assert_close(tilelang_out, ref_out, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(tilelang_lse, ref_lse, atol=BF16_ATOL, rtol=BF16_RTOL)


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_sparse_mla_tilelang_backward_matches_torch_sparse_mla():
    q, kv, indices = _tilelang_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])
    q_ref = q.detach().clone().requires_grad_()
    kv_ref = kv.detach().clone().requires_grad_()
    q_tilelang = q.detach().clone().requires_grad_()
    kv_tilelang = kv.detach().clone().requires_grad_()

    ref_out, _ = sparse_mla(q_ref, kv_ref, indices, scaling=scaling, value_dim=512, backend="torch")
    tilelang_out, _ = sparse_mla(q_tilelang, kv_tilelang, indices, scaling=scaling, value_dim=512, backend="tilelang")
    grad_output = torch.randn_like(ref_out)

    ref_out.backward(grad_output)
    tilelang_out.backward(grad_output)

    torch.testing.assert_close(q_tilelang.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
    # dKV accumulates duplicate sparse indices with fp32 atomics before casting to bf16;
    # PyTorch fallback uses a deterministic reduction order, so its bf16 rounding differs.
    torch.testing.assert_close(kv_tilelang.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


@pytest.mark.skipif(
    not (_tilelang_sparse_mla_available() and _cudnn_dsa_sparse_mla_available()),
    reason="requires CUDA, TileLang, and cuDNN DSA sparse attention backward",
)
def test_sparse_mla_cudnn_dsa_backward_matches_tilelang_sparse_mla():
    q, kv, indices = _cudnn_dsa_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])
    q_tilelang = q.detach().clone().requires_grad_()
    kv_tilelang = kv.detach().clone().requires_grad_()
    q_cudnn = q.detach().clone().requires_grad_()
    kv_cudnn = kv.detach().clone().requires_grad_()

    tilelang_out, _ = sparse_mla(q_tilelang, kv_tilelang, indices, scaling=scaling, value_dim=512, backend="tilelang")
    cudnn_out, _ = sparse_mla(q_cudnn, kv_cudnn, indices, scaling=scaling, value_dim=512, backend="cudnn_dsa")
    grad_output = torch.randn_like(tilelang_out)

    tilelang_out.backward(grad_output)
    cudnn_out.backward(grad_output)

    torch.testing.assert_close(cudnn_out, tilelang_out, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(q_cudnn.grad, q_tilelang.grad, atol=CUDNN_DQ_ATOL, rtol=CUDNN_DQ_RTOL)
    torch.testing.assert_close(kv_cudnn.grad, kv_tilelang.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


@pytest.mark.skipif(
    not (_tilelang_sparse_mla_available() and _cudnn_dsa_sparse_mla_available()),
    reason="requires CUDA, TileLang, and cuDNN DSA sparse attention backward",
)
def test_sparse_mla_cudnn_dsa_compile_backward_matches_tilelang_sparse_mla():
    q, kv, indices = _cudnn_dsa_sparse_mla_inputs()
    scaling = 1 / math.sqrt(q.shape[-1])

    def compiled_sparse_mla(q: torch.Tensor, kv: torch.Tensor, backend: str) -> torch.Tensor:
        out, _ = sparse_mla(q, kv, indices, scaling=scaling, value_dim=512, backend=backend)
        return out

    compiled_sparse_mla = torch.compile(compiled_sparse_mla, fullgraph=False)
    q_tilelang = q.detach().clone().requires_grad_()
    kv_tilelang = kv.detach().clone().requires_grad_()
    q_cudnn = q.detach().clone().requires_grad_()
    kv_cudnn = kv.detach().clone().requires_grad_()

    tilelang_out = compiled_sparse_mla(q_tilelang, kv_tilelang, "tilelang")
    cudnn_out = compiled_sparse_mla(q_cudnn, kv_cudnn, "cudnn_dsa")
    grad_output = torch.randn_like(tilelang_out)

    tilelang_out.backward(grad_output)
    cudnn_out.backward(grad_output)

    torch.testing.assert_close(cudnn_out, tilelang_out, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(q_cudnn.grad, q_tilelang.grad, atol=CUDNN_DQ_ATOL, rtol=CUDNN_DQ_RTOL)
    torch.testing.assert_close(kv_cudnn.grad, kv_tilelang.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


@pytest.mark.skipif(
    not (_tilelang_sparse_mla_available() and _cudnn_dsa_sparse_mla_available()),
    reason="requires CUDA, TileLang, and cuDNN DSA sparse attention backward",
)
def test_sparse_mla_cudnn_dsa_compile_backward_matches_after_microbatch_topk_split():
    seq_len = 64
    topk = 64
    seq_ctx_list = [
        SequenceContext.from_input_ids((torch.arange(seq_len, device="cuda").view(1, -1),), device="cuda"),
        SequenceContext.from_input_ids(
            (torch.arange(seq_len, 2 * seq_len, device="cuda").view(1, -1),),
            device="cuda",
        ),
    ]
    cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
    cat_seq_ctx.dsa_topk_cache.indices[0] = _cat_microbatch_topk_indices(seq_len, topk, "cuda")
    cat_seq_ctx.split_dsa_topk_indices_to(seq_ctx_list)
    indices = seq_ctx_list[1].dsa_topk_cache.indices[0]
    valid_indices = indices[indices != -1]
    assert valid_indices.max().item() < seq_len

    q, kv, _ = _cudnn_dsa_sparse_mla_inputs()
    scaling = 0.0625

    def compiled_sparse_mla(q: torch.Tensor, kv: torch.Tensor, backend: str) -> torch.Tensor:
        out, _ = sparse_mla(q, kv, indices, scaling=scaling, value_dim=512, backend=backend)
        return out

    compiled_sparse_mla = torch.compile(compiled_sparse_mla, fullgraph=False)
    q_tilelang = q.detach().clone().requires_grad_()
    kv_tilelang = kv.detach().clone().requires_grad_()
    q_cudnn = q.detach().clone().requires_grad_()
    kv_cudnn = kv.detach().clone().requires_grad_()

    tilelang_out = compiled_sparse_mla(q_tilelang, kv_tilelang, "tilelang")
    cudnn_out = compiled_sparse_mla(q_cudnn, kv_cudnn, "cudnn_dsa")
    grad_output = torch.randn_like(tilelang_out)

    tilelang_out.backward(grad_output)
    cudnn_out.backward(grad_output)

    assert torch.isfinite(q_cudnn.grad).all()
    assert torch.isfinite(kv_cudnn.grad).all()
    torch.testing.assert_close(cudnn_out, tilelang_out, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(q_cudnn.grad, q_tilelang.grad, atol=CUDNN_DQ_ATOL, rtol=CUDNN_DQ_RTOL)
    torch.testing.assert_close(kv_cudnn.grad, kv_tilelang.grad, atol=DKV_ATOL, rtol=DKV_RTOL)


def test_dsa_attention_topk_respects_packed_causal_boundaries():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
    hidden_states = torch.randn(1, 5, 4)
    seq_ctx = SequenceContext.from_input_ids(
        (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])),
        device="cpu",
    )
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    attn(hidden_states, position_embeddings, seq_ctx)

    topk = seq_ctx.dsa_topk_cache.indices[0]
    assert topk.dtype == torch.int64
    for token_idx, seq_start in [(0, 0), (1, 0), (2, 2), (3, 2), (4, 2)]:
        valid_indices = topk[token_idx, 0][topk[token_idx, 0] != -1]
        assert valid_indices.numel() == token_idx - seq_start + 1
        assert valid_indices.min().item() >= seq_start
        assert valid_indices.max().item() <= token_idx


def test_dsa_attention_forward_backward_on_packed_inputs():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
    hidden_states = torch.randn(1, 5, 4, requires_grad=True)
    seq_ctx = SequenceContext.from_input_ids(
        (torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])),
        device="cpu",
    )
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    attn_outputs = attn(hidden_states, position_embeddings, seq_ctx)
    loss = attn_outputs["projected_output"].square().mean()
    loss.backward()

    assert attn_outputs["projected_output"].shape == (1, 5, 4)
    assert attn_outputs["raw_output"].shape == (1, 5, 6)
    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_dsa_attention_selects_tilelang_sparse_mla_backend_explicitly():
    torch.manual_seed(0)
    attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0, sparse_mla_backend="tilelang")
    hidden_states = torch.randn(1, 5, 4)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4, 5]]),), device="cpu")
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    with pytest.raises(RuntimeError, match="TileLang"):
        attn(hidden_states, position_embeddings, seq_ctx)


def test_dsa_attention_tilelang_runtime_checked_once_at_build(monkeypatch):
    calls = 0

    def fake_ensure_tilelang_runtime_available():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(dsa_mla, "ensure_tilelang_runtime_available", fake_ensure_tilelang_runtime_available)
    _tiny_dsa_attention(indexer_types=["full"], layer_idx=0, sparse_mla_backend="tilelang")

    q = torch.randn(4, 2, 6)
    kv = torch.randn(5, 1, 6)
    indices = torch.zeros(4, 1, 4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires q, kv, and indices to be CUDA tensors"):
        sparse_mla(q, kv, indices, scaling=0.5, value_dim=4, backend="tilelang")

    assert calls == 1


def test_dsa_attention_cudnn_dsa_runtime_checked_once_at_build(monkeypatch):
    tilelang_calls = 0
    cudnn_calls = 0

    def fake_ensure_tilelang_runtime_available():
        nonlocal tilelang_calls
        tilelang_calls += 1

    def fake_ensure_cudnn_dsa_runtime_available():
        nonlocal cudnn_calls
        cudnn_calls += 1

    monkeypatch.setattr(dsa_mla, "ensure_tilelang_runtime_available", fake_ensure_tilelang_runtime_available)
    monkeypatch.setattr(dsa_mla, "ensure_cudnn_dsa_runtime_available", fake_ensure_cudnn_dsa_runtime_available)
    _tiny_dsa_attention(indexer_types=["full"], layer_idx=0, sparse_mla_backend="cudnn_dsa")

    assert tilelang_calls == 1
    assert cudnn_calls == 1


@pytest.mark.skipif(not _tilelang_sparse_mla_available(), reason="requires CUDA and importable TileLang runtime")
def test_dsa_attention_tilelang_long_packed_sequence_respects_boundaries_and_backward():
    torch.manual_seed(0)
    seq_lens = [128, 160, 224]
    seq_len = sum(seq_lens)
    attn = (
        DSAMLAConfig(
            num_attention_heads=16,
            head_dim=64,
            kv_lora_rank=512,
            q_lora_rank=64,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            index_topk=64,
            index_head_dim=128,
            index_n_heads=4,
            indexer_types=["full"],
            sparse_mla_backend="tilelang",
        )
        .build(hidden_size=64, layer_idx=0)
        .cuda()
        .bfloat16()
    )
    input_ids = []
    offset = 0
    for length in seq_lens:
        input_ids.append(torch.arange(offset, offset + length).view(1, -1))
        offset += length
    hidden_states = torch.randn(1, seq_len, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    position_embeddings = (
        torch.ones(1, seq_len, 64, device="cuda", dtype=torch.bfloat16),
        torch.zeros(1, seq_len, 64, device="cuda", dtype=torch.bfloat16),
    )
    seq_ctx = SequenceContext.from_input_ids(input_ids, device="cuda")

    attn_outputs = attn(hidden_states, position_embeddings, seq_ctx)
    loss = attn_outputs["projected_output"].float().square().mean()
    loss.backward()

    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert torch.isfinite(attn_outputs["raw_output"]).all()
    assert torch.isfinite(attn_outputs["softmax_lse"]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()

    topk = seq_ctx.dsa_topk_cache.indices[0]
    assert topk.dtype == torch.int64
    for seq_start, seq_end in zip(seq_ctx.cu_seq_lens_q[:-1].tolist(), seq_ctx.cu_seq_lens_q[1:].tolist()):
        for token_idx in range(seq_start, seq_end):
            valid_indices = topk[token_idx, 0][topk[token_idx, 0] != -1]
            assert valid_indices.numel() == min(64, token_idx - seq_start + 1)
            assert valid_indices.min().item() >= seq_start
            assert valid_indices.max().item() <= token_idx


def test_dsa_attention_shares_topk_within_microbatch_only():
    torch.manual_seed(0)
    full_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0)
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    hidden_states = torch.randn(1, 4, 4)
    full_attn(hidden_states, position_embeddings, seq_ctx)
    source_topk = seq_ctx.dsa_topk_cache.indices[0]

    shared_attn(hidden_states, position_embeddings, seq_ctx)

    assert set(seq_ctx.dsa_topk_cache.indices) == {0}
    assert seq_ctx.dsa_topk_cache.indices[0] is source_topk

    other_seq_ctx = SequenceContext.from_input_ids((torch.tensor([[5, 6, 7, 8]]),), device="cpu")
    full_attn(torch.randn(1, 4, 4), position_embeddings, other_seq_ctx)

    assert other_seq_ctx.dsa_topk_cache.indices is not seq_ctx.dsa_topk_cache.indices
    assert set(other_seq_ctx.dsa_topk_cache.indices) == {0}


def test_sequence_context_splits_cat_dsa_topk_cache_to_microbatches():
    seq_len0 = 2
    seq_len1 = 3
    seq_ctx_list = [
        SequenceContext.from_input_ids((torch.arange(seq_len0).view(1, -1),), device="cpu"),
        SequenceContext.from_input_ids((torch.arange(seq_len0, seq_len0 + seq_len1).view(1, -1),), device="cpu"),
    ]
    assert seq_ctx_list[0].dsa_topk_cache.indices == {}
    assert seq_ctx_list[0].dsa_topk_cache.indices is not seq_ctx_list[1].dsa_topk_cache.indices

    cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
    layer0_topk = torch.tensor(
        [
            [[0, -1, -1, -1]],
            [[1, 0, -1, -1]],
            [[2, -1, -1, -1]],
            [[3, 2, -1, -1]],
            [[4, 3, 2, -1]],
        ],
        dtype=torch.int64,
    )
    layer2_topk = layer0_topk.clone()
    cat_seq_ctx.dsa_topk_cache.indices[0] = layer0_topk
    cat_seq_ctx.dsa_topk_cache.indices[2] = layer2_topk

    cat_seq_ctx.split_dsa_topk_indices_to(seq_ctx_list)

    assert set(seq_ctx_list[0].dsa_topk_cache.indices) == {0, 2}
    assert set(seq_ctx_list[1].dsa_topk_cache.indices) == {0, 2}
    torch.testing.assert_close(seq_ctx_list[0].dsa_topk_cache.indices[0], layer0_topk[:2])
    torch.testing.assert_close(
        seq_ctx_list[1].dsa_topk_cache.indices[0],
        torch.tensor(
            [
                [[0, -1, -1, -1]],
                [[1, 0, -1, -1]],
                [[2, 1, 0, -1]],
            ],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(seq_ctx_list[0].dsa_topk_cache.indices[2], layer2_topk[:2])
    torch.testing.assert_close(
        seq_ctx_list[1].dsa_topk_cache.indices[2],
        torch.tensor(
            [
                [[0, -1, -1, -1]],
                [[1, 0, -1, -1]],
                [[2, 1, 0, -1]],
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DSA top-k offload")
def test_dsa_topk_offload_keeps_cat_cache_after_microbatch_split(monkeypatch):
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    seq_len0 = 2
    seq_len1 = 3
    seq_ctx_list = [
        SequenceContext.from_input_ids((torch.arange(seq_len0, device="cuda").view(1, -1),), device="cuda"),
        SequenceContext.from_input_ids(
            (torch.arange(seq_len0, seq_len0 + seq_len1, device="cuda").view(1, -1),),
            device="cuda",
        ),
    ]
    cat_seq_ctx = SequenceContext.cat(seq_ctx_list)
    topk = torch.tensor(
        [
            [[0, -1, -1, -1]],
            [[1, 0, -1, -1]],
            [[2, -1, -1, -1]],
            [[3, 2, -1, -1]],
            [[4, 3, 2, -1]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    expected_topk = topk.clone()
    cat_seq_ctx.dsa_topk_cache.indices[0] = topk
    cat_seq_ctx.split_dsa_topk_indices_to(seq_ctx_list)

    runtime = get_dsa_topk_sharing_runtime()
    source_attn = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0).cuda()
    with torch.no_grad():
        runtime.after_sparse_mla_use(layer=source_attn, seq_ctx=seq_ctx_list[0])
    torch.cuda.synchronize()

    torch.testing.assert_close(cat_seq_ctx.dsa_topk_cache.indices[0], expected_topk)
    runtime.after_sparse_mla_use(layer=source_attn, seq_ctx=seq_ctx_list[0])
    torch.cuda.synchronize()


def test_dsa_attention_mtp_physical_layer_uses_own_full_indexer():
    torch.manual_seed(0)
    mtp_attn = _tiny_dsa_attention(indexer_types=["full", "full", "full", "shared", "shared", "full"], layer_idx=5)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4, 5]]),), device="cpu")
    position_embeddings = (torch.ones(1, 5, 2), torch.zeros(1, 5, 2))

    attn_outputs = mtp_attn(torch.randn(1, 5, 4), position_embeddings, seq_ctx)

    assert torch.isfinite(attn_outputs["projected_output"]).all()
    assert mtp_attn.source_layer_idx == 5
    assert hasattr(mtp_attn, "indexer")
    assert set(seq_ctx.dsa_topk_cache.indices) == {5}


def test_dsa_attention_checkpoint_recompute_reuses_and_releases_source_topk():
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0)
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    source_block = _TinyDsaDecoderBlock(source_attn)
    shared_block = _TinyDsaDecoderBlock(shared_attn)
    hidden_states = torch.randn(1, 4, 4)
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)

    source_topk = seq_ctx.dsa_topk_cache.indices[0]
    assert seq_ctx.dsa_topk_cache.checkpoint_active

    recompute_hidden_states = hidden_states.detach().clone().requires_grad_()
    shared_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    assert seq_ctx.dsa_topk_cache.indices[0] is source_topk

    source_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {0}


def test_dsa_topk_mtp_iteration_recompute_releases_after_last_logical_depth():
    class FakeFullLayer:
        layer_idx = 3
        source_layer_idx = 3
        training = True
        indexer_types = ["full", "shared", "shared", "full"]
        index_skip_topk_offset = 0
        index_topk_freq = 1
        dsa_topk_last_use = {3: 3}
        dsa_topk_recompute_release = {3: 3}

    runtime = get_dsa_topk_sharing_runtime()
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    source_topk = torch.arange(4, dtype=torch.int64).view(4, 1, 1)
    compute_calls = 0

    def compute_source_topk():
        nonlocal compute_calls
        compute_calls += 1
        return source_topk

    runtime.register_mtp_iteration_topk_sharing(seq_ctx=seq_ctx, source_layer_idx=3, num_iterations=3)

    with torch.no_grad():
        for _ in range(3):
            assert (
                runtime.get_or_compute(
                    layer=FakeFullLayer(),
                    seq_ctx=seq_ctx,
                    compute_source_topk=compute_source_topk,
                )
                is source_topk
            )
            runtime.after_sparse_mla_use(layer=FakeFullLayer(), seq_ctx=seq_ctx)

    assert compute_calls == 1
    assert seq_ctx.dsa_topk_cache.checkpoint_active
    assert seq_ctx.dsa_topk_cache.indices[3] is source_topk

    for _ in range(2):
        assert (
            runtime.get_or_compute(
                layer=FakeFullLayer(),
                seq_ctx=seq_ctx,
                compute_source_topk=compute_source_topk,
            )
            is source_topk
        )
        runtime.after_sparse_mla_use(layer=FakeFullLayer(), seq_ctx=seq_ctx)
        assert seq_ctx.dsa_topk_cache.indices[3] is source_topk
        assert seq_ctx.dsa_topk_cache.released_sources == set()

    assert (
        runtime.get_or_compute(
            layer=FakeFullLayer(),
            seq_ctx=seq_ctx,
            compute_source_topk=compute_source_topk,
        )
        is source_topk
    )
    runtime.after_sparse_mla_use(layer=FakeFullLayer(), seq_ctx=seq_ctx)

    assert compute_calls == 1
    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {3}


def test_dsa_topk_mtp_iteration_checkpoint_reuses_forward_topk():
    torch.manual_seed(0)
    attention = _tiny_dsa_attention(indexer_types=["full"], layer_idx=0)
    attention.dsa_topk_last_use = {0: 0}
    attention.dsa_topk_recompute_release = {0: 0}
    indexer_calls = 0

    def count_indexer_calls(*_args):
        nonlocal indexer_calls
        indexer_calls += 1

    attention.indexer.register_forward_hook(count_indexer_calls)
    block = checkpoint_wrapper(
        _TinyDsaDecoderBlock(attention),
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    get_dsa_topk_sharing_runtime().register_mtp_iteration_topk_sharing(
        seq_ctx=seq_ctx,
        source_layer_idx=0,
        num_iterations=2,
    )
    hidden_states = torch.randn(1, 4, 4, requires_grad=True)
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    first_depth_ctx = seq_ctx.copy()
    second_depth_ctx = seq_ctx.copy()
    first_depth = block(hidden_states, position_embeddings=position_embeddings, seq_ctx=first_depth_ctx)
    second_depth = block(first_depth, position_embeddings=position_embeddings, seq_ctx=second_depth_ctx)
    second_depth.square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert indexer_calls == 1
    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DSA top-k offload")
def test_dsa_topk_mtp_iteration_sharing_with_topk_offload(monkeypatch):
    class FakeFullLayer:
        layer_idx = 3
        source_layer_idx = 3
        training = True
        indexer_types = ["full", "shared", "shared", "full"]
        index_skip_topk_offset = 0
        index_topk_freq = 1
        dsa_topk_last_use = {3: 3}
        dsa_topk_recompute_release = {3: 3}

    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    runtime = get_dsa_topk_sharing_runtime()

    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")
    source_topk = torch.arange(4, dtype=torch.int64, device="cuda").view(4, 1, 1)
    compute_calls = 0

    def compute_source_topk():
        nonlocal compute_calls
        compute_calls += 1
        return source_topk

    runtime.register_mtp_iteration_topk_sharing(seq_ctx=seq_ctx, source_layer_idx=3, num_iterations=3)
    layer = FakeFullLayer()

    # Reentrant checkpoint original forward runs under no_grad. With top-k
    # offload enabled, later MTP depths should read the first depth's top-k from
    # GPU residency instead of recomputing or transferring it between depths.
    with torch.no_grad():
        for iteration in range(3):
            runtime.before_layer_forward(layer=layer, seq_ctx=seq_ctx)
            runtime.get_or_compute(layer=layer, seq_ctx=seq_ctx, compute_source_topk=compute_source_topk)
            runtime.after_sparse_mla_use(layer=layer, seq_ctx=seq_ctx)
            if iteration < 2:
                assert set(seq_ctx.dsa_topk_cache.indices) == {3}
                assert seq_ctx.dsa_topk_cache.offloaded == {}

    assert compute_calls == 1
    assert seq_ctx.dsa_topk_cache.indices == {}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {3}

    for _ in range(2):
        runtime.before_layer_forward(layer=layer, seq_ctx=seq_ctx)
        runtime.get_or_compute(layer=layer, seq_ctx=seq_ctx, compute_source_topk=compute_source_topk)
        runtime.after_sparse_mla_use(layer=layer, seq_ctx=seq_ctx)
        assert set(seq_ctx.dsa_topk_cache.indices) == {3}
        assert seq_ctx.dsa_topk_cache.released_sources == set()

    runtime.before_layer_forward(layer=layer, seq_ctx=seq_ctx)
    runtime.get_or_compute(layer=layer, seq_ctx=seq_ctx, compute_source_topk=compute_source_topk)
    runtime.after_sparse_mla_use(layer=layer, seq_ctx=seq_ctx)

    assert compute_calls == 1
    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.offloaded == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {3}
    assert seq_ctx.dsa_topk_cache.mtp_forward_uses_remaining == {}
    assert seq_ctx.dsa_topk_cache.mtp_replays_remaining == {}
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA activation offload")
def test_dsa_attention_activation_offload_decoder_pre_hook_prefetches_without_sync_read(monkeypatch):
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "1")
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {0}

    runtime = get_dsa_topk_sharing_runtime()

    def fail_sync_read(*args, **kwargs):
        raise AssertionError("decoder pre-hook should prefetch top-k instead of synchronously reading it")

    monkeypatch.setattr(runtime._offloaded_residency, "_read_offloaded", fail_sync_read)

    before_dsa_topk_decoder_forward(shared_attn, seq_ctx)
    assert set(seq_ctx.dsa_topk_cache.indices) == {0}
    runtime._offloaded_residency.after_recompute_release(seq_ctx, 0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA activation offload")
def test_dsa_attention_topk_offload_waits_after_decoder_pre_hook_prefetch(monkeypatch):
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "1")
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    runtime = get_dsa_topk_sharing_runtime()
    before_dsa_topk_decoder_forward(shared_attn, seq_ctx)
    assert set(seq_ctx.dsa_topk_cache.indices) == {0}

    wait_calls = 0
    original_wait_prefetched = runtime._offloaded_residency._wait_prefetched

    def count_wait_prefetched(*args, **kwargs):
        nonlocal wait_calls
        wait_calls += 1
        return original_wait_prefetched(*args, **kwargs)

    monkeypatch.setattr(runtime._offloaded_residency, "_wait_prefetched", count_wait_prefetched)
    recompute_hidden_states = hidden_states.detach().clone().requires_grad_()
    shared_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)

    assert wait_calls == 1
    runtime._offloaded_residency.after_recompute_release(seq_ctx, 0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DSA top-k offload")
def test_dsa_attention_topk_offload_without_activation_offload(monkeypatch):
    monkeypatch.delenv("XTUNER_ACTIVATION_OFFLOAD", raising=False)
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {0}

    recompute_hidden_states = hidden_states.detach().clone().requires_grad_()
    shared_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    source_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.offloaded == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA activation offload")
def test_dsa_attention_activation_offload_keeps_topk_on_gpu_by_default(monkeypatch):
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "1")
    monkeypatch.delenv("XTUNER_DSA_TOPK_OFFLOAD", raising=False)
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert set(seq_ctx.dsa_topk_cache.indices) == {0}
    assert seq_ctx.dsa_topk_cache.offloaded == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA activation offload")
def test_dsa_attention_activation_offload_decoder_hooks_onload_and_clear_topk(monkeypatch):
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "1")
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")
    cache_during_shared_attention = []

    def record_cache_before_decoder_post_hook(_module, _args, _output):
        cache = seq_ctx.dsa_topk_cache
        cache_during_shared_attention.append((set(cache.indices), set(cache.offloaded)))

    shared_attn.register_forward_hook(record_cache_before_decoder_post_hook)

    with torch.no_grad():
        source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        source_topk = seq_ctx.dsa_topk_cache.indices[0].detach().cpu().clone()
        shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert cache_during_shared_attention == [({0}, set())]
    assert seq_ctx.dsa_topk_cache.indices == {}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {0}
    assert seq_ctx.dsa_topk_cache.offloaded[0] == f"dsa_topk_{seq_ctx.dsa_topk_cache.context_id}_0"

    recompute_hidden_states = hidden_states.detach().clone().requires_grad_()
    shared_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    assert set(seq_ctx.dsa_topk_cache.indices) == {0}
    torch.testing.assert_close(seq_ctx.dsa_topk_cache.indices[0].cpu(), source_topk)

    source_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.offloaded == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA activation offload")
def test_dsa_attention_activation_offload_compile_keeps_pin_memory_out_of_graph(monkeypatch):
    monkeypatch.setenv("XTUNER_ACTIVATION_OFFLOAD", "1")
    monkeypatch.setenv("XTUNER_DSA_TOPK_OFFLOAD", "1")
    torch.manual_seed(0)
    source_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=0).cuda()
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1).cuda()
    compiled_source_block = _TinyDsaDecoderBlock(source_attn).cuda()
    compiled_shared_block = _TinyDsaDecoderBlock(shared_attn).cuda()
    compiled_source_block.forward = torch.compile(compiled_source_block.forward, fullgraph=False)
    compiled_shared_block.forward = torch.compile(compiled_shared_block.forward, fullgraph=False)
    hidden_states = torch.randn(1, 4, 4, device="cuda")
    position_embeddings = (
        torch.ones(1, 4, 2, device="cuda"),
        torch.zeros(1, 4, 2, device="cuda"),
    )
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cuda")

    with torch.no_grad():
        compiled_source_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
        compiled_shared_block(hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {0}

    recompute_hidden_states = hidden_states.detach().clone().requires_grad_()
    compiled_shared_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    assert set(seq_ctx.dsa_topk_cache.indices) == {0}
    assert set(seq_ctx.dsa_topk_cache.offloaded) == {0}

    compiled_source_block(recompute_hidden_states, position_embeddings=position_embeddings, seq_ctx=seq_ctx)
    torch.cuda.synchronize()

    assert seq_ctx.dsa_topk_cache.indices == {}
    assert seq_ctx.dsa_topk_cache.offloaded == {}
    assert seq_ctx.dsa_topk_cache.released_sources == {0}


def test_dsa_attention_shared_layer_fails_when_source_topk_is_missing():
    shared_attn = _tiny_dsa_attention(indexer_types=["full", "shared"], layer_idx=1)
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    with pytest.raises(AssertionError, match="Cross-pipeline top-k sharing is not supported"):
        shared_attn(torch.randn(1, 4, 4), position_embeddings, seq_ctx)
