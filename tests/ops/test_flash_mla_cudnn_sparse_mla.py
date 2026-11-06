# Copyright (c) OpenMMLab. All rights reserved.
"""flash_mla_cudnn SparseMLA 后端（FlashMLA 前向 + cuDNN 反向），见设计文档 F5.b。

TestFlashMlaCudnnSparseMLA
    test_forward_backward_matches_torch_reference         前反向与 torch 参考一致
    test_rejects_576_head_dim_without_matching_value_dim  维度白名单不接受错配的 value_dim
    test_accepts_512_head_dim_whitelist_entry             NoPE 的 (512, 512) 在白名单内
TestDeterministicSparseMLABackward
    test_deterministic_mode_backward_bitwise_repeatable   确定性模式下反向逐位可重复
    test_deterministic_mode_multilaunch_bitwise_repeatable 多 launch（含整 chunk 与尾块边界）逐位可重复
    test_deterministic_multilaunch_duplicates_and_holes_bitwise 空洞+重复索引、512/576 维逐位可重复
    test_deterministic_backward_matches_torch_reference   确定性参考反向与 torch 参考一致
    test_deterministic_rejects_multi_head_tile            多 head-tile 形状被拒绝而非静默失确定性
    test_deterministic_rejects_launch_with_more_ctas_than_slices 单 launch CTA 数超分片数被拒绝
    test_deterministic_workspace_oom_gives_clear_error    分片工作区放不下时清晰报错而非静默 OOM
"""

import importlib.util

import pytest
import torch

from xtuner.v1.ops.sparse_mla import sparse_mla


BF16_ATOL = 1e-2
BF16_RTOL = 1.6e-2
# dKV accumulates gradient contributions from every query that selected a given compressed-KV
# position; at topk=512 >= seq_len=64 (near-full attention density) every position is selected
# by every later query, so this is a dense ~64-term reduction whose accumulation order differs
# between the fp32 torch reference and cuDNN's internal bf16 accumulation. GLM-5.2's own
# tilelang-vs-cudnn backward test (tests/module/attention/test_dsa_mla.py) already tolerates
# 1e-1/1e-1 for the same kernel pairing at a sparser density; this NoPE shape (head_dim=512 vs
# GLM-5.2's 576) needs a bit more headroom for the same reason, not a different bug class --
# forward, softmax_lse, and dQ all match at the tight BF16_ATOL/RTOL above.
DKV_ATOL = 3e-1
DKV_RTOL = 3e-1


def _flash_mla_cudnn_available() -> bool:
    try:
        from xtuner.v1.ops.sparse_mla import ensure_flash_mla_cudnn_runtime_available

        ensure_flash_mla_cudnn_runtime_available()
        return True
    except Exception:
        return False


def _sparse_indices(seq_len: int, topk: int) -> torch.Tensor:
    indices = torch.full((seq_len, 1, topk), -1, device="cuda", dtype=torch.int64)
    for token_idx in range(seq_len):
        valid = min(token_idx + 1, topk)
        indices[token_idx, 0, :valid] = torch.arange(token_idx + 1 - valid, token_idx + 1, device="cuda")
    return indices


def _nope_sparse_mla_inputs(seq_len: int = 64):
    torch.manual_seed(0)
    # NoPE absorbed latent: head_dim == value_dim == 512 (qk_rope_head_dim=0, no rope tail).
    # num_heads=64 matches FlashMLA's _FLASH_MLA_HEAD_ALIGNMENT.
    q = torch.randn(seq_len, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len, 1, 512, device="cuda", dtype=torch.bfloat16)
    return q, kv, _sparse_indices(seq_len, topk=512)


class TestFlashMlaCudnnSparseMLA:
    # conftest 默认 XTUNER_DETERMINISTIC=true；反向 dispatch 在调用时读模块级常量，
    # 必须显式参数化两条路径，否则 stock 反向在 pytest 下失去回归覆盖。
    @pytest.mark.parametrize("deterministic", [False, True])
    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_forward_backward_matches_torch_reference(self, monkeypatch, deterministic):
        # FlashMLA 前向 + cuDNN 反向的组合必须与纯 torch 参考在 bf16 容差内一致。
        from xtuner.v1.ops.sparse_mla import cudnn_dsa

        monkeypatch.setattr(cudnn_dsa, "XTUNER_DETERMINISTIC", deterministic)
        q, kv, indices = _nope_sparse_mla_inputs()
        scaling = 1 / (q.shape[-1] ** 0.5)
        q_ref = q.detach().clone().requires_grad_()
        kv_ref = kv.detach().clone().requires_grad_()
        q_actual = q.detach().clone().requires_grad_()
        kv_actual = kv.detach().clone().requires_grad_()

        expected = sparse_mla(q_ref, kv_ref, indices, scaling=scaling, value_dim=512, backend="torch")
        actual = sparse_mla(
            q_actual, kv_actual, indices.to(torch.int32), scaling=scaling, value_dim=512, backend="flash_mla_cudnn"
        )

        grad_output = torch.randn_like(expected.raw_output)
        expected.raw_output.backward(grad_output)
        actual.raw_output.backward(grad_output)

        torch.testing.assert_close(actual.raw_output, expected.raw_output, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(actual.softmax_lse, expected.softmax_lse, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(q_actual.grad, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(kv_actual.grad, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)

    def test_rejects_576_head_dim_without_matching_value_dim(self):
        """(576, 512) is GLM-5.2's whitelist entry; a mismatched value_dim must
        still fail -- widening the whitelist must not silently accept arbitrary
        shapes."""
        # 放宽维度白名单不能顺带接受任意形状，错配的 value_dim 仍要报错。
        from xtuner.v1.ops.sparse_mla.tilelang import validate_sparse_mla_inputs

        q = torch.randn(4, 8, 576, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
        kv = torch.randn(4, 1, 576, device=q.device, dtype=torch.bfloat16)
        indices = torch.zeros(4, 1, 64, device=q.device, dtype=torch.int32)
        if not q.is_cuda:
            pytest.skip("validation short-circuits on the CUDA check before reaching the shape check")
        with pytest.raises(RuntimeError, match="head_dim, value_dim"):
            validate_sparse_mla_inputs(q, kv, indices, value_dim=256)

    def test_accepts_512_head_dim_whitelist_entry(self):
        # NoPE 的 (512, 512) 必须在白名单内，否则 GLM-5.3 走不通。
        from xtuner.v1.ops.sparse_mla.tilelang import validate_sparse_mla_inputs

        if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
            pytest.skip("requires CUDA")
        q = torch.randn(4, 8, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(4, 1, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.zeros(4, 1, 64, device="cuda", dtype=torch.int32)
        validate_sparse_mla_inputs(q, kv, indices, value_dim=512)  # must not raise


class TestDeterministicSparseMLABackward:
    """XTUNER_DETERMINISTIC 下 flash_mla_cudnn 的 fork 分片归约确定性反向。

    cuDNN 稀疏注意力反向的 dkv 是无序原子累加，同输入两次反向不逐位一致，CI 双跑 grad_norm 因此抖动；确定性模式改走 fork 分片归约（cudnn_dsa_det.py： 每 CTA 写自己的 FP32
    分片、按固定顺序归约，零跨 CTA 原子竞争）。
    """

    @pytest.fixture
    def deterministic_mode(self, monkeypatch):
        from xtuner.v1.ops.sparse_mla import cudnn_dsa

        # 反向分支在调用时读模块级常量，monkeypatch 即可覆盖 import 期绑定。
        monkeypatch.setattr(cudnn_dsa, "XTUNER_DETERMINISTIC", True)
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        yield
        torch.use_deterministic_algorithms(previous)

    def _run_backward(self, q0, kv0, indices, grad_output):
        q = q0.detach().clone().requires_grad_()
        kv = kv0.detach().clone().requires_grad_()
        out = sparse_mla(
            q, kv, indices.to(torch.int32), scaling=q0.shape[-1] ** -0.5, value_dim=512, backend="flash_mla_cudnn"
        )
        out.raw_output.backward(grad_output)
        return q.grad, kv.grad

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_mode_backward_bitwise_repeatable(self, deterministic_mode):
        # 同输入两次反向必须逐位一致——cuDNN 反向在此会因 dkv 原子序不同而失败。
        q0, kv0, indices = _nope_sparse_mla_inputs()
        grad_output = torch.randn(64, 64, 512, device="cuda", dtype=torch.bfloat16)

        dq1, dkv1 = self._run_backward(q0, kv0, indices, grad_output)
        dq2, dkv2 = self._run_backward(q0, kv0, indices, grad_output)

        assert torch.equal(dq1, dq2)
        assert torch.equal(dkv1, dkv2)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    @pytest.mark.parametrize("seq_len", [128, 129, 257])
    def test_deterministic_mode_multilaunch_bitwise_repeatable(self, deterministic_mode, seq_len):
        # 128=恰好一个完整 chunk，129=整块+1 条尾块，257=两块+1 条尾块；
        # 覆盖分片跨 launch 复用这一核心机制；topk=512 > seq_len 同时制造 -1 空洞。
        q0, kv0, indices = _nope_sparse_mla_inputs(seq_len=seq_len)
        grad_output = torch.randn(seq_len, 64, 512, device="cuda", dtype=torch.bfloat16)

        dq1, dkv1 = self._run_backward(q0, kv0, indices, grad_output)
        dq2, dkv2 = self._run_backward(q0, kv0, indices, grad_output)

        assert torch.equal(dq1, dq2)
        assert torch.equal(dkv1, dkv2)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    @pytest.mark.parametrize("head_dim", [512, 576])
    def test_deterministic_multilaunch_duplicates_and_holes_bitwise(self, head_dim):
        # 直接驱动 multi-launch driver：topk 索引含 -1 空洞与重复合法索引；
        # 576 维（GLM-5.2 形状）额外覆盖 hdim_chunk=64 的 postprocess 路径。
        # 两次运行 dq/dkv 必须逐位一致。
        from xtuner.v1.ops.sparse_mla.cudnn_dsa_det import flash_attn_bwd_sm90_multilaunch

        torch.manual_seed(0)
        seq_len, num_head, topk = 129, 64, 64
        head_dim_v = 512 if head_dim == 576 else head_dim
        q = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(seq_len, num_head, head_dim_v, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(seq_len, num_head, head_dim_v, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(seq_len, num_head, device="cuda", dtype=torch.float32)
        topk_idxs = torch.randint(0, seq_len, (seq_len, topk), device="cuda", dtype=torch.int32)
        topk_idxs[:, ::3] = -1  # 每三个位置留一个空洞，其余位置允许重复

        dq1, dkv1 = flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)
        dq2, dkv2 = flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)

        assert torch.equal(dq1, dq2)
        assert torch.equal(dkv1, dkv2)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_rejects_multi_head_tile(self):
        # 128 个 query head/kv head > tile_m=64 时 scheduler 会用 blockIdx.y 区分
        # head-tile，仅靠 blockIdx.x 的分片偏移会让两个 CTA 共享分片、静默失去
        # 逐位可重复性——必须拒绝而不是回退到 stock。
        from xtuner.v1.ops.sparse_mla.cudnn_dsa_det import flash_attn_bwd_sm90_multilaunch

        seq_len, num_head, head_dim, topk = 64, 128, 512, 64
        q = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(seq_len, num_head, device="cuda", dtype=torch.float32)
        topk_idxs = torch.arange(seq_len, device="cuda", dtype=torch.int32)[:, None].expand(seq_len, topk).contiguous()

        with pytest.raises(RuntimeError, match="qhead_per_kvhead"):
            flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_rejects_launch_with_more_ctas_than_slices(self):
        # 单次 launch 的 CTA 数（=seqlen_q，每 query 一个 CTA）超过分片数时，
        # blockIdx.x % n_splits 会让多个 CTA 共享分片——必须拒绝，引导走
        # multi-launch driver。
        from xtuner.v1.ops.sparse_mla.cudnn_dsa_det import flash_attn_bwd_sm90

        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        seq_len = sm_count + 64
        num_head, head_dim, topk = 64, 512, 64
        q = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(seq_len, num_head, device="cuda", dtype=torch.float32)
        topk_idxs = torch.arange(seq_len, device="cuda", dtype=torch.int32)[:, None].expand(seq_len, topk).contiguous()

        with pytest.raises(RuntimeError, match="one accumulator slice per CTA"):
            flash_attn_bwd_sm90(q, kv, out, dout, lse, topk_idxs=topk_idxs)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_workspace_oom_gives_clear_error(self, monkeypatch):
        # 分片工作区为 n_splits × full KV × head_dim 的 FP32，随 S_k 线性增长
        # （16K/512 维约 4 GiB，512K/576 维约 144 GiB）；放不下的形状必须清晰
        # 报错并给出 chunk 建议，而不是静默 CUDA OOM。
        from xtuner.v1.ops.sparse_mla import cudnn_dsa_det

        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024, 1024))

        seq_len, num_head, head_dim, topk = 64, 64, 512, 64
        q = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(seq_len, num_head, device="cuda", dtype=torch.float32)
        topk_idxs = torch.arange(seq_len, device="cuda", dtype=torch.int32)[:, None].expand(seq_len, topk).contiguous()

        with pytest.raises(RuntimeError, match="per-CTA"):
            cudnn_dsa_det.flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_multilaunch_shrinks_chunk_to_fit_workspace(self, monkeypatch):
        # 空闲显存只够 2 个分片时，driver 必须把每次 launch 的 query 数自动收缩到
        # 2（分片数同步减半、launch 次数等比增多）而不是直接报错；收缩后两次运行
        # 仍须逐位一致，且 dq 与正常分片数逐位相同、dkv 数值一致（仅归约分组不同，
        # 容差对齐参考测试）。
        from xtuner.v1.ops.sparse_mla import cudnn_dsa_det

        seq_len, num_head, head_dim, topk = 256, 64, 512, 64
        q = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
        out = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(seq_len, num_head, head_dim, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(seq_len, num_head, device="cuda", dtype=torch.float32)
        topk_idxs = torch.randint(0, seq_len, (seq_len, topk), device="cuda", dtype=torch.int32)

        dq_ref, dkv_ref = cudnn_dsa_det.flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)

        # 每片 4*256(round64(seq_len))*512 = 0.5 MiB；2 MiB 空闲的 80% 预算只够 2 片，
        # 默认 n_splits=128 会被逐级减半收缩到 2（2 个 launch 变成 128 个）。
        free_bytes = 2 * 1024**2
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_bytes, free_bytes))

        dq1, dkv1 = cudnn_dsa_det.flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)
        dq2, dkv2 = cudnn_dsa_det.flash_attn_bwd_sm90_multilaunch(q, kv, out, dout, lse, topk_idxs)

        assert torch.equal(dq1, dq2)
        assert torch.equal(dkv1, dkv2)
        assert torch.equal(dq1, dq_ref)  # dq 路径与分片几何无关
        assert torch.allclose(dkv1, dkv_ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(not _flash_mla_cudnn_available(), reason="requires FlashMLA + cuDNN DSA runtimes")
    def test_deterministic_backward_matches_torch_reference(self, deterministic_mode):
        # 参考反向只是换累加方式，数值语义必须保持与 torch 参考一致（容差同主测试）。
        q0, kv0, indices = _nope_sparse_mla_inputs()
        scaling = 1 / (q0.shape[-1] ** 0.5)
        q_ref = q0.detach().clone().requires_grad_()
        kv_ref = kv0.detach().clone().requires_grad_()
        expected = sparse_mla(q_ref, kv_ref, indices, scaling=scaling, value_dim=512, backend="torch")
        grad_output = torch.randn_like(expected.raw_output)
        expected.raw_output.backward(grad_output)

        dq, dkv = self._run_backward(q0, kv0, indices, grad_output)

        torch.testing.assert_close(dq, q_ref.grad, atol=BF16_ATOL, rtol=BF16_RTOL)
        torch.testing.assert_close(dkv, kv_ref.grad, atol=DKV_ATOL, rtol=DKV_RTOL)
