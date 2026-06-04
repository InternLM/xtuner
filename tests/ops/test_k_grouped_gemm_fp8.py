import pytest
import torch


REPLAY_K_INDICES = (
    0,
    0,
    384,
    128,
    128,
    0,
    0,
    128,
    1536,
    0,
    0,
    0,
    0,
    128,
    0,
    0,
    0,
    0,
    0,
    0,
    896,
    0,
    0,
    128,
    0,
    896,
    0,
    0,
    512,
    2560,
    0,
    0,
    0,
    0,
    128,
    0,
    128,
    512,
    0,
    128,
    128,
    128,
    0,
    128,
    8064,
    0,
    7552,
    0,
    128,
    0,
    0,
    2560,
    0,
    128,
    0,
    0,
    256,
    0,
    128,
    0,
    0,
    0,
    0,
    0,
    128,
    1536,
    0,
    128,
    128,
    0,
    256,
    256,
    128,
    256,
    7296,
    128,
    128,
    256,
    0,
    0,
    0,
    256,
    0,
    0,
    5760,
    0,
    0,
    1152,
    0,
    0,
    0,
    128,
    2688,
    128,
    0,
    0,
    0,
    0,
    128,
    0,
    128,
    384,
    128,
    0,
    0,
    0,
    0,
    128,
    0,
    7552,
    128,
    0,
    0,
    5120,
    0,
    0,
    128,
    0,
    6912,
    0,
    128,
    0,
    128,
    0,
    128,
    768,
    256,
    0,
)

LHS_SCALE_MIN = -1e-5
LHS_SCALE_MAX = 1e-5
RHS_SCALE_MIN = -0.5
RHS_SCALE_MAX = 0.5
ASSERT_ATOL = 4e-3
ASSERT_RTOL = 5e-3
DETERMINISM_REPEATS = 3
SCALED_GROUPED_MM_REF_IMPLEMENTATION = "fp32"
TEST_SEEDS = (1024, 2048, 4096)


def _calibrate_to_range(scales: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    midpoint = (min_value + max_value) / 2
    half_range = (max_value - min_value) / 2
    centered = scales.float() - scales.float().mean()
    denom = centered.abs().amax().clamp_min(1e-12)
    return centered / denom * half_range + midpoint


def _quantize_lhs(lhs: torch.Tensor, block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = lhs.shape
    assert k % block_k == 0

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    lhs_view = lhs.view(m, k // block_k, block_k)
    lhs_amax = lhs_view.abs().float().amax(dim=2, keepdim=True).clamp_min(1e-12)
    lhs_scales = (lhs_amax / fp8_max).squeeze(2)
    lhs_fp8 = (lhs_view.float() * (fp8_max / lhs_amax)).to(torch.float8_e4m3fn).view(m, k)
    return lhs_fp8.contiguous(), lhs_scales.contiguous()


def _quantize_rhs(rhs: torch.Tensor, block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = rhs.shape
    assert k % block_k == 0

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    n_blocks = (n + block_k - 1) // block_k
    k_blocks = k // block_k
    rhs_padded = torch.zeros((n_blocks * block_k, k), device=rhs.device, dtype=rhs.dtype)
    rhs_padded[:n, :] = rhs
    rhs_view = rhs_padded.view(n_blocks, block_k, k_blocks, block_k).permute(0, 2, 1, 3)
    rhs_amax = rhs_view.abs().float().amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    rhs_scales = (rhs_amax / fp8_max).view(n_blocks, k_blocks)

    rhs_fp8 = (rhs_view.float() * (fp8_max / rhs_amax)).to(torch.float8_e4m3fn)
    rhs_fp8 = rhs_fp8.permute(0, 2, 1, 3).reshape(n_blocks * block_k, k)[:n, :]
    return rhs_fp8, rhs_scales


def _make_inputs(
    m: int,
    n: int,
    k_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_k = 128
    total_k = int(k_indices.sum().item())
    assert total_k % block_k == 0
    assert bool((k_indices % block_k == 0).all().item())

    lhs_bf16 = torch.randn((m, total_k), device="cuda", dtype=torch.bfloat16)
    rhs_bf16 = torch.randn((n, total_k), device="cuda", dtype=torch.bfloat16)
    lhs, lhs_scales = _quantize_lhs(lhs_bf16, block_k)
    rhs, rhs_scales = _quantize_rhs(rhs_bf16, block_k)

    # Keep AdaptiveGEMM's tile layout, but exercise the signed scale ranges
    # used by the fused kernel caller.
    lhs_scales = _calibrate_to_range(lhs_scales, LHS_SCALE_MIN, LHS_SCALE_MAX)
    rhs_scales = _calibrate_to_range(rhs_scales, RHS_SCALE_MIN, RHS_SCALE_MAX)
    return lhs, lhs_scales.contiguous(), rhs.contiguous(), rhs_scales.contiguous()


def _scaled_grouped_mm_ref(
    lhs: torch.Tensor,
    rhs_t: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_ends: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    implementation: str,
) -> torch.Tensor:
    if implementation == "torch":
        return torch._scaled_grouped_mm(
            lhs,
            rhs_t,
            lhs_scale.contiguous().view(-1),
            rhs_scale.contiguous().view(-1),
            offs=group_ends,
            out_dtype=out_dtype,
            use_fast_accum=False,
        )

    if implementation != "fp32":
        raise ValueError(f"Unknown scaled grouped mm ref implementation: {implementation}")

    m, total_k = lhs.shape
    rhs_k, n = rhs_t.shape
    assert total_k == rhs_k
    assert lhs_scale.shape == (group_ends.numel(), m)
    assert rhs_scale.shape == (group_ends.numel(), n)

    out = torch.empty((group_ends.numel(), m, n), device=lhs.device, dtype=out_dtype)
    k_start = 0
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for group_idx, k_end in enumerate(group_ends.cpu().tolist()):
            if k_end == k_start:
                out[group_idx].zero_()
            else:
                accum = lhs[:, k_start:k_end].float() @ rhs_t[k_start:k_end, :].float()
                accum *= lhs_scale[group_idx].view(m, 1)
                accum *= rhs_scale[group_idx].view(1, n)
                out[group_idx] = accum.to(out_dtype)
            k_start = k_end
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return out


def _k_grouped_gemm_scaled_grouped_mm_ref(
    lhs: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs: torch.Tensor,
    rhs_scales: torch.Tensor,
    k_indices: torch.Tensor,
    *,
    scaled_grouped_mm_implementation: str = SCALED_GROUPED_MM_REF_IMPLEMENTATION,
) -> torch.Tensor:
    block_k = 128
    m, total_k = lhs.shape
    n, rhs_k = rhs.shape
    assert total_k == rhs_k and total_k % block_k == 0
    assert bool((k_indices % block_k == 0).all().item())

    block_ends = torch.arange(
        block_k,
        total_k + block_k,
        block_k,
        device=lhs.device,
        dtype=torch.int32,
    )
    lhs_scale = lhs_scales.t().contiguous()
    # RHS scales are 128-row N tiles in AdaptiveGEMM. Expand each tile scale to
    # the per-output-column scale format expected by torch._scaled_grouped_mm.
    rhs_scale = rhs_scales.t().repeat_interleave(block_k, dim=1)[:, :n].contiguous()

    ref = torch.empty((k_indices.numel(), m, n), device=lhs.device, dtype=torch.bfloat16)
    if scaled_grouped_mm_implementation == "torch":
        block_ref = _scaled_grouped_mm_ref(
            lhs,
            rhs.t(),
            lhs_scale,
            rhs_scale,
            out_dtype=torch.bfloat16,
            group_ends=block_ends,
            implementation=scaled_grouped_mm_implementation,
        )
        assert block_ref.shape == (total_k // block_k, m, n)

        block_start = 0
        for group_idx, k_size in enumerate(k_indices.cpu().tolist()):
            block_count = k_size // block_k
            if block_count == 0:
                ref[group_idx].zero_()
            else:
                ref[group_idx] = block_ref[block_start : block_start + block_count].sum(dim=0)
                block_start += block_count
        return ref

    if scaled_grouped_mm_implementation != "fp32":
        raise ValueError(
            f"Unknown scaled grouped mm ref implementation: {scaled_grouped_mm_implementation}"
        )

    local_group_ends = torch.tensor([block_k], device=lhs.device, dtype=torch.int32)
    block_start = 0
    for group_idx, k_size in enumerate(k_indices.cpu().tolist()):
        block_count = k_size // block_k
        if block_count == 0:
            ref[group_idx].zero_()
        else:
            accum = torch.zeros((m, n), device=lhs.device, dtype=torch.float32)
            for block_idx in range(block_start, block_start + block_count):
                k_start = block_idx * block_k
                block_out = _scaled_grouped_mm_ref(
                    lhs[:, k_start : k_start + block_k],
                    rhs[:, k_start : k_start + block_k].t(),
                    lhs_scale[block_idx : block_idx + 1],
                    rhs_scale[block_idx : block_idx + 1],
                    out_dtype=torch.float32,
                    group_ends=local_group_ends,
                    implementation=scaled_grouped_mm_implementation,
                )
                accum += block_out[0]
            ref[group_idx] = accum.to(torch.bfloat16)
            block_start += block_count
    return ref


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_k_grouped_gemm_fp8_matches_torch_scaled_grouped_mm_ref(seed: int) -> None:
    adaptive_gemm = pytest.importorskip("adaptive_gemm")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    m, n = 2048, 768
    k_indices = torch.tensor(REPLAY_K_INDICES, device="cuda", dtype=torch.int32)
    lhs, lhs_scales, rhs, rhs_scales = _make_inputs(m, n, k_indices)
    total_k_blocks = int(k_indices.sum().item()) // 128
    assert lhs_scales.shape == (m, total_k_blocks)
    assert rhs_scales.shape == ((n + 127) // 128, total_k_blocks)
    lhs_scale_min, lhs_scale_max = lhs_scales.min().item(), lhs_scales.max().item()
    rhs_scale_min, rhs_scale_max = rhs_scales.min().item(), rhs_scales.max().item()
    assert LHS_SCALE_MIN <= lhs_scale_min < 0 < lhs_scale_max <= LHS_SCALE_MAX
    assert RHS_SCALE_MIN <= rhs_scale_min < 0 < rhs_scale_max <= RHS_SCALE_MAX

    ref = _k_grouped_gemm_scaled_grouped_mm_ref(lhs, lhs_scales, rhs, rhs_scales, k_indices)

    def run_kernel() -> torch.Tensor:
        out = torch.empty((k_indices.numel(), m, n), device="cuda", dtype=torch.bfloat16)
        adaptive_gemm.k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(
            lhs,
            lhs_scales,
            rhs,
            rhs_scales,
            out,
            k_indices,
        )
        torch.cuda.synchronize()
        return out

    out = run_kernel()
    torch.testing.assert_close(out, ref, atol=ASSERT_ATOL, rtol=ASSERT_RTOL)
    assert bool((out[k_indices == 0] == 0).all().item())

    for _ in range(DETERMINISM_REPEATS):
        repeated_out = run_kernel()
        torch.testing.assert_close(repeated_out, out, atol=0, rtol=0)
