# Copyright (c) OpenMMLab. All rights reserved.
"""Fused MHC RMSNorm/projection with FP32 GEMMs.

The split-K formulation follows DeepSeek TileKernels' MHCPreNormFn:
https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/modeling/mhc/ops/norm_fn.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_mul(x_ptr, fn_ptr, out_ptr, sqrsum_ptr, seq_len, hidden: tl.constexpr, mix_size: tl.constexpr):
    token_block: tl.constexpr = 32
    hidden_block: tl.constexpr = 128
    mix: tl.constexpr = 32
    pid = tl.program_id(0)
    tokens = pid * token_block + tl.arange(0, token_block)
    features = tl.arange(0, mix)
    acc = tl.zeros((token_block, mix), dtype=tl.float32)
    sqrsum = tl.zeros((token_block,), dtype=tl.float32)
    for start in range(0, hidden, hidden_block):
        offsets = start + tl.arange(0, hidden_block)
        x = tl.load(x_ptr + tokens[:, None] * hidden + offsets[None, :], mask=tokens[:, None] < seq_len, other=0.0).to(
            tl.float32
        )
        fn = tl.load(
            fn_ptr + features[None, :] * hidden + offsets[:, None], mask=features[None, :] < mix_size, other=0.0
        )
        acc += tl.dot(x, fn, allow_tf32=False)
        sqrsum += tl.sum(x * x, axis=1)
    tl.store(
        out_ptr + tokens[:, None] * mix_size + features[None, :],
        acc,
        mask=(tokens[:, None] < seq_len) & (features[None, :] < mix_size),
    )
    tl.store(sqrsum_ptr + tokens, sqrsum, mask=tokens < seq_len)


@triton.jit
def _bwd_x(
    out_grad_ptr,
    sqrsum_grad_ptr,
    x_ptr,
    fn_ptr,
    x_grad_ptr,
    seq_len,
    hidden: tl.constexpr,
    mix_size: tl.constexpr,
):
    token_block: tl.constexpr = 32
    hidden_block: tl.constexpr = 128
    mix: tl.constexpr = 32
    tokens = tl.program_id(0) * token_block + tl.arange(0, token_block)
    token_mask = tokens < seq_len
    offsets = tl.program_id(1) * hidden_block + tl.arange(0, hidden_block)
    features = tl.arange(0, mix)
    out_grad = tl.load(
        out_grad_ptr + tokens[:, None] * mix_size + features[None, :],
        mask=token_mask[:, None] & (features[None, :] < mix_size),
        other=0.0,
    )
    sqrsum_grad = tl.load(sqrsum_grad_ptr + tokens, mask=token_mask, other=0.0)
    x = tl.load(x_ptr + tokens[:, None] * hidden + offsets[None, :], mask=token_mask[:, None], other=0.0).to(
        tl.float32
    )
    fn = tl.load(fn_ptr + features[:, None] * hidden + offsets[None, :], mask=features[:, None] < mix_size, other=0.0)
    x_grad = tl.dot(out_grad, fn, allow_tf32=False) + 2 * x * sqrsum_grad[:, None]
    tl.store(x_grad_ptr + tokens[:, None] * hidden + offsets[None, :], x_grad, mask=token_mask[:, None])


@triton.jit
def _bwd_fn(out_grad_ptr, x_ptr, fn_grad_ptr, seq_len, hidden: tl.constexpr, mix_size: tl.constexpr):
    token_block: tl.constexpr = 32
    hidden_block: tl.constexpr = 128
    mix: tl.constexpr = 32
    pid = tl.program_id(0)
    offsets = pid * hidden_block + tl.arange(0, hidden_block)
    features = tl.arange(0, mix)
    fn_grad = tl.zeros((mix, hidden_block), dtype=tl.float32)
    for start in range(0, seq_len, token_block):
        tokens = start + tl.arange(0, token_block)
        token_mask = tokens < seq_len
        out_grad = tl.load(
            out_grad_ptr + tokens[:, None] * mix_size + features[None, :],
            mask=token_mask[:, None] & (features[None, :] < mix_size),
            other=0.0,
        )
        x = tl.load(x_ptr + tokens[:, None] * hidden + offsets[None, :], mask=token_mask[:, None], other=0.0).to(
            tl.float32
        )
        fn_grad += tl.dot(tl.trans(out_grad), x, allow_tf32=False)
    tl.store(fn_grad_ptr + features[:, None] * hidden + offsets[None, :], fn_grad, mask=features[:, None] < mix_size)


class MHCPreNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, fn: torch.Tensor, norm_eps: float) -> torch.Tensor:
        assert x.is_cuda and fn.is_cuda
        assert x.dtype == torch.bfloat16 and fn.dtype == torch.float32
        assert fn.ndim == 2 and x.shape[-2] * x.shape[-1] == fn.shape[1]
        assert fn.shape[0] <= 32 and fn.shape[1] % 128 == 0
        assert fn.is_contiguous()
        x_flat = x.flatten(2).reshape(-1, fn.shape[1]).contiguous()
        seq_len, hidden = x_flat.shape
        mix_size = fn.shape[0]
        out_mul = torch.empty(seq_len, mix_size, device=x.device, dtype=torch.float32)
        sqrsum = torch.empty(seq_len, device=x.device, dtype=torch.float32)
        _fwd_mul[(triton.cdiv(seq_len, 32),)](x_flat, fn, out_mul, sqrsum, seq_len, hidden, mix_size)
        rms = torch.rsqrt(sqrsum / hidden + norm_eps)
        ctx.save_for_backward(x, fn, out_mul, sqrsum)
        ctx.norm_eps = norm_eps
        return (out_mul * rms.unsqueeze(-1)).reshape(*x.shape[:-2], mix_size)

    @staticmethod
    def backward(ctx, out_grad: torch.Tensor):
        x, fn, out_mul, sqrsum = ctx.saved_tensors
        x_flat = x.flatten(2).reshape(-1, fn.shape[1]).contiguous()
        seq_len, hidden = x_flat.shape
        mix_size = fn.shape[0]
        out_grad = out_grad.reshape(-1, mix_size)
        rms = torch.rsqrt(sqrsum / hidden + ctx.norm_eps)
        out_mul_grad = out_grad * rms.unsqueeze(-1)
        sqrsum_grad = -0.5 * (out_grad * out_mul).sum(dim=-1) * rms / (sqrsum + ctx.norm_eps * hidden)
        x_grad = torch.empty_like(x_flat)
        fn_grad = torch.empty_like(fn)
        _bwd_x[(triton.cdiv(seq_len, 32), hidden // 128)](
            out_mul_grad, sqrsum_grad, x_flat, fn, x_grad, seq_len, hidden, mix_size
        )
        _bwd_fn[(hidden // 128,)](out_mul_grad, x_flat, fn_grad, seq_len, hidden, mix_size)
        return x_grad.view_as(x), fn_grad, None


def mhc_pre_norm_fn(x: torch.Tensor, fn: torch.Tensor, norm_eps: float) -> torch.Tensor:
    return MHCPreNormFn.apply(x, fn.float(), norm_eps)
