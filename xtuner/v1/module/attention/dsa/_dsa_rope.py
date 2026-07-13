# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Rope (rotary position embedding) ops for DeepSeek Sparse Attention.
#
# Split out of ``dsa.py``: these helpers apply rotate-half rope to the trailing
# ``rope_head_dim`` slice of a head while leaving the NoPE prefix untouched.
# ``_apply_rope`` / ``_apply_rope_inverse`` assume the structured cos/sin layout
# produced by :class:`xtuner.v1.module.rope.DualRotaryEmbedding`. On bf16 CUDA
# tensors the fused Triton kernel below collapses the NoPE-prefix copy and the
# rope-tail rotation into one HBM pass; the slice+cat path is the fallback.
# ============================================================================

from typing import Any

import torch
from torch import Tensor


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rope_split_triton_kernel(
        x_ptr,  # [T, H, full_dim] bf16 — input (token × head × head_dim)
        cos_ptr,  # [T, rope_dim]  fp32 — pre-arranged cos (sign pattern folded in)
        sin_ptr,  # [T, rope_dim]  fp32 — pre-arranged sin
        out_ptr,  # [T, H, full_dim] bf16 — output
        H,  # num heads (runtime; 128 for q/attn_out, 1 for kv)
        full_dim,
        nope_dim,
        rope_dim,
        BLOCK: tl.constexpr,  # next_power_of_2(full_dim); fits the entire head in one tile
        FORWARD: tl.constexpr,  # True = forward rotation, False = inverse (negate sin)
    ):
        # 2-D grid: axis-0 = token t, axis-1 = head h.
        # Each program reads the full head_dim once and writes once — no cat, no
        # intermediate rotated-tail allocation.
        t = tl.program_id(0)
        h = tl.program_id(1)

        x_base = x_ptr + (t * H + h) * full_dim
        out_base = out_ptr + (t * H + h) * full_dim
        cos_base = cos_ptr + t * rope_dim
        sin_base = sin_ptr + t * rope_dim

        i = tl.arange(0, BLOCK)
        x_val = tl.load(x_base + i, mask=i < full_dim, other=0.0).to(tl.float32)

        in_rope = i >= nope_dim
        # j: position within rope region [0..rope_dim-1]; clamped to 0 for nope lanes
        # so masked cos/sin loads never go OOB.
        j = tl.where(in_rope, i - nope_dim, 0)

        cos_val = tl.load(cos_base + j, mask=in_rope, other=1.0).to(tl.float32)
        sin_val = tl.load(sin_base + j, mask=in_rope, other=0.0).to(tl.float32)

        # flip_pairs: swap adjacent pairs (x[2k], x[2k+1]) within the rope region.
        # XOR of last bit: 0↔1, 2↔3, etc.  Implements the rotate-half convention.
        j_swap = j ^ 1
        x_swap = tl.load(x_base + nope_dim + j_swap, mask=in_rope, other=0.0).to(tl.float32)

        if FORWARD:
            rot = x_val * cos_val + x_swap * sin_val
        else:
            # Inverse: negate sin component, which undoes the forward rotation.
            rot = x_val * cos_val - x_swap * sin_val

        out = tl.where(in_rope, rot, x_val)
        tl.store(out_base + i, out.to(tl.bfloat16), mask=i < full_dim)

    @torch.library.custom_op("xtuner::rope_split", mutates_args=())
    def _rope_split_op(x: Tensor, cos: Tensor, sin: Tensor, rope_head_dim: int, forward: bool) -> Tensor:
        """Fused NoPE-copy + rope-tail rotation in one HBM pass.

        Args:
            x (Tensor): Shape ``[T, H, full_dim]`` bf16 — normalized (token, head, head_dim).
            cos (Tensor): Shape ``[T, rope_dim]`` — pre-arranged cos.
            sin (Tensor): Shape ``[T, rope_dim]`` — pre-arranged sin.
            rope_head_dim (int): Length of the rope suffix.
            forward (bool): ``True`` for forward rotation, ``False`` for inverse.

        Returns:
            Tensor: Same shape and dtype as ``x``.
        """
        T, H, full_dim = x.shape
        nope_dim = full_dim - rope_head_dim
        BLOCK = triton.next_power_of_2(full_dim)
        out = torch.empty_like(x)
        _rope_split_triton_kernel[(T, H)](
            x.contiguous(),
            cos.contiguous(),
            sin.contiguous(),
            out,
            H=H,
            full_dim=full_dim,
            nope_dim=nope_dim,
            rope_dim=rope_head_dim,
            BLOCK=BLOCK,
            FORWARD=forward,
            num_warps=4,
        )
        return out

    @_rope_split_op.register_fake
    def _(x: Tensor, cos: Tensor, sin: Tensor, rope_head_dim: int, forward: bool) -> Tensor:
        return torch.empty_like(x)

    def _rope_split_setup_ctx(ctx: Any, inputs: tuple, output: Tensor) -> None:
        x, cos, sin, rope_head_dim, forward = inputs
        ctx.save_for_backward(cos, sin)
        ctx.rope_head_dim = rope_head_dim
        ctx.forward = forward

    def _rope_split_backward(ctx: Any, grad_out: Tensor) -> tuple:
        cos, sin = ctx.saved_tensors
        # Backward of a rotation R is its transpose R^T.  For DualRotaryEmbedding's
        # structured cos/sin (sin[2k]=-s, sin[2k+1]=+s, c²+s²=1) the rotation is
        # orthogonal so R^T = R^{-1}, which the `not ctx.forward` path computes.
        grad_x = _rope_split_op(grad_out.contiguous(), cos, sin, ctx.rope_head_dim, not ctx.forward)
        return grad_x, None, None, None, None

    _rope_split_op.register_autograd(_rope_split_backward, setup_context=_rope_split_setup_ctx)


def _apply_rope_split(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    # Apply rotate-half rope to the final `rope_head_dim` slice of `x`,
    # leaving the NoPE prefix untouched.
    #
    # On bf16 CUDA tensors the Triton path fuses the copy of the NoPE prefix
    # and the rope rotation into one HBM pass, avoiding the intermediate
    # rotated-tail allocation and the separate cat copy of the nope prefix.
    if _HAS_TRITON and x.is_cuda and x.dtype == torch.bfloat16:
        orig_shape = x.shape
        # Normalise to [T, H, full_dim]; kv has no head dim — add a dummy axis.
        x_3d = x.reshape(-1, 1, orig_shape[-1]) if x.dim() == 3 else x.reshape(-1, orig_shape[-2], orig_shape[-1])
        cos_2d = cos.reshape(-1, rope_head_dim)
        sin_2d = sin.reshape(-1, rope_head_dim)
        return _rope_split_op(x_3d, cos_2d, sin_2d, rope_head_dim, True).reshape(orig_shape)
    nope = x[..., : x.size(-1) - rope_head_dim]
    rope_tail = x[..., x.size(-1) - rope_head_dim :]
    rope_tail = _apply_rope(rope_tail, cos, sin)
    return torch.cat([nope, rope_tail], dim=-1)


def _apply_rope_inverse_split(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    if _HAS_TRITON and x.is_cuda and x.dtype == torch.bfloat16:
        orig_shape = x.shape
        x_3d = x.reshape(-1, 1, orig_shape[-1]) if x.dim() == 3 else x.reshape(-1, orig_shape[-2], orig_shape[-1])
        cos_2d = cos.reshape(-1, rope_head_dim)
        sin_2d = sin.reshape(-1, rope_head_dim)
        return _rope_split_op(x_3d, cos_2d, sin_2d, rope_head_dim, False).reshape(orig_shape)
    nope = x[..., : x.size(-1) - rope_head_dim]
    rope_tail = x[..., x.size(-1) - rope_head_dim :]
    rope_tail = _apply_rope_inverse(rope_tail, cos, sin)
    return torch.cat([nope, rope_tail], dim=-1)


def _apply_rope(x: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor) -> torch.Tensor:
    # Interleaved RoPE rotation, pair-wise: ``(x[2i], x[2i+1])`` →
    # ``(x[2i]·cos_i − x[2i+1]·sin_i, x[2i]·sin_i + x[2i+1]·cos_i)``.
    # Mathematically identical to HF ``apply_rotary_pos_emb_interleave`` and
    # the V4-Flash reference's complex-pair ``apply_rotary_emb``.
    #
    # ``cos_full`` / ``sin_full`` are D-dim and pre-arranged by
    # :class:`xtuner.v1.module.rope.DualRotaryEmbedding`:
    #   ``cos_full[..., 2i] == cos_full[..., 2i+1] == cos_half[..., i]``
    #   ``sin_full[..., 2i]   = -sin_half[..., i]``
    #   ``sin_full[..., 2i+1] = +sin_half[..., i]``
    # so the rotation reduces to ``x * cos_full + flip_pairs(x) * sin_full``
    # with no ``unbind`` on ``x`` and no per-call ``repeat_interleave`` /
    # ``stack``. ``flip_pairs`` swaps the two elements of each adjacent pair:
    # ``(x[2i], x[2i+1]) → (x[2i+1], x[2i])``. Read off position 2i:
    #   ``x[2i]·cos_half[i] + x[2i+1]·(-sin_half[i])  =  x[2i]·cos − x[2i+1]·sin``  ✓
    # And 2i+1:
    #   ``x[2i+1]·cos_half[i] + x[2i]·(+sin_half[i])  =  x[2i+1]·cos + x[2i]·sin``  ✓
    cos_b, sin_b = _broadcast_cos_sin(cos_full, sin_full, x)
    x_swap = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    return x * cos_b + x_swap * sin_b


def _apply_rope_inverse(x: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor) -> torch.Tensor:
    # Inverse of :func:`_apply_rope` — same layout assumptions on ``cos_full``
    # / ``sin_full``; the rotation angle flips sign, which here is just
    # ``+ x_swap * sin_full`` → ``- x_swap * sin_full``.
    cos_b, sin_b = _broadcast_cos_sin(cos_full, sin_full, x)
    x_swap = x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
    return x * cos_b - x_swap * sin_b


def _broadcast_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin are [B, S, D]; x is either [B, S, D] (KV, MQA) or
    # [B, S, H, D] (queries / output). Insert a head axis when needed so the
    # broadcast multiplication doesn't accidentally flatten the heads.
    if x.dim() == cos.dim():
        return cos, sin
    if x.dim() == cos.dim() + 1:
        return cos.unsqueeze(-2), sin.unsqueeze(-2)
    raise ValueError(f"Cannot broadcast cos/sin {tuple(cos.shape)} against x {tuple(x.shape)}")
