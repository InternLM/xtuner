# Copyright (c) OpenMMLab. All rights reserved.
#
# The structural reference for the Hyper-Connections wrapper (``hc_pre`` / ``hc_post``
# semantics, parameter shapes, init policy and per-block forward order) is
# DeepSeek-V4-Flash's ``inference/model.py::Block`` (MIT-licensed):
#   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# We re-implement the same numerical contract in pure PyTorch on top of
# :func:`xtuner.v1.module.decoder_layer.hc_sinkhorn.hc_split_sinkhorn` so XTuner
# can train without depending on TileLang.
"""Hyper-Connections (HC) primitives for DeepSeek-V4-Flash.

The HC machinery keeps ``hc_mult`` copies of the hidden state and replaces the
plain ``x = x + block(norm(x))`` residual with a learned mix:

1. :func:`hc_pre` reduces the ``hc_mult`` streams to one weighted stream that
   an attention or FFN sub-block consumes.
2. :func:`hc_post` re-expands the sub-block output into ``hc_mult`` streams
   using a learned doubly-stochastic combination of the original streams plus
   the sub-block output.

These two functions plus :class:`HCWrapperConfig` and
:func:`_unshard_hc_params` are the public surface — they are consumed by
:class:`xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer`, which inlines the
HC residual-mix pattern around its own attention and FFN sub-blocks.

History note. An earlier version of this file shipped a ``HCDecoderLayer``
class plus an ``HCInnerBlock`` structural protocol, intended as a generic
"any attn+ffn block" wrapper. V4 was the only user and the protocol was too
narrow for DSA's kwargs (it forced a ``set_context`` side-channel on the
inner block). The class was removed during the V4 layer consolidation; the
math is now consumed directly as functions, which is also more compile-
friendly (no nested ``nn.Module`` for dynamo to trace into).
"""

import os

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from xtuner.v1.utils.compile import maybe_compile

from .hc_sinkhorn import hc_split_sinkhorn


# Opt-in switch for the HF-bit-parity ``hc_pre`` path. Setting
# ``XTUNER_V4_HF_PARITY=1`` in the environment reverts ``hc_pre`` to the
# all-fp32 RMS + fp32 Linear chain that matches HF's
# ``DeepseekV4HyperConnection.forward`` bitwise (used by the ``atol=0``
# parity tests). Default off → bf16 Linear with cuBLAS fp32 accumulator
# (~10-20× faster on H100 at K=16384/N=24, ~1.5e-2 max abs drift on the
# layer output, which is below the already-accepted MoE cutlass GEMM
# diff of 2.7e-2). The env var is read once at module import time —
# changing it mid-process has no effect; restart the worker.
_HC_HF_PARITY = os.getenv("XTUNER_V4_HF_PARITY", "0") == "1"


class HCWrapperConfig(BaseModel):
    """Configuration for the HC residual-mix pattern.

    Mirrors the three HC-related fields of the DeepSeek-V4-Flash config:
    ``hc_mult``, ``hc_eps``, ``hc_sinkhorn_iters``.

    Args:
        hc_mult (int): Number of hyper-connection streams. ``1`` makes the
            HC math degenerate to a plain pre-norm residual block (used as a
            structural parity anchor in tests).
        hc_eps (float): Stabilizer used inside the Sinkhorn normalization.
        hc_sinkhorn_iters (int): Number of Sinkhorn iterations.
    """

    model_config = ConfigDict(extra="forbid")

    hc_mult: int
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20


@maybe_compile
def hc_pre(
    x: Tensor,
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
    norm_eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reduce ``hc_mult`` streams down to one, returning the reduced state and
    the ``post``/``comb`` weights that the matching :func:`hc_post` call will
    consume.

    Faithful port of ``Block.hc_pre`` in DeepSeek-V4-Flash ``inference/model.py``
    (L673-682): apply an RMS-style rescale to the flattened streams, project to
    ``mixes`` via ``hc_fn``, run :func:`hc_split_sinkhorn`, then take a weighted
    sum over the stream axis with ``pre`` as the weights.

    The HC parameters are expected to arrive as plain :class:`torch.Tensor`
    (not :class:`DTensor`); the enclosing :class:`HCDecoderLayer.forward`
    eagerly materialises the locals via ``_unshard_hc_params`` before calling
    in, so this function stays a clean compile region without intermediate
    DTensor unshard graph-breaks.

    Args:
        x (Tensor): Hidden states, shape ``[B, S, hc_mult, hidden_size]``.
        hc_fn (Tensor): Mixing projection, shape ``[(2 + hc_mult) * hc_mult, hc_mult * hidden_size]``.
        hc_scale (Tensor): Sub-block scales, shape ``[3]``.
        hc_base (Tensor): Per-slot bias, shape ``[(2 + hc_mult) * hc_mult]``.
        hc_mult (int): Number of streams (``H``).
        iters (int): Sinkhorn iterations.
        eps (float): Sinkhorn stabilizer.
        norm_eps (float): RMS-norm stabilizer applied before projecting to ``mixes``.

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - ``y`` (``[B, S, hidden_size]``): reduced stream consumed by the inner block.
            - ``post`` (``[B, S, hc_mult]``): post weights used by :func:`hc_post`.
            - ``comb`` (``[B, S, hc_mult, hc_mult]``): combination weights used by :func:`hc_post`.
    """
    shape, dtype = x.size(), x.dtype
    # HF's ``DeepseekV4HyperConnection.forward`` does the RMS rescale and the
    # ``hc_fn`` linear *entirely in fp32* (modeling_deepseek_v4.py:923-924):
    #
    #     flat = self.input_norm(hidden_streams.flatten(2).float())
    #     pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(...)
    #
    # Two precision paths gated by ``XTUNER_V4_HF_PARITY``:
    #   * Default (env unset / "0"): bf16 Linear with cuBLAS' fp32 accumulator.
    #     ~10-20× faster than the fp32 GEMM at K=16384/N=24 on H100
    #     (``sm80_xmma_gemm_f32f32_*`` has no tensor-core acceleration). Cost
    #     is one bf16 rounding of the GEMM inputs, propagating to ~1.5e-2 max
    #     abs diff at the layer output (below the already-accepted MoE
    #     cutlass GEMM diff of 2.7e-2). ``F.rms_norm`` keeps the fp32
    #     variance accumulator inside one fused ATen op so we never
    #     materialise a full-tensor fp32 copy.
    #   * ``XTUNER_V4_HF_PARITY=1``: keep both the RMS rescale and the Linear
    #     in fp32 to reproduce HF bitwise. This is the path the ``atol=0``
    #     parity tests exercise (``test_csa_parity_full_hf_anchor`` and
    #     friends, run with ``XTUNER_V4_HF_PARITY=1`` in the env).
    #
    # The downstream ``hc_split_sinkhorn`` runs in fp32 in both paths
    # (Sinkhorn 20-iter loop is bf16-NaN-prone), so the sinkhorn input is
    # ``mixes`` cast to fp32 explicitly here.
    x_flat = x.flatten(2)
    if _HC_HF_PARITY:
        x_flat_f32 = x_flat.float()
        rsqrt = torch.rsqrt(x_flat_f32.square().mean(-1, keepdim=True) + norm_eps)
        flat_normed = x_flat_f32 * rsqrt
        mixes = torch.nn.functional.linear(flat_normed, hc_fn.float())
    else:
        flat_normed = torch.nn.functional.rms_norm(
            x_flat, normalized_shape=(x_flat.size(-1),), weight=None, eps=norm_eps
        )
        mixes = torch.nn.functional.linear(flat_normed, hc_fn.to(dtype)).float()

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)

    # ``pre`` is fp32 (Sinkhorn output). Multiplying fp32 × bf16 in PyTorch's
    # TensorIterator path casts elementwise on read, so we don't need to
    # materialise a fp32 copy of ``x_flat`` here — the auto-promoted product
    # is fp32 of the same shape but the bf16 input stays bf16 in memory.
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
    return y.to(dtype), post, comb


def _unshard_hc_params(
    hc_fn: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Eager-mode helper: replace any DTensor-wrapped HC parameter with its
    full local tensor.

    Why this lives outside :func:`hc_pre`: FSDP shards every
    :class:`nn.Parameter` into a :class:`DTensor` and only unshards them under
    the pre-forward hook of the *enclosing* :class:`nn.Module`. The HC params
    live on the wrapper but are consumed by ``hc_pre``, so we materialise the
    local view here (cheap, ~100KB per layer) *before* the compile boundary
    instead of inside the compiled graph (which would graph-break ×3 per HC
    call: one per parameter).

    Args:
        hc_fn (Tensor): Mixing projection (potentially a DTensor).
        hc_scale (Tensor): Sub-block scales (potentially a DTensor).
        hc_base (Tensor): Per-slot bias (potentially a DTensor).

    Returns:
        tuple[Tensor, Tensor, Tensor]: Plain-tensor counterparts. Non-DTensor
        inputs are returned as-is.
    """
    from torch.distributed.tensor import DTensor as _DTensor

    if isinstance(hc_fn, _DTensor):
        hc_fn = hc_fn.full_tensor()
    if isinstance(hc_scale, _DTensor):
        hc_scale = hc_scale.full_tensor()
    if isinstance(hc_base, _DTensor):
        hc_base = hc_base.full_tensor()
    return hc_fn, hc_scale, hc_base


@maybe_compile
def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Expand the single-stream block output back into ``hc_mult`` streams.

    Port of ``Block.hc_post`` in DeepSeek-V4-Flash ``inference/model.py`` (L683-686):
    ``out[..., h, d] = post[..., h] * x[..., d] + sum_{h'} comb[..., h, h'] * residual[..., h', d]``.

    Args:
        x (Tensor): Inner-block output, shape ``[B, S, hidden_size]``.
        residual (Tensor): HC-expanded residual saved before :func:`hc_pre`,
            shape ``[B, S, hc_mult, hidden_size]``.
        post (Tensor): Post weights from :func:`hc_pre`, shape ``[B, S, hc_mult]``.
        comb (Tensor): Combination matrix from :func:`hc_pre`, shape ``[B, S, hc_mult, hc_mult]``.

    Returns:
        Tensor: Updated HC-expanded streams, shape ``[B, S, hc_mult, hidden_size]``,
            cast back to ``x.dtype``.
    """
    # post * x is broadcast across hidden dim; comb mixes across the residual stream axis.
    #
    # Implementation note. The natural form is::
    #
    #   mixed = torch.matmul(comb.to(residual.dtype), residual)
    #
    # but ``matmul`` (and ``einsum`` with the same contraction) lowers to
    # ``aten.bmm`` at the inductor level, and inductor delegates ``bmm`` to
    # cuBLAS via ``extern_kernels.bmm`` rather than codegening a triton kernel
    # — ``epilogue_fusion=True`` only fuses pointwise tails onto the matmul,
    # not the matmul body itself. Here the bmm has inner dim ``K = hc_mult =
    # 4`` which is below Hopper's wgmma tile floor (K=16), so cuBLAS falls
    # back to a CUDA-core path that's bandwidth-bound and slow — at
    # pack=16384 it cost ~3 ms per call × 86 calls/step ≈ 250 ms/step.
    #
    # To force inductor to codegen a fused triton kernel for the mixing, we
    # express it as broadcast-multiply + reduce-sum — both pointwise / reduce
    # primitives, which are inductor's strongest fusion targets. The
    # ``[B, S, H_out, H_in, D]`` intermediate that the unsqueeze + multiply
    # implies in eager mode **never materialises under compile**: inductor
    # fuses the multiply with the trailing ``.sum(dim=-2)`` into a single
    # triton kernel that does the H_in reduction in registers and writes only
    # the ``[B, S, H_out, D]`` output. The cast, the ``post * x`` broadcast
    # and the final add all join that kernel's epilogue, so the whole
    # ``hc_post`` body compiles to one fused kernel.
    #
    # WARNING: this function MUST be in the active compile cfg (see
    # ``_V4_LAYER_TARGETS`` in ``deepseek_v4.py``). Running it eagerly would
    # materialise the 8 GB ``[B, S, H, H, D]`` 5D tensor at pack=16384,
    # hc_mult=4, hidden=4096. The compile cfg covers it by default; the
    # ``hc_mult=1`` degenerate path in ``V4DecoderLayer.forward`` skips
    # ``hc_post`` entirely so unit tests with that setting don't hit this.
    #
    # ``comb`` is consumed *transposed* — HF / V4-ref compute
    # ``out[h_out, d] = sum_{h_in} comb[h_in, h_out] * residual[h_in, d]``
    # (i.e. the FIRST hc axis is the reduction axis, equivalent to
    # ``comb.T @ residual``). ``comb`` is doubly-stochastic from the
    # Sinkhorn projection but NOT symmetric, so the direction matters.
    # HF's matching expression in ``DeepseekV4DecoderLayer.forward``::
    #
    #     torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)
    #
    # We use exactly that ``torch.matmul`` here so cuBLAS' fp32-accumulator
    # bf16 GEMM gives the same precision as HF.
    #
    # Compile-speed note. At K = ``hc_mult`` = 4 this is below Hopper's
    # tensor-core tile floor; cuBLAS falls back to a CUDA-core CUDA gemm
    # which is bandwidth-bound and slow under heavy training schedules. An
    # earlier version of this code expanded ``comb`` over the inner dim and
    # summed in bf16 (``(comb_t.unsqueeze(-1) * residual.unsqueeze(-3)).sum(-2)``)
    # to dodge cuBLAS — that compiled to a fused triton kernel inductor was
    # happy with, *but* its bf16 reduction over the H_in axis lost ~1.2e-2
    # absolute precision vs HF (see ``test_subcomponent_probe``). The
    # parity contract was deemed more important than the K=4 speed
    # workaround; if the compile-speed regression becomes painful we'll
    # need an explicit fp32-accumulator broadcast+sum or a custom triton
    # kernel, not a precision compromise.
    # Cast ``post`` to residual's dtype BEFORE the broadcast multiply, matching
    # HF's exact statement: ``post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2)``.
    # XTuner used to leave ``post`` in fp32 (Sinkhorn output is fp32) and run
    # ``fp32 × bf16`` which auto-upcasts to fp32; the final ``+ mixed`` (bf16)
    # then needed a ``.type_as(x)`` cast. That extra-precision detour was
    # ~ULP-correct in isolation but accumulated differently than HF's all-bf16
    # path, giving a residual ~7e-3 abs diff at ``test_subcomponent_probe``'s
    # ``hc_post (attn)`` step. Casting up-front makes the whole expression
    # match HF bit-for-bit.
    #
    # Use broadcast-multiply + ``.sum(-2)`` (NOT ``torch.matmul``) so inductor
    # codegens one fused triton kernel for the whole hc_post body. At K=hc_mult=4
    # cuBLAS ``aten.bmm`` falls back to the CUDA-core path (below Hopper's
    # wgmma tile floor) and is bandwidth-bound — ~3 ms/call × 86 calls/step
    # ≈ 250 ms/step at pack=16384. The ``[B, S, H, H, D]`` 5D intermediate
    # the unsqueeze + multiply implies in eager mode does NOT materialise
    # under compile: inductor fuses the multiply with the trailing reduction
    # in registers and writes only the ``[B, S, H, D]`` output.
    #
    # The ``.transpose(-1, -2)`` is semantically required, NOT a perf
    # rearrangement — HF / V4-ref reduce over the FIRST hc axis of ``comb``
    # (``out[h_out, d] = sum_h_in comb[h_in, h_out] * residual[h_in, d]``,
    # i.e. ``comb.T @ residual``). Sinkhorn output is doubly stochastic but
    # asymmetric, so the un-transposed broadcast+sum was a different operation
    # entirely (~4.0 elementwise diff vs HF; not a precision drift).
    #
    # Reduction is in fp32 to match cuBLAS' bf16-input / fp32-accumulator bmm
    # bit-for-bit (within bf16 ULP). bf16 broadcast+sum here lost ~1.2e-2 vs
    # HF — see ``test_subcomponent_probe``'s ``hc_post (attn)`` step on the
    # earlier all-bf16 commit. The upcast costs ~1.5× HBM read on
    # comb/residual but inductor still fuses the whole hc_post body into one
    # triton kernel (the ``[B, S, H, H, D]`` 5D intermediate stays in
    # registers, never materialises), so the K=4 bandwidth-bound regime is
    # essentially unchanged in wall-clock — strictly better than the
    # ``torch.matmul`` path which falls back to cuBLAS' CUDA-core small-K
    # gemm (~250 ms/step at pack=16384).
    #
    # WARNING: this function MUST be in the active compile cfg (see
    # ``_V4_LAYER_TARGETS`` in ``deepseek_v4.py``). Running it eagerly would
    # materialise the 8 GB ``[B, S, H, H, D]`` 5D tensor at pack=16384,
    # hc_mult=4, hidden=4096.
    post_dt = post.to(residual.dtype)
    # ``comb`` arrives in fp32 (Sinkhorn output). HF first casts it to
    # ``residual.dtype`` (bf16) and then runs ``torch.matmul`` whose cuBLAS
    # backend uses an fp32 accumulator internally. To bit-match HF we
    # reproduce both halves: bf16 round on the input (one rounding boundary
    # that HF has and our fp32-throughout path would otherwise skip), then
    # promote to fp32 for the multiply + reduction.
    comb_b = comb.to(residual.dtype)
    mixed = (
        comb_b.float().transpose(-1, -2).unsqueeze(-1)  # [B, S, H_out, H_in, 1]
        * residual.float().unsqueeze(-3)                # [B, S, 1,     H_in, D]
    ).sum(dim=-2).to(residual.dtype)                     # → [B, S, H_out, D]  (sum over H_in)
    return post_dt.unsqueeze(-1) * x.unsqueeze(-2) + mixed
