# Copyright (c) OpenMMLab. All rights reserved.
#
# The structural reference for the Hyper-Connections wrapper (``hc_pre`` / ``hc_post``
# semantics, parameter shapes, and per-block forward order) is DeepSeek-V4-Flash's
# ``inference/model.py::Block`` (MIT-licensed), ported into XTuner for DeepSeek-V4 in
# ``xtuner/v1/module/decoder_layer/deepseek_v4/{hc_block,hc_sinkhorn}.py`` (dsv4 branch,
# commit 01c31a833). GLM-5.3-Flash's mHC uses the identical math (confirmed against HF's
# ``transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextHyperConnection`` /
# ``Glm5NextTextDecoderLayer.forward``), so this module ports that implementation as the
# shared, model-agnostic ``xtuner/v1/module/decoder_layer/mhc.py`` the design doc calls for
# (doc/xtuner_glm5p3flash_design.md F4), dropping the DeepSeek-V4-specific
# ``XTUNER_V4_HF_PARITY`` global toggle and the optional ``quack`` fused-RMSNorm fast path
# (neither exists on this branch; the eager fallback here is already bit-faithful to HF).
"""Manifold-Constrained Hyper-Connections (mHC) primitives.

The HC machinery keeps ``hc_mult`` copies of the hidden state and replaces the plain
``x = x + block(norm(x))`` residual with a learned mix:

1. :func:`hc_pre` reduces the ``hc_mult`` streams to one weighted stream that an attention
   or FFN sub-block consumes.
2. :func:`hc_post` re-expands the sub-block output into ``hc_mult`` streams using a learned
   doubly-stochastic combination of the original streams plus the sub-block output.

Consumed by ``xtuner/v1/model/moe/glm53/decoder_layer.py``, which wraps its attention and
FFN sub-blocks with this ``hc_pre`` / sub_block / ``hc_post`` pattern.
"""

import os

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from xtuner.v1.utils.compile import maybe_compile


__all__ = ["MHCConfig", "hc_split_sinkhorn", "hc_pre", "hc_post"]

# Opt-in switch for the DeepSeek TileKernels ``mhc`` backend (Hopper TileLang; see
# https://github.com/deepseek-ai/TileKernels). Not yet ported to this branch -- porting the
# 721-line xtuner/v1/ops/mhc.py blind, without hardware to validate it against, would trade
# a known-correct eager/Triton path for an unverifiable one. Left as a documented gap (see
# doc/progress.md F4) rather than silently wired to a nonexistent module.
_USE_MHC_KERNELS = os.getenv("XTUNER_USE_MHC_KERNELS", "0") == "1"


class MHCConfig(BaseModel):
    """Configuration for the mHC residual-mix pattern.

    Mirrors the three HC-related fields of the GLM-5.3-Flash config: ``hc_mult``,
    ``hc_eps``, ``hc_sinkhorn_iters``.

    Args:
        hc_mult (int): Number of hyper-connection streams. ``1`` makes the HC math
            degenerate to a plain pre-norm residual block (used as a structural parity
            anchor in tests).
        hc_eps (float): Stabilizer used inside the Sinkhorn normalization.
        hc_sinkhorn_iters (int): Number of Sinkhorn iterations.
    """

    model_config = ConfigDict(extra="forbid")

    hc_mult: int
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20


def hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute mHC ``pre``/``post``/``comb`` weights via split-sinkhorn.

    Matches HF's ``Glm5NextTextHyperConnection.forward`` bitwise: the first ``hc_mult``
    slots of ``mixes`` produce the per-stream ``pre`` weights, the next ``hc_mult`` slots
    produce ``post``, and the remaining ``hc_mult * hc_mult`` slots produce a
    doubly-stochastic ``comb`` matrix via ``iters`` rounds of Sinkhorn-Knopp normalization
    (row softmax + col norm, then alternating row/col).

    Computation is upcast to fp32 internally and cast back to the input dtype on output,
    which keeps the ``20`` Sinkhorn iterations stable under bf16.

    Args:
        mixes (Tensor): Pre-activation mixing scores, shape ``[..., (2 + hc_mult) * hc_mult]``.
        hc_scale (Tensor): Three scalars scaling the pre/post/comb sub-blocks, shape ``[3]``.
        hc_base (Tensor): Per-slot bias, shape ``[(2 + hc_mult) * hc_mult]``.
        hc_mult (int): Number of hyper-connection streams (``H``).
        iters (int): Number of Sinkhorn iterations on the ``comb`` block.
        eps (float): Stabilizer added to ``pre`` and to row/col sums during Sinkhorn.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(pre, post, comb)`` where
            - ``pre`` has shape ``[..., hc_mult]`` (sigmoid-gated, plus ``eps``),
            - ``post`` has shape ``[..., hc_mult]`` (``2 * sigmoid``, no eps),
            - ``comb`` has shape ``[..., hc_mult, hc_mult]`` (doubly-stochastic +
              ``eps``-stabilized).
    """
    orig_dtype = mixes.dtype
    assert mixes.dtype == torch.float32, f"hc_split_sinkhorn expects fp32 mixes; got {mixes.dtype}"

    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre = torch.sigmoid(pre_logits) + eps

    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post = 2.0 * torch.sigmoid(post_logits)

    comb_flat = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb_shape = comb_flat.shape[:-1] + (hc_mult, hc_mult)
    comb = comb_flat.reshape(comb_shape)

    # First iteration matches HF ``Glm5NextTextHyperConnection.forward`` exactly: using
    # ``torch.softmax`` (not a hand-rolled amax/exp/sum chain) matters -- an earlier
    # mathematically-equivalent manual softmax differed from HF by ~6e-8 fp32 ULP, enough to
    # flip a bf16 rounding boundary downstream (see the DeepSeek-V4 port's hc_sinkhorn.py).
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre.to(orig_dtype), post.to(orig_dtype), comb.to(orig_dtype)


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

    Faithful port of HF's ``Glm5NextTextHyperConnection.forward``: apply an unweighted
    RMS rescale to the flattened streams, project to ``mixes`` via ``hc_fn``, run
    :func:`hc_split_sinkhorn`, then take a weighted sum over the stream axis with ``pre``
    as the weights.

    The HC parameters are expected to arrive as plain :class:`torch.Tensor` (not
    :class:`DTensor`); the enclosing decoder layer materializes the locals via
    :func:`~xtuner.v1.utils.dtensor.materialize_full` outside the compile boundary, so this
    function stays a clean compile region without one graph break per parameter.

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
    x_flat = x.flatten(2)
    # HF's `Glm5NextTextUnweightedRMSNorm` rescales in fp32 (no learned weight), then the
    # `fn` projection runs on the *input* dtype and only the sinkhorn split is upcast --
    # matching `F.linear(flat_normed, hc_fn.to(dtype)).float()` below.
    flat_normed = torch.nn.functional.rms_norm(x_flat.float(), (x_flat.size(-1),), weight=None, eps=norm_eps).to(dtype)
    mixes = torch.nn.functional.linear(flat_normed, hc_fn.to(dtype)).float()

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)

    # Weighted reduce over the hc_mult axis: y[..., d] = sum_h pre[..., h] * x[..., h, d].
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
    return y.to(dtype), post, comb


def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Expand the single-stream block output back into ``hc_mult`` streams.

    ``out[..., h, d] = post[..., h] * x[..., d] + sum_{h'} comb[..., h', h] * residual[..., h', d]``,
    matching HF's ``Glm5NextTextDecoderLayer.forward``::

        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

    ``comb`` is doubly-stochastic but **not symmetric**, so the ``transpose(-1, -2)`` (the
    reduction runs over the *first* hc axis of ``comb``) is semantically required, not a
    perf rearrangement.

    Dispatches to a fused Triton kernel (:func:`xtuner.v1.ops.hc_post.hc_post_fused`) on the
    default bf16/CUDA path: the naive broadcast-multiply + reduce-sum below re-reads
    ``residual`` once per output stream, which is HBM-bound at pack=16384; the Triton kernel
    reads ``residual`` once per token and does the ``hc_mult x hc_mult`` mix in registers.
    Non-CUDA / non-bf16 inputs fall back to the eager path.

    Args:
        x (Tensor): Inner-block output, shape ``[B, S, hidden_size]``.
        residual (Tensor): HC-expanded residual saved before :func:`hc_pre`, shape
            ``[B, S, hc_mult, hidden_size]``.
        post (Tensor): Post weights from :func:`hc_pre`, shape ``[B, S, hc_mult]``.
        comb (Tensor): Combination matrix from :func:`hc_pre`, shape
            ``[B, S, hc_mult, hc_mult]``.

    Returns:
        Tensor: Updated HC-expanded streams, shape ``[B, S, hc_mult, hidden_size]``, cast
            back to ``x.dtype``.
    """
    if _USE_MHC_KERNELS:
        raise NotImplementedError(
            "XTUNER_USE_MHC_KERNELS=1 requests the DeepSeek TileKernels mhc backend, which "
            "has not been ported to this branch yet (see doc/progress.md F4). Unset the env "
            "var to use the default Triton-fused / eager hc_post."
        )
    if residual.is_cuda and residual.dtype == torch.bfloat16 and _hc_post_fused_available():
        from xtuner.v1.ops.hc_post import hc_post_fused

        return hc_post_fused(x, residual, post, comb)
    return _hc_post_eager(x, residual, post, comb)


def _hc_post_fused_available() -> bool:
    # Imported lazily so a Triton-less build (CPU-only unit tests) can still import this
    # module; the fast path is simply never taken there.
    try:
        from xtuner.v1.ops.hc_post import is_available

        return is_available()
    except ImportError:
        return False


@maybe_compile
def _hc_post_eager(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """Eager fp32-accumulate ``hc_post``, bit-faithful to HF's ``torch.matmul``
    expression.

    WARNING: this function MUST stay in the active compile cfg. Running it eagerly would
    materialize the ``[B, S, H, H, D]`` intermediate the broadcast-multiply implies -- at
    pack=16384, ``hc_mult=4``, ``hidden_size=4096`` that is ~8 GB. Under compile, inductor
    fuses the multiply with the trailing ``sum(dim=-2)`` into a single kernel that never
    materializes it. Both GLM-5.3-Flash tables in ``xtuner/v1/model/moe/glm53/glm53.py`` list
    it by name, including the EP one, which drops the enclosing decoder-layer boundary.
    """
    post_dt = post.to(residual.dtype)
    comb_b = comb.to(residual.dtype)
    mixed = (
        (
            comb_b.float().transpose(-1, -2).unsqueeze(-1)  # [B, S, H_out, H_in, 1]
            * residual.float().unsqueeze(-3)  # [B, S, 1,     H_in, D]
        )
        .sum(dim=-2)
        .to(residual.dtype)
    )  # -> [B, S, H_out, D]  (sum over H_in)
    return post_dt.unsqueeze(-1) * x.unsqueeze(-2) + mixed
