# Copyright (c) OpenMMLab. All rights reserved.
#
# Portions of this file are derived from:
#   - DeepSeek-V4-Flash inference reference (MIT)
#       https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/kernel.py
#       hc_split_sinkhorn_kernel / hc_split_sinkhorn (TileLang JIT). We re-implement
#       the same numerical contract in pure PyTorch so training does not depend on
#       TileLang.
#   - lucidrains/hyper-connections (MIT, Phil Wang)
#       https://github.com/lucidrains/hyper-connections
#       commit e89e30357d1f79945b0d3558e6721451ff68789a
#       hyper_connections/mHCv2.py::sinkhorn_knopps (bf16-safe via fp32 upcast and
#       subtraction of `amax(dim=-2).detach()` before exp).
#
# Both upstreams are MIT-licensed. The combined file is released under the same
# license as the rest of XTuner (see LICENSE at the repository root).
"""Pure-PyTorch port of DeepSeek-V4-Flash ``hc_split_sinkhorn``.

Used by :func:`xtuner.v1.module.decoder_layer.hc_block.hc_pre` to compute the
``pre`` / ``post`` / ``comb`` mixing weights that
:class:`xtuner.v1.model.moe.deepseek_v4.V4DecoderLayer` consumes around its
attention and FFN sub-blocks.
"""

import torch
from torch import Tensor


def hc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Hyper-Connections ``pre``/``post``/``comb`` weights via split-
    sinkhorn.

    Matches the numerical contract of
    ``deepseek_v4_reference/kernel.py::hc_split_sinkhorn`` (TileLang JIT). The first
    ``hc_mult`` slots of ``mixes`` produce the per-stream ``pre`` weights, the next
    ``hc_mult`` slots produce ``post``, and the remaining ``hc_mult * hc_mult`` slots
    produce a doubly-stochastic ``comb`` matrix via ``iters`` rounds of
    Sinkhorn-Knopp normalization (row softmax + col norm, then alternating row/col).

    Computation is upcast to fp32 internally and cast back to the input dtype on
    output, which keeps the ``20`` Sinkhorn iterations stable under bf16.

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
    # Sinkhorn runs in fp32 to mirror the TileLang reference and avoid bf16 NaNs
    # from low-precision softmax + repeated divisions. Inputs are expected to
    # already be fp32: ``mixes`` comes from ``hc_block.hc_pre`` which casts the
    # Linear output to fp32 before passing here; ``hc_scale`` / ``hc_base`` are
    # fp32 ``nn.Parameter``s (the HC params are stored in fp32, see
    # :class:`V4DecoderLayer.__init__`). Older revisions called ``.float()`` on
    # each defensively; those calls were no-ops on the V4 path and only added
    # identity casts to the compile graph.
    assert mixes.dtype == torch.float32, f"hc_split_sinkhorn expects fp32 mixes; got {mixes.dtype}"

    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre = torch.sigmoid(pre_logits) + eps

    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post = 2.0 * torch.sigmoid(post_logits)

    comb_flat = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb_shape = comb_flat.shape[:-1] + (hc_mult, hc_mult)
    comb = comb_flat.reshape(comb_shape)

    # First iteration matches HF ``DeepseekV4HyperConnection.forward`` exactly
    # (modeling_deepseek_v4.py:931-932)::
    #
    #     comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    #     comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    #
    # An earlier manual ``comb - amax(-1).detach() + exp + / sum`` chain
    # produced the same softmax mathematically but in a different
    # reduction order — measured as ~6e-8 fp32 ULP diff vs HF's
    # ``torch.softmax`` in ``test_subcomponent_probe``'s ``hc_pre.comb``
    # step, which is enough to flip the rounding direction on a few bf16
    # cast boundaries downstream and break the
    # ``_full_hf_anchor`` (attention+MoE delegated to HF) test's
    # ``atol=0`` goal.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # Remaining (iters - 1) rounds: alternating row then column normalization,
    # each with the same `+ eps` stabilizer as the reference kernel.
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre.to(orig_dtype), post.to(orig_dtype), comb.to(orig_dtype)
