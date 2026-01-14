# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/fanshiqing/grouped_gemm/blob/v1.1.4/grouped_gemm/ops.py
Support torch compile."""

import os

import torch
from torch import Tensor

from ...comm.nvls_agrs import SymmBufferManager


try:
    from grouped_gemm import backend
except ImportError:
    backend = None

# TODO: (yehaochen) maybe replace inherit from `torch.autograd.Function` with `register_autograd`
USE_CUSTOM_RS = int(os.getenv("XTUNER_USE_CUSTOM_RS_IN_DISPATCHER", 0)) == 1


@torch.library.custom_op("moe::permute", mutates_args=())
def _permute(
    input_act: Tensor,
    indices: Tensor,
    num_topK: int,
    num_out_tokens: int,
    num_negative_one_in_indices: int,
) -> tuple[Tensor, Tensor]:
    input_max_expanded_token_num = input_act.size(0) * num_topK
    workspace_fw: list[Tensor] = []
    permuted_act, row_id_map, _ = backend.permute(
        input_act,
        indices,
        num_out_tokens,
        workspace_fw,
        input_max_expanded_token_num,
        num_negative_one_in_indices,
    )
    return permuted_act, row_id_map


@_permute.register_fake
def _(
    input_act: Tensor,
    indices: Tensor,
    num_topK: int,
    num_out_tokens: int,
    num_negative_one_in_indices: int,
):
    permuted_act = input_act.new_empty((input_act.shape[0] * num_topK, *input_act.shape[1:]))
    row_id_map = indices.new_empty((indices.numel(),))
    return permuted_act, row_id_map


@torch.library.custom_op("moe::unpermute", mutates_args=())
def _unpermute(input: Tensor, row_id_map: Tensor, prob: Tensor, max_tokens: int, num_topK: int) -> Tensor:
    if not input.is_contiguous():
        input = input.contiguous()
    return backend.unpermute(input, row_id_map, prob, max_tokens, num_topK)


@_unpermute.register_fake
def _(input: Tensor, row_id_map: Tensor, prob: Tensor, max_tokens: int, num_topK: int) -> Tensor:
    return input.new_empty((input.shape[0] // num_topK, *input.shape[1:]))


# NOTE:
# We intentionally DO NOT expose `backend.unpermute_inplace` as a `torch.library.custom_op`
# (i.e., with `mutates_args=("output",)`) for the symmetric-buffer path.
#
# Reason: in `UnpermuteMoE_topK.forward` we write into a view of a reusable global symmetric
# buffer (`agrs.rs_symm.get_buffer(...).view(...)[...].view(...)`) and may return that tensor.
# From PyTorch's perspective this is "a view created inside a custom Function whose base (or
# another view of its base) gets modified inplace later". Autograd/functionalization/torch.compile
# must conservatively assume buffer reuse can happen at any time, which triggers:
#   RuntimeError: Output ... is a view and its base ... has been modified inplace ...
# The framework forbids this pattern because the view+inplace aliasing logic could override the
# custom backward and lead to incorrect gradients.
#
# We therefore keep this call as a plain Python wrapper (`_unpermute_inplace`) instead of a
# `custom_op`, so it is not subjected to the strict alias/mutation tracking imposed by the
# dispatcher for `custom_op`s.
#
# Safety assumption (required by design): the returned `unpermuted_output` is treated as a
# short-lived ephemeral buffer. The symmetric buffer manager guarantees that the underlying
# storage will not be written inplace again until all consumers of `unpermuted_output` have
# finished using it (the actual values of `unpermuted_output` do not affect the final forward
# result; it is only a staging tensor on the custom reduce-scatter path).
#
# Additionally, this `unpermute_inplace` call is not expected to be captured by `torch.compile`
# (it runs outside the compiled region as part of the dispatcher/custom RS path), so keeping it
# as a Python wrapper (rather than a `custom_op`) is acceptable and avoids compiler/dispatcher
# aliasing restrictions.


# @torch.library.custom_op("moe::unpermute_inplace", mutates_args=("output",))
# def _unpermute_inplace(input: Tensor, output: Tensor, row_id_map: Tensor, prob: Tensor, max_tokens: int, num_topK: int) -> None:
#     if not input.is_contiguous():
#         input = input.contiguous()

#     backend.unpermute_inplace(input, output, row_id_map, prob, max_tokens, num_topK)


# @_unpermute_inplace.register_fake
# def _(input: Tensor, output: Tensor, row_id_map: Tensor, prob: Tensor, max_tokens: int, num_topK: int) -> None:
#     return


def _unpermute_inplace(
    input: Tensor, output: Tensor, row_id_map: Tensor, prob: Tensor | None, max_tokens: int, num_topK: int
) -> None:
    if not input.is_contiguous():
        input = input.contiguous()

    backend.unpermute_inplace(input, output, row_id_map, prob, max_tokens, num_topK)


@torch.library.custom_op("moe::unpermute_bwd", mutates_args=())
def _unpermute_bwd(
    input_bwd: Tensor,
    input_fwd: Tensor,
    row_id_map: Tensor,
    prob: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if not input_bwd.is_contiguous():
        input_bwd = input_bwd.contiguous()
    topk = input_fwd.shape[0] // input_bwd.shape[0]
    if prob is None:
        prob = torch.ones([input_bwd.size(0), topk], dtype=torch.float32, device=input_bwd.device)
    return backend.unpermute_bwd(input_bwd, input_fwd, row_id_map, prob)


@_unpermute_bwd.register_fake
def _(
    input_bwd: Tensor,
    input_fwd: Tensor,
    row_id_map: Tensor,
    prob: Tensor,
) -> tuple[Tensor, Tensor]:
    act_grad = torch.empty_like(input_fwd)
    topk = input_fwd.shape[0] // input_bwd.shape[0]
    prob_grad = torch.empty((input_bwd.size(0), topk), dtype=torch.float32, device=input_bwd.device)
    return act_grad, prob_grad


class PermuteMoE_topK(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_act: Tensor,
        indices: Tensor,
        num_out_tokens: int = 0,
        num_negative_one_in_indices: int = 0,
    ):
        if not input_act.numel():
            return input_act, None

        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)

        if indices.dim() == 1:
            indices = indices.view(-1, 1)
        if not input_act.is_contiguous():
            input_act = input_act.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()

        num_topK = indices.size(1)

        permuted_act, row_id_map = _permute(
            input_act,
            indices,
            num_topK,
            num_out_tokens,
            num_negative_one_in_indices,
        )

        ctx.row_id_map = row_id_map
        ctx.num_tokens = indices.size(0)
        ctx.num_topK = num_topK
        return permuted_act, row_id_map

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, permuted_act_grad: Tensor, row_id_map_grad: None
    ) -> tuple[Tensor, None, None, None]:
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, None

        permuted_act_grad = permuted_act_grad.contiguous()

        row_id_map = ctx.row_id_map
        num_tokens = ctx.num_tokens
        num_topK = ctx.num_topK

        if USE_CUSTOM_RS:
            import xtuner.v1.module.dispatcher.agrs as agrs

            # We keep a module-level global `rs_symm` (in `xtuner.v1.module.dispatcher.agrs`) as a shared
            # symmetric-memory buffer manager used by the dispatcher reduce-scatter path.
            #
            # Why do we set `agrs.rs_symm` here?
            # - The custom reduce-scatter kernel requires its input tensors to be allocated in symmetric memory.
            # - `SymmBufferManager` hands out such symmetric buffers, so we must initialize it before requesting buffers.
            # - Storing it as `agrs.rs_symm` (a module attribute) makes the instance shared and reusable across calls
            #   and across different modules that import `agrs`, avoiding repeated allocations and redundant memcpy.
            # - We only initialize it once (lazy init) to reduce startup overhead and because the required size
            #   can depend on runtime environment variables (e.g., `SYMM_BUF_SIZE`) and actual tensor shapes.
            if agrs.rs_symm is None:
                agrs.rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
            send_bytes = permuted_act_grad.numel() * permuted_act_grad.element_size()
            device = permuted_act_grad.device
            dtype = permuted_act_grad.dtype
            send_numel = permuted_act_grad.numel()
            symm_input = agrs.rs_symm.get_buffer(bytes=send_bytes, device=device)

            unpermuted_act_grad = symm_input.view(dtype)[:send_numel].view(permuted_act_grad.shape)
            _unpermute_inplace(
                permuted_act_grad, unpermuted_act_grad, row_id_map, torch.tensor([]), num_tokens, num_topK
            )
        else:
            unpermuted_act_grad = _unpermute(permuted_act_grad, row_id_map, torch.tensor([]), num_tokens, num_topK)

        return unpermuted_act_grad, None, None, None


class UnpermuteMoE_topK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_act: Tensor, row_id_map: Tensor, probs: Tensor | None = None):
        if not input_act.numel():
            ctx.probs = probs
            return input_act

        if not input_act.is_contiguous():
            input_act = input_act.contiguous()
        if not row_id_map.is_contiguous():
            row_id_map = row_id_map.contiguous()
        if probs is not None and not probs.is_contiguous():
            probs = probs.contiguous()

        if probs is not None and probs.dtype != torch.float32:
            probs = probs.to(torch.float32)

        num_tokens = probs.size(0) if probs is not None else input_act.size(0)
        num_topK = probs.size(1) if probs is not None else 1

        if USE_CUSTOM_RS:
            import xtuner.v1.module.dispatcher.agrs as agrs

            # We keep a module-level global `rs_symm` (in `xtuner.v1.module.dispatcher.agrs`) as a shared
            # symmetric-memory buffer manager used by the dispatcher reduce-scatter path.
            #
            # Why do we set `agrs.rs_symm` here?
            # - The custom reduce-scatter kernel requires its input tensors to be allocated in symmetric memory.
            # - `SymmBufferManager` hands out such symmetric buffers, so we must initialize it before requesting buffers.
            # - Storing it as `agrs.rs_symm` (a module attribute) makes the instance shared and reusable across calls
            #   and across different modules that import `agrs`, avoiding repeated allocations and redundant memcpy.
            # - We only initialize it once (lazy init) to reduce startup overhead and because the required size
            #   can depend on runtime environment variables (e.g., `SYMM_BUF_SIZE`) and actual tensor shapes.
            if agrs.rs_symm is None:
                agrs.rs_symm = SymmBufferManager(int(os.getenv("SYMM_BUF_SIZE", 0)), num_buffers=1)
            send_bytes = input_act.numel() * input_act.element_size()
            device = input_act.device
            dtype = input_act.dtype
            send_numel = input_act.numel()
            symm_input = agrs.rs_symm.get_buffer(bytes=send_bytes, device=device)

            symm_input = symm_input.view(dtype)[:send_numel].view(input_act.shape)
            unpermuted_output = symm_input
            _unpermute_inplace(
                input_act,
                unpermuted_output,
                row_id_map,
                probs,
                num_tokens,
                num_topK,
            )
        else:
            unpermuted_output = _unpermute(
                input_act,
                row_id_map,
                probs if probs is not None else torch.tensor([]),
                num_tokens,
                num_topK,
            )

        ctx.save_for_backward(input_act, row_id_map, probs)
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad: Tensor) -> tuple[Tensor | None, None, Tensor | None]:  # type: ignore[override]
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.probs

        input_act, row_id_map, probs = ctx.saved_tensors

        act_grad = None
        prob_grad = None
        if ctx.needs_input_grad[0]:
            act_grad, prob_grad = _unpermute_bwd(unpermuted_act_grad, input_act, row_id_map, probs)

        if not ctx.needs_input_grad[2]:
            prob_grad = None

        return act_grad, None, prob_grad


def cuda_token_permute(
    input_act, indices, num_topK: int | None = None, num_out_tokens=0, num_negative_one_in_indices=0
) -> tuple[Tensor, Tensor]:
    return PermuteMoE_topK.apply(input_act, indices, num_out_tokens, num_negative_one_in_indices)  # type: ignore[return-value]


def cuda_token_unpermute(input_act, row_id_map, probs=None) -> Tensor:
    return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)  # type: ignore[return-value]


def cuda_token_permute_torch(
    input_act, indices, num_topK: int | None = None, num_out_tokens=0, num_negative_one_in_indices=0
) -> tuple[Tensor, Tensor]:
    assert num_topK is None and num_out_tokens == 0 and num_negative_one_in_indices == 0, (
        f"Unexpected arguments: num_topK={num_topK}, num_out_tokens={num_out_tokens}, num_negative_one_in_indices={num_negative_one_in_indices}"
    )
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)

    permuted_tokens = input_act.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def cuda_token_unpermute_torch(
    input_act: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> Tensor:
    assert row_id_map.numel() == input_act.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probs
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = input_act.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, input_act.shape[-1]],
        dtype=input_act.dtype,
        device=input_act.device,
    )
    unpermuted_tokens.index_put_((row_id_map,), input_act, accumulate=False)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, input_act.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens.to(input_act.dtype)
