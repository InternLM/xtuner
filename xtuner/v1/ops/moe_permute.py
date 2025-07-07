# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/fanshiqing/grouped_gemm/blob/v1.1.4/grouped_gemm/ops.py
Support torch compile."""

import torch
from grouped_gemm import backend
from torch import Tensor


# TODO: (yehaochen) maybe replace inherit from `torch.autograd.Function` with `register_autograd`


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

        if indices.dtype is torch.int32:
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


def permute(input_act, indices, num_out_tokens=0, num_negative_one_in_indices=0) -> tuple[Tensor, Tensor]:
    return PermuteMoE_topK.apply(input_act, indices, num_out_tokens, num_negative_one_in_indices)  # type: ignore[return-value]


def unpermute(input_act, row_id_map, probs=None) -> Tensor:
    return UnpermuteMoE_topK.apply(input_act, row_id_map, probs)  # type: ignore[return-value]
