# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from ..base import (
    BaseRLLossConfig,
    BaseRLLossContext,
    BaseRLLossKwargs,
    RLLossContextInputItem,
    compute_kl_loss_weight,
)
from ..loss_fn import get_policy_loss_fn, kl_penalty, sft_loss_fn
from ..utils import gather_logprobs


class OrealLossConfig(BaseRLLossConfig):
    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    positive_loss_factor: float = 1.0
    pos_sft_loss_weight: float = 1.0
    pos_policy_loss_weight: float = 1.0
    negative_loss_factor: float = 1.0

    @property
    def loss_ctx_cls(self) -> type["OrealLossContext"]:
        return OrealLossContext

    @property
    def loss_kwargs_cls(self) -> type["OrealLossKwargs"]:
        return OrealLossKwargs


class OrealLossKwargs(BaseRLLossKwargs):
    sft_loss_weight: torch.Tensor | None = None


class OrealLossContext(BaseRLLossContext):
    loss_cfg: OrealLossConfig
    loss_kwargs: OrealLossKwargs

    def __init__(self, loss_cfg: OrealLossConfig, loss_kwargs: OrealLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)
        self.policy_loss_fn = get_policy_loss_fn(self.loss_cfg.policy_loss_cfg.get("loss_type", "vanilla"))

    # TODO: this function is not used anymore, we should remove it later
    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[RLLossContextInputItem],
        loss_cfg: OrealLossConfig,
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list[OrealLossKwargs]:
        shifted_labels_list = [item.shifted_labels for item in data_batches]
        advantages_list = [item.advantages for item in data_batches]

        # Compute the denominator used in the global calibration of the loss
        rank_grad_tokens = sum((labels != loss_cfg.ignore_idx).sum() for labels in shifted_labels_list)
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        rank_positive_tokens = sum(
            ((labels != loss_cfg.ignore_idx) & (adv > 0)).sum()
            for labels, adv in zip(shifted_labels_list, advantages_list)
        )
        global_grad_tokens = rank_grad_tokens
        global_positive_tokens = rank_positive_tokens
        if dist.is_initialized():
            dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_positive_tokens, op=dist.ReduceOp.SUM)
        global_negative_tokens = global_grad_tokens - global_positive_tokens

        batches_loss_kwargs = []
        for i, item in enumerate(data_batches):
            shifted_labels = shifted_labels_list[i]
            advantages = advantages_list[i]
            assert item.old_logprobs is not None, "old_logprobs can not be None"
            # compute sft loss_weights
            # TODO: oreal 官方实现里 sft loss weights 要乘两个 loss factor，需要进一步 check 下
            sft_loss_weights = (
                torch.ones_like(shifted_labels, dtype=torch.float32)
                * loss_cfg.pos_sft_loss_weight
                * loss_cfg.positive_loss_factor
                / global_positive_tokens
            )
            sft_loss_weights[shifted_labels == loss_cfg.ignore_idx] = 0.0
            sft_loss_weights[advantages <= 0] = 0.0  # only positive advantages tokens contribute to sft loss

            # compute policy loss_weights
            policy_loss_weights = torch.ones_like(shifted_labels, dtype=torch.float32)
            policy_loss_weights[shifted_labels == loss_cfg.ignore_idx] = 0.0
            policy_loss_weights[advantages > 0] *= (
                loss_cfg.pos_policy_loss_weight * loss_cfg.positive_loss_factor / global_positive_tokens
            )
            policy_loss_weights[advantages <= 0] *= loss_cfg.negative_loss_factor / global_negative_tokens

            if item.is_weights is not None:
                policy_loss_weights = policy_loss_weights * item.is_weights

            # compute kl loss weights
            if loss_cfg.use_kl_loss and item.ref_logprobs is not None:
                # Maybe get ref_logprobs after init
                ref_logprobs = item.ref_logprobs
                kl_loss_weight = compute_kl_loss_weight(
                    shifted_labels,
                    global_grad_tokens,
                    loss_cfg.kl_loss_coef,
                    loss_cfg.ignore_idx,
                )
            else:
                ref_logprobs = None
                kl_loss_weight = None

            loss_kwargs = OrealLossKwargs(
                old_logprobs=item.old_logprobs,
                shifted_labels=shifted_labels,
                advantages=advantages,
                sft_loss_weight=sft_loss_weights,
                policy_loss_weight=policy_loss_weights,
                ref_logprobs=ref_logprobs,
                kl_loss_weight=kl_loss_weight,
                global_grad_tokens=global_grad_tokens,
            )
            batches_loss_kwargs.append(loss_kwargs)
        return batches_loss_kwargs

    @staticmethod
    def build_batches(loss_ctx_list: list["OrealLossContext"]) -> list["OrealLossContext"]:  # type: ignore[override]
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"

        loss_cfg = loss_ctx_list[0].loss_cfg

        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        advantages_list = [loss_ctx.loss_kwargs.advantages for loss_ctx in loss_ctx_list]

        # Compute the denominator used in the global calibration of the loss
        rank_grad_tokens = sum((labels != loss_cfg.ignore_idx).sum() for labels in shifted_labels_list)
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        rank_positive_tokens = sum(
            ((labels != loss_cfg.ignore_idx) & (adv > 0)).sum()
            for labels, adv in zip(shifted_labels_list, advantages_list)
        )
        global_grad_tokens = rank_grad_tokens
        global_positive_tokens = rank_positive_tokens
        if dist.is_initialized():
            dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_positive_tokens, op=dist.ReduceOp.SUM)
        global_negative_tokens = global_grad_tokens - global_positive_tokens

        for loss_ctx in loss_ctx_list:
            loss_kwargs = loss_ctx.loss_kwargs

            shifted_labels = loss_kwargs.shifted_labels
            advantages = loss_kwargs.advantages
            assert loss_kwargs.old_logprobs is not None, "old_logprobs can not be None"
            # compute sft loss_weights
            # TODO: oreal 官方实现里 sft loss weights 要乘两个 loss factor，需要进一步 check 下
            sft_loss_weights = (
                torch.ones_like(shifted_labels, dtype=torch.float32)
                * loss_cfg.pos_sft_loss_weight
                * loss_cfg.positive_loss_factor
                / global_positive_tokens
            )
            sft_loss_weights[shifted_labels == loss_cfg.ignore_idx] = 0.0
            sft_loss_weights[advantages <= 0] = 0.0  # only positive advantages tokens contribute to sft loss

            # compute policy loss_weights
            policy_loss_weights = torch.ones_like(shifted_labels, dtype=torch.float32)
            policy_loss_weights[shifted_labels == loss_cfg.ignore_idx] = 0.0
            policy_loss_weights[advantages > 0] *= (
                loss_cfg.pos_policy_loss_weight * loss_cfg.positive_loss_factor / global_positive_tokens
            )
            policy_loss_weights[advantages <= 0] *= loss_cfg.negative_loss_factor / global_negative_tokens

            if loss_kwargs.is_weights is not None:
                policy_loss_weights = policy_loss_weights * loss_kwargs.is_weights

            # compute kl loss weights
            if loss_cfg.use_kl_loss:
                assert loss_kwargs.ref_logprobs is not None, "ref_logprobs can not be None"
                kl_loss_weight = compute_kl_loss_weight(
                    shifted_labels,
                    global_grad_tokens,
                    loss_cfg.kl_loss_coef,
                    loss_cfg.ignore_idx,
                )
            else:
                kl_loss_weight = None

            loss_kwargs.sft_loss_weight = sft_loss_weights
            loss_kwargs.policy_loss_weight = policy_loss_weights
            loss_kwargs.kl_loss_weight = kl_loss_weight
            loss_kwargs.global_grad_tokens = global_grad_tokens
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: OrealLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Step 2.a and 2.b in the loss calculation in
        xtuner/v1/loss/base_loss_ctx.py."""
        # We do linear forward here to simplify the implementation of chunk loss (saving memory).
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = logits.float()

        shifted_labels = loss_kwargs.shifted_labels
        old_logprobs = loss_kwargs.old_logprobs
        advantages = loss_kwargs.advantages
        policy_loss_weight = loss_kwargs.policy_loss_weight
        sft_loss_weight = loss_kwargs.sft_loss_weight
        assert sft_loss_weight is not None, "sft_loss_weight can not be None"
        sft_loss = sft_loss_fn(logits, shifted_labels, sft_loss_weight, ignore_idx=self.loss_cfg.ignore_idx)
        logprobs = gather_logprobs(logits, shifted_labels)
        policy_loss = self.policy_loss_fn(
            logprobs,
            old_logprobs,
            advantages,
            policy_loss_weight,
            self.loss_cfg.policy_loss_cfg,
        )

        loss = sft_loss + policy_loss

        if self.loss_cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            kl_loss = kl_penalty(logprobs, ref_logprobs, kl_loss_weight, self.loss_cfg.kl_loss_type)
            loss = loss + kl_loss

        return loss, (logits, {})
