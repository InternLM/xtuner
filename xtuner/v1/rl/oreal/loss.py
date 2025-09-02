# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.loss import BaseLossConfig, BaseLossContext, BaseLossKwargs

from ..grpo.loss import RLLossContextInputItem
from ..loss_fn import get_policy_loss_fn, kl_penalty, sft_loss_fn
from ..utils import gather_logprobs


class OrealLossConfig(BaseLossConfig):
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


class OrealLossKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    policy_loss_weight: torch.Tensor
    ref_logprobs: torch.Tensor | None = None
    kl_loss_weight: torch.Tensor | None = None
    sft_loss_weight: torch.Tensor


class OrealLossContext(BaseLossContext[RLLossContextInputItem]):
    loss_cfg: OrealLossConfig
    loss_kwargs: OrealLossKwargs

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

            # compute kl loss weights
            if loss_cfg.use_kl_loss:
                assert item.ref_logprobs is not None, "ref_logprobs can not be None when use_kl_loss=True"
                ref_logprobs = item.ref_logprobs
                kl_loss_weight = (
                    torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens * loss_cfg.kl_loss_coef
                )
                kl_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
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
            )
            batches_loss_kwargs.append(loss_kwargs)
        return batches_loss_kwargs

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: OrealLossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        sft_loss = sft_loss_fn(logits, shifted_labels, sft_loss_weight, ignore_idx=self.loss_cfg.ignore_idx)
        logprobs = gather_logprobs(logits, shifted_labels)
        policy_loss_fn = get_policy_loss_fn(self.loss_cfg.policy_loss_cfg.get("loss_type", "vanilla"))
        policy_loss = policy_loss_fn(
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

        return loss, logits
