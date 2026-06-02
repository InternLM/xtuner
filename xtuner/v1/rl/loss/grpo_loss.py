# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xtuner.v1.utils import get_logger

from ..utils import gather_logprobs
from .base_loss import (
    BaseRLLossConfig,
    BaseRLLossContext,
    BaseRLLossKwargs,
    compute_kl_loss_weight,
)
from .loss_fn import get_policy_loss_fn, kl_penalty


logger = get_logger()


class GRPOLossConfig(BaseRLLossConfig):
    """Configuration for GRPO loss computation in XTuner RL.

    ``GRPOLossConfig`` implements the loss configuration used by Group Relative
    Policy Optimization. It consumes advantages computed for each rollout group
    and supports optional KL regularization through ``BaseRLLossConfig``.

    Args:
        policy_loss_cfg (dict[str, Any]): Configuration parameters for the main policy loss.
            Contains algorithm-specific parameters for policy optimization.
        use_kl_loss (bool): Whether to include KL divergence penalty in the loss.
            When True, requires a reference model for KL computation. Defaults to False.
        kl_loss_coef (float): Coefficient for weighting the KL divergence penalty.
            Controls the strength of regularization against the reference policy. Defaults to 0.001.
        kl_loss_type (Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None):
            Type of KL penalty computation method. Different types provide various
            regularization behaviors and numerical stability properties. Defaults to None.
        rollout_is (RolloutImportanceSampling): Rollout importance sampling
            configuration. Defaults to ``RolloutImportanceSampling()``.

    **Examples:**

    Example GRPO loss configuration::

        config = GRPOLossConfig(
            policy_loss_cfg={
                "loss_type": "vanilla",
                "cliprange_low": 0.2,
                "cliprange_high": 0.2,
            },
            use_kl_loss=True,
            kl_loss_coef=0.001,
            kl_loss_type="low_var_kl",
        )
    """

    @property
    def loss_ctx_cls(self) -> type["GRPOLossContext"]:
        return GRPOLossContext

    @property
    def _loss_kwargs_cls(self) -> type["GRPOLossKwargs"]:
        return GRPOLossKwargs


class GRPOLossKwargs(BaseRLLossKwargs):
    """Keyword arguments for GRPO loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        old_logprobs (torch.Tensor): Log probabilities from the old policy.
        advantages (torch.Tensor): Advantage estimates for the actions taken.
        policy_loss_weight (torch.Tensor): Weights for each token in the policy loss computation.
        ref_logprobs (torch.Tensor | None): Reference log probabilities for KL penalty, if used.
        kl_loss_weight (torch.Tensor | None): Weights for each token in the KL loss computation, if used.
    """

    pass


class GRPOLossContext(BaseRLLossContext):
    """GRPO loss context for reinforcement learning.

    Args:
        loss_cfg (GRPOLossConfig): Configuration for GRPO loss computation.
        loss_kwargs (GRPOLossKwargs): Keyword arguments required for loss calculation.
    """

    loss_cfg: GRPOLossConfig
    loss_kwargs: GRPOLossKwargs

    def __init__(self, loss_cfg: GRPOLossConfig, loss_kwargs: GRPOLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)
        self.policy_loss_fn = get_policy_loss_fn(self.loss_cfg.policy_loss_cfg.get("loss_type", "vanilla"))

    @staticmethod
    def build_batches(loss_ctx_list: list["GRPOLossContext"]) -> list["GRPOLossContext"]:  # type: ignore[override]
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"

        loss_cfg = loss_ctx_list[0].loss_cfg

        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]

        # Compute the denominator used in the global calibration of the loss
        rank_grad_tokens = sum((labels != loss_cfg.ignore_idx).sum() for labels in shifted_labels_list)
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        if dist.is_initialized():
            dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        if global_grad_tokens == 0:
            logger.warning(
                "Global gradient tokens is 0, which may lead to division by zero in loss weight calculation."
            )
            global_grad_tokens.add_(1)  # Avoid division by zero

        for loss_ctx in loss_ctx_list:
            loss_kwargs = loss_ctx.loss_kwargs

            shifted_labels = loss_kwargs.shifted_labels
            assert loss_kwargs.old_logprobs is not None, "old_logprobs can not be None"
            # compute loss weight
            policy_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens
            policy_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if loss_kwargs.is_weights is not None:
                policy_loss_weight = policy_loss_weight * loss_kwargs.is_weights
            if loss_cfg.use_kl_loss:
                assert loss_kwargs.ref_logprobs is not None, "ref_logprobs can not be None"
                kl_loss_weight = compute_kl_loss_weight(
                    shifted_labels, global_grad_tokens, loss_cfg.kl_loss_coef, loss_cfg.ignore_idx
                )
            else:
                kl_loss_weight = None
            loss_kwargs.policy_loss_weight = policy_loss_weight
            loss_kwargs.kl_loss_weight = kl_loss_weight
            loss_kwargs.global_grad_tokens = global_grad_tokens
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: GRPOLossKwargs,
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

        logprobs = gather_logprobs(logits, shifted_labels)
        loss = self.policy_loss_fn(
            logprobs,
            old_logprobs,
            advantages,
            policy_loss_weight,
            self.loss_cfg.policy_loss_cfg,
        )

        assert old_logprobs is not None
        valid_mask = shifted_labels != self.loss_cfg.ignore_idx
        valid_float = valid_mask.float()
        cliprange_low = self.loss_cfg.policy_loss_cfg.get("cliprange_low")
        cliprange_high = self.loss_cfg.policy_loss_cfg.get("cliprange_high")

        log_ratio = logprobs.detach() - old_logprobs.detach()
        log_ratio_safe = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio_safe)
        kl1 = -log_ratio
        kl3 = ratio - 1.0 - log_ratio_safe
        ratio_abs_dev = (ratio - 1.0).abs()
        ratio_max = ratio.masked_fill(~valid_mask, 0.0).max()
        ratio_min = ratio.masked_fill(~valid_mask, float("inf")).min()
        extra_info = {
            "max_ratio": ratio_max,
            "reduced_train_policy_ratio_abs_dev_sum": (ratio_abs_dev * valid_float).sum(),
            "reduced_train_policy_kl1_sum": (kl1 * valid_float).sum(),
            "reduced_train_policy_kl3_sum": (kl3 * valid_float).sum(),
            "reduced_train_policy_valid_count": valid_float.sum(),
            "reduced_train_policy_ratio_max": ratio_max,
            "reduced_train_policy_ratio_min": ratio_min,
        }
        if cliprange_low is not None and cliprange_high is not None:
            clip_low_mask = ratio < 1 - cliprange_low
            clip_high_mask = ratio > 1 + cliprange_high
            extra_info["reduced_train_policy_clip_low_count"] = (clip_low_mask & valid_mask).float().sum()
            extra_info["reduced_train_policy_clip_high_count"] = (clip_high_mask & valid_mask).float().sum()

        if self.loss_cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            kl_loss = kl_penalty(logprobs, ref_logprobs, kl_loss_weight, self.loss_cfg.kl_loss_type)
            loss = loss + kl_loss

        return loss, (logits, extra_info)
