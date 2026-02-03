# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.loss.chunk_linear import FusedLinearForPPOFunction as ChunkLinear
from xtuner.v1.loss.utils import sp_gather, sp_gather_differentiable
from xtuner.v1.utils import get_logger

from ..base import (
    BaseRLLossConfig,
    BaseRLLossContext,
    BaseRLLossKwargs,
    compute_kl_loss_weight,
)
from ..loss_fn import get_policy_loss_fn, kl_penalty
from ..utils import gather_logprobs


logger = get_logger()


class GRPOLossConfig(BaseRLLossConfig):
    """Configuration for GRPO loss computation in XTuner RL.

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

    def __init__(self, loss_cfg: GRPOLossConfig, loss_kwargs: GRPOLossKwargs, sp_mesh: DeviceMesh | None = None):
        super().__init__(loss_cfg, loss_kwargs, sp_mesh)
        # TODO: Refactor it to PolicyLossFn and its Cfg. And support verl core algos seamlessly
        #       such as https://github.com/verl-project/verl/blob/de9880d76467af6bcb9b5c12fad6dfa980e83d57/verl/trainer/ppo/core_algos.py#L1254
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
        enable_chunk_linear: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        """Step 2.a and 2.b in the loss calculation in
        xtuner/v1/loss/base_loss_ctx.py."""
        # If enable chunk_linear mode, then all the activation tensors in this function will be sp_gathered.
        # So the policy loss and kl loss function will receive these whole sequence activations to calculate loss.

        shifted_labels = loss_kwargs.shifted_labels
        old_logprobs = loss_kwargs.old_logprobs
        advantages = loss_kwargs.advantages
        policy_loss_weight = loss_kwargs.policy_loss_weight
        rollout_logprobs = loss_kwargs.rollout_logprobs

        if not enable_chunk_linear:
            # We do linear forward here to simplify the implementation of chunk loss (saving memory).
            logits = F.linear(hidden_states, head_weight, head_bias)
            logits = logits.float()
            logprobs = gather_logprobs(logits, shifted_labels)
        else:
            assert head_bias is None, "head_bias must be None when enable_chunk_linear is True"

            # Use differentiable gather for hidden_states to allow gradient flow.
            # grad_scale="up" multiplies gradient by sp_size to compensate for `loss / sp_size` later,
            # because hidden_states gradient is split back to each rank (not duplicated like head_weight).
            # 在本函数最后 `loss = loss / self.sp_mesh.size()` 会影响所有梯度
            # 对于 head_weight：每个 SP rank 计算完整梯度，需要这个 /sp_size
            # 对于 hidden_states：split 后每个 rank 只拿自己那部分，不应该被 /sp_size 影响
            # 因此，需要在 sp_gather_differentiable 的 backward 中乘以 sp_size 来补偿，即 grad_scale="up"。
            hidden_states = sp_gather_differentiable(hidden_states, sp_mesh=self.sp_mesh, dim=1, grad_scale="up")
            shifted_labels = sp_gather(shifted_labels, sp_mesh=self.sp_mesh, dim=1)
            # clone shifted_labels to avoid modifying the original tensor which is used in loss calculation
            # and set ignore_idx to 0 to avoid index out of bounds error in ChunkLinear.apply
            chunk_linear_labels = shifted_labels.clone()
            chunk_linear_labels[chunk_linear_labels == self.loss_cfg.ignore_idx] = 0
            logprobs, entropy = ChunkLinear.apply(
                hidden_states, head_weight, chunk_linear_labels, 1.0, self.loss_cfg.chunk_size
            )
            logits = None

            rollout_logprobs = sp_gather(rollout_logprobs, sp_mesh=self.sp_mesh, dim=1)
            old_logprobs = sp_gather(old_logprobs, sp_mesh=self.sp_mesh, dim=1)
            advantages = sp_gather(advantages, sp_mesh=self.sp_mesh, dim=1)
            policy_loss_weight = sp_gather(policy_loss_weight, sp_mesh=self.sp_mesh, dim=1)

        loss = self.policy_loss_fn(
            logprobs,
            old_logprobs,
            advantages,
            policy_loss_weight,
            self.loss_cfg.policy_loss_cfg,
            enable_chunk_linear,
            self.loss_kwargs.num_tokens,
            shifted_labels,
        )

        assert old_logprobs is not None
        ratio = (logprobs - old_logprobs.detach()).exp()
        ratio = ratio * (shifted_labels != self.loss_cfg.ignore_idx).float()
        extra_info = {"max_ratio": ratio.max()}  # metrics for logging

        if self.loss_cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            if enable_chunk_linear:
                ref_logprobs = sp_gather(ref_logprobs, sp_mesh=self.sp_mesh, dim=1)
                kl_loss_weight = sp_gather(kl_loss_weight, sp_mesh=self.sp_mesh, dim=1)
            kl_loss = kl_penalty(logprobs, ref_logprobs, kl_loss_weight, self.loss_cfg.kl_loss_type)
            loss = loss + kl_loss

        if enable_chunk_linear and self.sp_mesh and self.sp_mesh.size() > 1:
            loss = loss / self.sp_mesh.size()

        return loss, (logits, extra_info)
