# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss import BaseLossConfig, BaseLossContext, BaseLossKwargs
from xtuner.v1.utils import get_logger

from ..loss_fn import get_policy_loss_fn, kl_penalty
from ..utils import gather_logprobs, sp_split


logger = get_logger()


class GRPOLossConfig(BaseLossConfig):
    policy_loss_cfg: dict[str, Any]
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None

    @property
    def loss_ctx_cls(self) -> type["GRPOLossContext"]:
        return GRPOLossContext


class GRPOLossKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    policy_loss_weight: torch.Tensor
    ref_logprobs: torch.Tensor | None = None
    kl_loss_weight: torch.Tensor | None = None


class RLLossContextInputItem(BaseModel):
    model_config = ConfigDict(title="RLLossContextInputItem", extra="allow", arbitrary_types_allowed=True)
    shifted_labels: torch.Tensor
    advantages: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        advantages = sp_split(self.advantages, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        # 这里不用对old_logprobs和ref_logprobs进行sp_split，因为他是模型 fwd 生成的
        # 模型 fwd 前一定会对 seq_ctx 进行 sp_split
        return type(self)(
            shifted_labels=shifted_labels,
            advantages=advantages,
            old_logprobs=self.old_logprobs,
            ref_logprobs=self.ref_logprobs,
        )

    def to(self, device: torch.device | str) -> Self:
        self.shifted_labels = self.shifted_labels.to(device)
        self.advantages = self.advantages.to(device)
        if self.old_logprobs is not None:
            self.old_logprobs = self.old_logprobs.to(device)
        if self.ref_logprobs is not None:
            self.ref_logprobs = self.ref_logprobs.to(device)
        return self


class GRPOLossContext(BaseLossContext[RLLossContextInputItem]):
    loss_cfg: GRPOLossConfig
    loss_kwargs: GRPOLossKwargs

    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[RLLossContextInputItem],
        loss_cfg: GRPOLossConfig,
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list[GRPOLossKwargs]:
        shifted_labels_list = [item.shifted_labels for item in data_batches]

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

        batches_loss_kwargs = []
        for i, item in enumerate(data_batches):
            shifted_labels = shifted_labels_list[i]
            advantages = item.advantages
            assert item.old_logprobs is not None, "old_logprobs can not be None"
            # compute loss weight
            policy_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens
            policy_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if loss_cfg.use_kl_loss:
                assert item.ref_logprobs is not None, "ref_logprobs can not be None when use_kl_loss=True"
                ref_logprobs = item.ref_logprobs
                kl_loss_weight = policy_loss_weight.clone() * loss_cfg.kl_loss_coef
            else:
                ref_logprobs = None
                kl_loss_weight = None
            loss_kwargs = GRPOLossKwargs(
                old_logprobs=item.old_logprobs,
                shifted_labels=shifted_labels,
                advantages=advantages,
                policy_loss_weight=policy_loss_weight,
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
        loss_kwargs: GRPOLossKwargs,
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

        logprobs = gather_logprobs(logits, shifted_labels)
        policy_loss_fn = get_policy_loss_fn(self.loss_cfg.policy_loss_cfg.get("loss_type", "vanilla"))
        loss = policy_loss_fn(
            logprobs,
            old_logprobs,
            advantages,
            policy_loss_weight,
            self.loss_cfg.policy_loss_cfg,
        )

        if self.loss_cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            kl_loss = kl_penalty(logprobs, ref_logprobs, kl_loss_weight, self.loss_cfg.kl_loss_type)
            loss = loss + kl_loss

        return loss, logits
