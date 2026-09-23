# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.loss.base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class CriticLossConfig(BaseLossConfig):
    """Configuration for PPO critic clipped value-loss.

    Unlike GRPO / PPO policy loss, this config does not inherit
    BaseRLLossConfig. It is used by ValueHead, not the language-model
    vocabulary head.

    Args:
        cliprange_value (float): Clip range applied to the difference between
            the current value and old_values. Defaults to 0.2.
        ignore_idx (int): Token index excluded from the loss. Defaults to -100.
    """

    cliprange_value: float = 0.2

    @property
    def loss_ctx_cls(self) -> type["CriticLossContext"]:
        return CriticLossContext

    @property
    def _loss_kwargs_cls(self) -> type["CriticLossKwargs"]:
        return CriticLossKwargs

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
    ) -> "CriticLossContext | None":
        """Build critic loss context from data dict.

        Args:
            data (dict): Data dictionary containing critic-specific fields:
                - shifted_labels (torch.Tensor): The shifted labels
                - returns (torch.Tensor): Per-token regression targets
                - old_values (torch.Tensor | None): Frozen values for clipping
                  (optional, can be omitted for SFT MSE)
            sp_mesh (DeviceMesh | None): Sequence parallel device mesh

        Returns:
            CriticLossContext | None: The built loss context, or None if required
            fields are missing.
        """
        if "shifted_labels" not in data or "returns" not in data:
            return None

        shifted_labels = data["shifted_labels"]
        returns = data["returns"]
        old_values = data.get("old_values", None)

        loss_kwargs = CriticLossKwargs(
            shifted_labels=shifted_labels,
            returns=returns,
            old_values=old_values,
        ).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        return CriticLossContext(self, loss_kwargs)


class CriticLossKwargs(BaseLossKwargs):
    """Keyword arguments for critic loss computation.

    Args:
        shifted_labels (torch.Tensor): The shifted labels for the input sequences.
        returns (torch.Tensor): Per-token regression targets (GAE return or SFT target).
        old_values (torch.Tensor | None): Frozen values used for clipping. If None,
            the loss is unclipped MSE.
        loss_weight (torch.Tensor | None): Weights for each token in the loss.
    """

    shifted_labels: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor | None = None
    loss_weight: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> "CriticLossKwargs":
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        self.returns = sp_split(self.returns, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        if self.old_values is not None:
            self.old_values = sp_split(self.old_values, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        return self

    def to(self, device: torch.device | str) -> "CriticLossKwargs":
        self.shifted_labels = self.shifted_labels.to(device)
        self.returns = self.returns.to(device)
        if self.old_values is not None:
            self.old_values = self.old_values.to(device)
        return self


class CriticLossContext(BaseLossContext):
    """Critic loss context for PPO value-head training.

    Args:
        loss_cfg (CriticLossConfig): Configuration for critic loss computation.
        loss_kwargs (CriticLossKwargs): Keyword arguments required for loss calculation.
    """

    loss_cfg: CriticLossConfig
    loss_kwargs: CriticLossKwargs

    def __init__(self, loss_cfg: CriticLossConfig, loss_kwargs: CriticLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)

    @staticmethod
    def build_batches(loss_ctx_list: list["CriticLossContext"]) -> list["CriticLossContext"]:  # type: ignore[override]
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"

        loss_cfg = loss_ctx_list[0].loss_cfg
        loss_weight_list: list[torch.Tensor] = []
        for loss_ctx in loss_ctx_list:
            shifted_labels = loss_ctx.loss_kwargs.shifted_labels
            loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32)
            loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            loss_ctx.loss_kwargs.loss_weight = loss_weight
            loss_weight_list.append(loss_weight)

        rank_denominator = sum(loss_weight.sum() for loss_weight in loss_weight_list)
        rank_denominator = cast(torch.Tensor, rank_denominator)
        global_denominator = rank_denominator
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)

        if global_denominator == 0:
            global_denominator.add_(1)

        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
            assert loss_ctx.loss_kwargs.loss_weight is not None
            loss_ctx.loss_kwargs.loss_weight = loss_ctx.loss_kwargs.loss_weight / global_denominator
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CriticLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        values = F.linear(hidden_states, head_weight, head_bias).float()
        values_squeezed = values.squeeze(-1)

        shifted_labels = loss_kwargs.shifted_labels
        returns = loss_kwargs.returns
        old_values = loss_kwargs.old_values
        loss_weight = loss_kwargs.loss_weight
        assert loss_weight is not None, "loss_weight can not be None"

        returns = returns.squeeze(-1)
        loss_weight = loss_weight.squeeze(-1)
        if old_values is not None:
            old_values = old_values.squeeze(-1)

        rank_grad_tokens = (shifted_labels != self.loss_cfg.ignore_idx).sum()
        if rank_grad_tokens == 0:
            loss = values.sum() * 0
        elif old_values is None:
            per_token = 0.5 * (values_squeezed - returns) ** 2
            loss = (per_token * loss_weight).sum()
        else:
            cliprange = self.loss_cfg.cliprange_value
            v_clip = old_values + (values_squeezed - old_values).clamp(-cliprange, cliprange)
            per_token = 0.5 * torch.maximum((values_squeezed - returns) ** 2, (v_clip - returns) ** 2)
            loss = (per_token * loss_weight).sum()

        return loss, (values, {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        loss, (values, extra_info) = self.loss_fn(hidden_states, head_weight, head_bias, self.loss_kwargs)
        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return loss, (values, extra_info)
