# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import model_validator
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

    Only mode="eager" is supported: the value head output is 1-dim and
    needs no chunking. Setting mode="chunk" raises a validation error.

    Args:
        cliprange_value (float): Clip range applied to the difference between
            the current value and old_values. Defaults to 0.2.
        ignore_idx (int): Token index excluded from the loss. Defaults to -100.
    """

    cliprange_value: float = 0.2

    @model_validator(mode="after")
    def _reject_chunk_mode(self):
        if self.mode != "eager":
            raise ValueError(
                "CriticLossConfig only supports mode='eager'; the value head output is 1-dim and needs no chunking"
            )
        return self

    @property
    def loss_ctx_cls(self) -> type["CriticLossContext"]:
        return CriticLossContext

    @property
    def _loss_kwargs_cls(self) -> type["CriticLossKwargs"]:
        return CriticLossKwargs

    def finalize_metrics(self, extra_info_dict: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
        """Finalize value-clip metrics emitted by
        :meth:`CriticLossContext.loss_fn`.

        Args:
            extra_info_dict (dict[str, Any]): Per-rank metric dict after micro-batch reduction.
            device (str | torch.device): Device used for the all-reduce tensor.

        Returns:
            dict[str, Any]: The updated metric dict with global clip/valid counts
            and the derived ``reduced_critic_value_clip_frac``.
        """
        return finalize_critic_value_metrics(extra_info_dict, device)

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
        device: str | torch.device | None = None,
    ) -> "CriticLossContext | None":
        """Build critic loss context from data dict.

        Args:
            data (dict): Data dictionary containing critic-specific fields:
                - shifted_labels (torch.Tensor): The shifted labels
                - returns (torch.Tensor): Per-token regression targets
                - old_values (torch.Tensor | None): Frozen values for clipping
                  (optional, can be omitted for SFT MSE)
            sp_mesh (DeviceMesh | None): Sequence parallel device mesh
            device (str | torch.device): Target device for shifted_labels / returns /
                old_values. Matches the RL loss `build` protocol, where worker.py packs
                contexts on CPU (`device="cpu"`) before moving them to the compute device.

        Returns:
            CriticLossContext | None: The built loss context, or None if required
            fields are missing.
        """
        if "shifted_labels" not in data or "returns" not in data:
            return None
        if device is None:
            device = DEVICE

        shifted_labels = data["shifted_labels"]
        returns = data["returns"]
        old_values = data.get("old_values", None)
        # Only the trailing value channel may be squeezed; [B, S] targets and S==1
        # must never collapse the sequence dimension.
        if returns.ndim == 3 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if old_values is not None and old_values.ndim == 3 and old_values.shape[-1] == 1:
            old_values = old_values.squeeze(-1)
        if returns.shape != shifted_labels.shape or (
            old_values is not None and old_values.shape != shifted_labels.shape
        ):
            raise ValueError(
                "Critic targets must match shifted_labels shape "
                f"{tuple(shifted_labels.shape)}, got returns {tuple(returns.shape)}"
            )
        # Full-sample supervision weight; sp_split pads it with 0 so the global
        # denominator never counts padding.
        loss_weight = shifted_labels.ne(self.ignore_idx).to(torch.float32)
        loss_kwargs = CriticLossKwargs(
            shifted_labels=shifted_labels,
            returns=returns,
            old_values=old_values,
            loss_weight=loss_weight,
        )
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return CriticLossContext(self, loss_kwargs.to(device))


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
        if self.loss_weight is not None:
            self.loss_weight = sp_split(self.loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        return self

    def to(self, device: torch.device | str) -> "CriticLossKwargs":
        self.shifted_labels = self.shifted_labels.to(device)
        self.returns = self.returns.to(device)
        if self.old_values is not None:
            self.old_values = self.old_values.to(device)
        if self.loss_weight is not None:
            self.loss_weight = self.loss_weight.to(device)
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
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["CriticLossContext"],
        cu_seq_lens_list: Sequence[torch.IntTensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list["CriticLossContext"]:
        """Calibrate critic contexts with their full-sample supervision
        weights.

        Args:
            loss_ctx_list (list[CriticLossContext]): Contexts of one optimizer step.
            cu_seq_lens_list (Sequence[torch.IntTensor] | None): Unused; kept for
                the shared build protocol.
            sp_mesh (DeviceMesh | None): Unused; kept for the shared build protocol.

        Returns:
            list[CriticLossContext]: The same contexts with calibrated weights.
        """
        del cu_seq_lens_list, sp_mesh
        assert len(loss_ctx_list) > 0, "loss_ctx_list can not be empty"

        loss_weight_list: list[torch.Tensor] = []
        for loss_ctx in loss_ctx_list:
            if loss_ctx.loss_kwargs.loss_weight is None:
                loss_ctx.loss_kwargs.loss_weight = loss_ctx.loss_kwargs.shifted_labels.ne(
                    loss_ctx.loss_cfg.ignore_idx
                ).to(torch.float32)
            loss_weight_list.append(cast(torch.Tensor, loss_ctx.loss_kwargs.loss_weight))

        rank_denominator = sum(weight.sum() for weight in loss_weight_list)
        rank_denominator = cast(torch.Tensor, rank_denominator)
        global_denominator = rank_denominator
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        if global_denominator == 0:
            global_denominator = global_denominator + 1

        for loss_ctx in loss_ctx_list:
            loss_ctx._batch_size = len(loss_ctx_list)
            assert loss_ctx.loss_kwargs.loss_weight is not None
            loss_ctx.loss_kwargs.loss_weight /= global_denominator
        return loss_ctx_list

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CriticLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        values = F.linear(hidden_states, head_weight, head_bias).float()
        # Only the head output dim (always 1) may be squeezed; supervision fields are
        # aligned via reshape_as so S==1 cannot collapse the sequence dim.
        v = values.squeeze(-1)  # [B, S]

        shifted_labels = loss_kwargs.shifted_labels
        returns = loss_kwargs.returns.reshape_as(v)
        assert loss_kwargs.loss_weight is not None, "loss_weight can not be None"
        loss_weight = loss_kwargs.loss_weight.reshape_as(v)
        old_values = loss_kwargs.old_values
        if old_values is not None:
            old_values = old_values.reshape_as(v)

        valid_mask = shifted_labels != self.loss_cfg.ignore_idx
        rank_grad_tokens = valid_mask.sum()
        extra_info: dict[str, Any] = {}
        if old_values is None:
            if rank_grad_tokens == 0:
                loss = values.sum() * 0
            else:
                per_token = 0.5 * (v - returns) ** 2
                loss = (per_token * loss_weight).sum()
        else:
            cliprange = self.loss_cfg.cliprange_value
            v_clip = old_values + (v - old_values).clamp(-cliprange, cliprange)
            per_token = 0.5 * torch.maximum((v - returns) ** 2, (v_clip - returns) ** 2)
            loss = (per_token * loss_weight).sum()
            # Emit the counters on every rank whenever old_values is present, including
            # fully-masked ranks whose counts are zero: a missing key would make
            # ``finalize_critic_value_metrics`` skip its all-reduce rank-conditionally
            # and deadlock NCCL against ranks that do reduce (the pack builder avoids
            # the same class of bug by zero-filling old_values for missing packs).
            # Match the clip condition used by ``clamp`` exactly: equality keeps the
            # unclipped value, so only a strict deviation beyond the range counts.
            clipped = (v - old_values).abs() > cliprange
            extra_info = {
                "reduced_critic_value_clip_count": (clipped & valid_mask).float().sum(),
                "reduced_critic_value_valid_count": valid_mask.float().sum(),
            }

        return loss, (values, extra_info)

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


def finalize_critic_value_metrics(extra_info_dict: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """All-reduce critic value-clip counters and derive the clip fraction.

    Consumes the per-rank sums emitted by ``CriticLossContext.loss_fn`` after
    micro-batch reduction, all-reduces them into global counts, and derives
    ``reduced_critic_value_clip_frac``. Dictionaries without the critic keys
    (e.g. the unclipped SFT-MSE path) pass through unchanged.

    The all-reduce below must stay rank-symmetric: within one fit round either
    every rank carries the keys (``loss_fn`` emits them whenever ``old_values``
    is present, zero-filled on fully-masked ranks) or none does. A rank that
    skips it while peers reduce deadlocks NCCL until the watchdog kills the
    worker.

    Args:
        extra_info_dict (dict[str, Any]): Metric dict after micro-batch reduction.
        device (str | torch.device): Device used for the all-reduce tensor.

    Returns:
        dict[str, Any]: The updated metric dict with global counts and the clip fraction.
    """
    if "reduced_critic_value_valid_count" not in extra_info_dict:
        return extra_info_dict

    sum_keys = ["reduced_critic_value_clip_count", "reduced_critic_value_valid_count"]
    values = torch.tensor([extra_info_dict.pop(key, 0.0) for key in sum_keys], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    clip_count, valid_count = values.tolist()
    extra_info_dict["reduced_critic_value_clip_count"] = clip_count
    extra_info_dict["reduced_critic_value_valid_count"] = valid_count
    extra_info_dict["reduced_critic_value_clip_frac"] = clip_count / valid_count if valid_count > 0 else 0.0
    return extra_info_dict
