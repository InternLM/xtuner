# Copyright (c) OpenMMLab. All rights reserved.
from typing import Annotated, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss import BaseLossConfig, BaseLossContext, BaseLossKwargs

from .utils import sp_gather, sp_split


class CELossConfig(BaseLossConfig):
    loss_reduction: Annotated[Literal["token", "sample", "square"], Parameter(help="loss reduction mode")] = "token"

    @property
    def loss_ctx_cls(self) -> type["CELossContext"]:
        return CELossContext


class CELossKwargs(BaseLossKwargs):
    shifted_labels: torch.Tensor
    loss_weight: torch.Tensor


class CELossContextInputItem(BaseModel):
    model_config = ConfigDict(title="CELossContextInputItem", extra="allow", arbitrary_types_allowed=True)
    shifted_labels: torch.Tensor

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        return type(self)(shifted_labels=shifted_labels)

    def to(self, device: torch.device | str) -> Self:
        self.shifted_labels = self.shifted_labels.to(device)
        return self


class CELossContext(BaseLossContext[CELossContextInputItem]):
    loss_cfg: CELossConfig
    loss_kwargs: CELossKwargs

    @classmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[CELossContextInputItem],
        loss_cfg: CELossConfig,
        # "sample" and "square" reduction need sp_mesh and cu_seq_lens_list
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list[CELossKwargs]:
        shifted_labels_list = [item.shifted_labels for item in data_batches]

        loss_weight_list: list[torch.Tensor] = []
        for i, shifted_labels in enumerate(shifted_labels_list):
            if loss_cfg.loss_reduction == "token":
                loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32)
            else:
                assert cu_seq_lens_list is not None, "cu_seq_lens_list must be provided for sample or square reduction"
                cu_seq_lens = cu_seq_lens_list[i]
                boundaries = cu_seq_lens[1:]
                num_tokens = cu_seq_lens[1:] - cu_seq_lens[:-1]

                if sp_mesh is not None:
                    # gather shifted_labels from different sp ranks to compute the correct loss weight
                    shifted_labels = sp_gather(shifted_labels, sp_mesh=sp_mesh, dim=1)

                mask = (shifted_labels != loss_cfg.ignore_idx).int()
                num_grad_tokens = torch.zeros_like(boundaries, dtype=torch.int32)
                prev_idx = 0
                for i, boundary in enumerate(boundaries):
                    num_grad_tokens[i] = mask[0, prev_idx:boundary].sum()
                    prev_idx = boundary
                if loss_cfg.loss_reduction == "sample":
                    loss_weight = 1.0 / num_grad_tokens
                elif loss_cfg.loss_reduction == "square":
                    loss_weight = 1.0 / torch.sqrt(num_grad_tokens.float())
                else:
                    raise NotImplementedError(loss_cfg.loss_reduction)
                loss_weight = loss_weight.repeat_interleave(num_tokens).unsqueeze(0)

                if sp_mesh is not None:
                    loss_weight = sp_split(loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)

            loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if torch.isnan(loss_weight).any() or torch.isinf(loss_weight).any():
                raise AssertionError(
                    "loss_weight contains NaN or Inf values. Please filter out samples with no valid tokens."
                )
            loss_weight_list.append(loss_weight)

        # Compute the denominator used in the global calibration of the loss
        rank_denominator = sum(loss_weight.sum() for loss_weight in loss_weight_list)
        rank_denominator = cast(torch.Tensor, rank_denominator)
        global_denominator = rank_denominator
        if dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)

        batches_loss_kwargs = []
        for i, item in enumerate(data_batches):
            shifted_labels = shifted_labels_list[i]
            loss_weight = loss_weight_list[i]
            loss_weight = loss_weight / (global_denominator + 1e-12)
            loss_kwargs = CELossKwargs(
                shifted_labels=shifted_labels,
                loss_weight=loss_weight,
            )
            batches_loss_kwargs.append(loss_kwargs)
        return batches_loss_kwargs

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: CELossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # We do linear forward here to simplify the implementation of chunk loss (saving memory).
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = logits.float()  # (bs, seq_len, vocab_size)

        shifted_labels = loss_kwargs.shifted_labels  # (bs, seq_len)
        loss_weight = loss_kwargs.loss_weight  # (bs, seq_len)

        logits = logits.reshape(-1, logits.size(-1))  # (bs * seq_len, vocab_size)
        shifted_labels = shifted_labels.flatten()
        loss_weight = loss_weight.flatten()

        rank_grad_tokens = (shifted_labels != self.loss_cfg.ignore_idx).sum()
        if rank_grad_tokens == 0:
            loss = logits.sum() * 0
        else:
            loss = F.cross_entropy(logits, shifted_labels, reduction="none", ignore_index=self.loss_cfg.ignore_idx)
            loss = (loss * loss_weight).sum()

        return loss, logits
