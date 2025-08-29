# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xtuner.v1.utils import get_logger

from ..base_loss import BaseLoss, BaseLossKwargs

# from mmengine.dist import dist
from ..loss_context import ForwardItem
from ..loss_fn import get_policy_loss_fn, kl_penalty
from ..utils import gather_logprobs, sp_split


logger = get_logger()


class GRPOLossKwargs(BaseLossKwargs):
    pass


class GRPOLoss(BaseLoss):
    def __init__(
        self,
        policy_loss_cfg: dict,
        ignore_idx: int = -100,
        use_kl_loss: bool = False,
        kl_loss_coef: float = 0.001,
        kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None,
        mode: Literal["eager", "chunk"] = "eager",
        chunk_size: int | None = None,
    ):
        super().__init__(mode=mode, chunk_size=chunk_size)
        self.policy_loss_cfg = policy_loss_cfg
        self.ignore_idx = ignore_idx
        self.use_kl_loss = use_kl_loss
        self.kl_loss_coef = kl_loss_coef
        self.kl_loss_type = kl_loss_type

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: GRPOLossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Step 2.a and 2.b in the loss calculation in
        xtuner/v1/rl/base_loss.py."""
        # We do linear forward here to simplify the implementation of chunk loss (saving memory).
        logits = F.linear(hidden_states, head_weight, head_bias)
        logits = logits.float()

        shifted_labels = loss_kwargs.shifted_labels
        old_logprobs = loss_kwargs.old_logprobs
        advantages = loss_kwargs.advantages
        policy_loss_weight = loss_kwargs.policy_loss_weight

        logprobs = gather_logprobs(logits, shifted_labels)
        policy_loss_fn = get_policy_loss_fn(self.policy_loss_cfg.get("loss_type", "vanilla"))
        loss = policy_loss_fn(
            logprobs,
            old_logprobs,
            advantages,
            policy_loss_weight,
            self.policy_loss_cfg,
        )

        if self.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None, (
                "loss_kwargs.ref_logprobs and loss_kwargs.kl_loss_weight can not be None when use_kl_loss=True"
            )
            kl_loss = kl_penalty(logprobs, ref_logprobs, kl_loss_weight, self.kl_loss_type)
            loss = loss + kl_loss

        return loss, logits

    def build_loss_kwargs(self, forward_item: ForwardItem) -> GRPOLossKwargs:
        iter_idx = forward_item["iter_idx"]
        sp_mesh = forward_item["data_batch"][iter_idx]["seq_ctx"].sequence_parallel_mesh

        shifted_labels_list = [item["shifted_labels"] for item in forward_item["data_batch"]]
        shifted_labels = shifted_labels_list[iter_idx]
        advantages = forward_item["data_batch"][iter_idx]["advantages"]

        # Compute the denominator used in the global calibration of the loss
        rank_grad_tokens = sum((labels != self.ignore_idx).sum() for labels in shifted_labels_list)
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)
        if sp_mesh is not None:
            # data in different sp ranks are replicated
            global_grad_tokens = global_grad_tokens / sp_mesh.size()  # type: ignore

        # compute loss weight
        if global_grad_tokens == 0:
            logger.warning(
                "Global gradient tokens is 0, which may lead to division by zero in loss weight calculation."
            )
            global_grad_tokens.add_(1)  # Avoid division by zero
        policy_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens
        policy_loss_weight[shifted_labels == self.ignore_idx] = 0.0

        if self.use_kl_loss:
            ref_logprobs = forward_item["data_batch"][iter_idx]["ref_logprobs"]
            kl_loss_weight = policy_loss_weight.clone() * self.kl_loss_coef
        else:
            ref_logprobs = None
            kl_loss_weight = None

        if sp_mesh is not None:
            shifted_labels = sp_split(shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=self.ignore_idx)
            advantages = sp_split(advantages, sp_mesh=sp_mesh, split_dim=1, padding_value=0)
            policy_loss_weight = sp_split(policy_loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
            kl_loss_weight = (
                sp_split(kl_loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
                if kl_loss_weight is not None
                else None
            )

        loss_kwargs = GRPOLossKwargs(
            old_logprobs=forward_item["data_batch"][iter_idx]["old_logprobs"],
            shifted_labels=shifted_labels,
            advantages=advantages,
            policy_loss_weight=policy_loss_weight,
            ref_logprobs=ref_logprobs,
            kl_loss_weight=kl_loss_weight,
        )
        return loss_kwargs
