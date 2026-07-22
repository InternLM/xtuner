# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from xtuner.v1.loss.utils import sp_split
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.device import get_device

from ..utils import gather_logprobs
from .base_loss import (
    BaseRLLossConfig,
    BaseRLLossContext,
    BaseRLLossKwargs,
    compute_kl_loss_weight,
)
from .loss_fn import get_policy_loss_fn, kl_penalty


DEVICE = get_device()
logger = get_logger()

TopKDistillationMode = Literal["forward", "reverse", "forward_kl_topk"]
DistillationLossMode = Literal["k1", "k3", "forward", "reverse", "forward_kl_topk"]


def compute_topk_distillation_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mode: TopKDistillationMode,
    log_prob_min_clamp: float | None = None,
    loss_max_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a KL loss over teacher-selected Top-K tokens."""
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    student_selected_mass = student_probs.sum(dim=-1)
    teacher_selected_mass = teacher_probs.sum(dim=-1)

    if loss_mode == "forward_kl_topk":
        if log_prob_min_clamp is not None:
            student_logprobs = student_logprobs.clamp_min(log_prob_min_clamp)
            teacher_logprobs = teacher_logprobs.clamp_min(log_prob_min_clamp)
        teacher_probs = teacher_logprobs.exp()
        loss = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
        if loss_max_clamp is not None:
            loss = loss.clamp(min=-loss_max_clamp, max=loss_max_clamp)
        return loss, student_selected_mass, teacher_selected_mass

    eps = torch.finfo(student_logprobs.dtype).tiny
    student_tail = (1.0 - student_selected_mass).clamp_min(eps)
    teacher_tail = (1.0 - teacher_selected_mass).clamp_min(eps)
    if loss_mode == "forward":
        selected_kl = teacher_probs * (teacher_logprobs - student_logprobs)
        tail_kl = teacher_tail * (teacher_tail.log() - student_tail.log())
    else:
        selected_kl = student_probs * (student_logprobs - teacher_logprobs)
        tail_kl = student_tail * (student_tail.log() - teacher_tail.log())
    loss = selected_kl.sum(dim=-1) + tail_kl
    if loss_max_clamp is not None:
        loss = loss.clamp(max=loss_max_clamp)
    return loss, student_selected_mass, teacher_selected_mass


class DistillationLossConfig(BaseRLLossConfig):
    """Configuration shared by sampled-token PG-OPD and direct GKD losses."""

    loss_mode: DistillationLossMode = "k1"
    use_policy_gradient: bool = True
    task_adv_weight: float = Field(default=0.0, ge=0.0)
    distillation_loss_weight: float = Field(default=1.0, ge=0.0)
    top_k: int | None = Field(default=None, gt=0)
    log_prob_min_clamp: float | None = None
    loss_max_clamp: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_loss_options(self) -> Self:
        if self.loss_mode in ("forward", "reverse", "forward_kl_topk"):
            if self.use_policy_gradient:
                raise ValueError(f"{self.loss_mode} only supports direct backpropagation")
            if self.top_k is None:
                raise ValueError(f"{self.loss_mode} requires top_k")
        elif self.top_k is not None:
            raise ValueError(f"top_k is not used by loss_mode={self.loss_mode}")

        if self.loss_mode == "k1" and not self.use_policy_gradient:
            raise ValueError("k1 only supports the policy-gradient path")
        return self

    @property
    def uses_sampled_token_targets(self) -> bool:
        return self.loss_mode in ("k1", "k3")

    @property
    def uses_topk_targets(self) -> bool:
        return not self.uses_sampled_token_targets

    @property
    def loss_ctx_cls(self) -> type["DistillationLossContext"]:
        return DistillationLossContext

    @property
    def _loss_kwargs_cls(self) -> type["DistillationLossKwargs"]:
        return DistillationLossKwargs

    def build(
        self,
        data: dict,
        sp_mesh: DeviceMesh | None = None,
    ) -> "DistillationLossContext | None":
        if "shifted_labels" not in data or "advantages" not in data:
            return None

        loss_kwargs = DistillationLossKwargs(
            shifted_labels=data["shifted_labels"],
            advantages=data["advantages"],
            rollout_logprobs=data.get("rollout_logprobs"),
            old_logprobs=data.get("old_logprobs"),
            ref_logprobs=data.get("ref_logprobs"),
            is_weights=data.get("rollout_is_weights"),
            teacher_logprobs=data.get("teacher_logprobs"),
            target_token_ids=data.get("target_token_ids"),
        ).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)
        return self.loss_ctx_cls(self, loss_kwargs)


class DistillationLossKwargs(BaseRLLossKwargs):
    teacher_logprobs: torch.Tensor | None = None
    target_token_ids: torch.Tensor | None = None
    distillation_loss_weight: torch.Tensor | None = None

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        super().sp_split(sp_mesh)
        if self.teacher_logprobs is not None:
            self.teacher_logprobs = sp_split(
                self.teacher_logprobs,
                sp_mesh=sp_mesh,
                split_dim=1,
                padding_value=0.0,
            )
        if self.target_token_ids is not None:
            self.target_token_ids = sp_split(
                self.target_token_ids,
                sp_mesh=sp_mesh,
                split_dim=1,
                padding_value=0,
            )
        return self

    def to(self, device: torch.device | str) -> Self:
        super().to(device)
        if self.teacher_logprobs is not None:
            self.teacher_logprobs = self.teacher_logprobs.to(device)
        if self.target_token_ids is not None:
            self.target_token_ids = self.target_token_ids.to(device)
        if self.distillation_loss_weight is not None:
            self.distillation_loss_weight = self.distillation_loss_weight.to(device)
        return self


class DistillationLossContext(BaseRLLossContext):
    loss_cfg: DistillationLossConfig
    loss_kwargs: DistillationLossKwargs

    def __init__(self, loss_cfg: DistillationLossConfig, loss_kwargs: DistillationLossKwargs):
        super().__init__(loss_cfg, loss_kwargs)
        self.policy_loss_fn = get_policy_loss_fn(self.loss_cfg.policy_loss_cfg.get("loss_type", "vanilla"))

    @staticmethod
    def build_batches(  # type: ignore[override]
        loss_ctx_list: list["DistillationLossContext"],
    ) -> list["DistillationLossContext"]:
        assert loss_ctx_list, "loss_ctx_list can not be empty"
        loss_cfg = loss_ctx_list[0].loss_cfg
        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        rank_grad_tokens = sum((labels != loss_cfg.ignore_idx).sum() for labels in shifted_labels_list)
        global_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        if dist.is_initialized():
            dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)
        if global_grad_tokens == 0:
            logger.warning("Global gradient tokens is 0; using one as the loss denominator")
            global_grad_tokens.add_(1)

        for loss_ctx in loss_ctx_list:
            loss_kwargs = loss_ctx.loss_kwargs
            shifted_labels = loss_kwargs.shifted_labels
            assert loss_kwargs.old_logprobs is not None, "old_logprobs can not be None"

            policy_loss_weight = torch.ones_like(shifted_labels, dtype=torch.float32) / global_grad_tokens
            policy_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0
            if loss_kwargs.is_weights is not None:
                policy_loss_weight = policy_loss_weight * loss_kwargs.is_weights

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

            distillation_loss_weight = None
            if not loss_cfg.use_policy_gradient:
                distillation_loss_weight = (
                    torch.ones_like(shifted_labels, dtype=torch.float32)
                    / global_grad_tokens
                    * loss_cfg.distillation_loss_weight
                )
                distillation_loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0

            loss_kwargs.policy_loss_weight = policy_loss_weight
            loss_kwargs.kl_loss_weight = kl_loss_weight
            loss_kwargs.distillation_loss_weight = distillation_loss_weight
            loss_kwargs.global_grad_tokens = global_grad_tokens
        return loss_ctx_list

    def _compute_sampled_distillation_loss(
        self,
        student_logprobs: torch.Tensor,
        teacher_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_cfg.loss_mode == "k1":
            loss = student_logprobs - teacher_logprobs
        else:
            log_ratio = (teacher_logprobs - student_logprobs).clamp(min=-20.0, max=20.0)
            loss = torch.exp(log_ratio) - log_ratio - 1.0
        if self.loss_cfg.loss_max_clamp is not None:
            loss = loss.clamp(
                min=-self.loss_cfg.loss_max_clamp,
                max=self.loss_cfg.loss_max_clamp,
            )
        return loss

    def _compute_topk_distillation_loss(
        self,
        logits: torch.Tensor,
        loss_kwargs: DistillationLossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_token_ids = cast(torch.Tensor, loss_kwargs.target_token_ids)
        teacher_logprobs = cast(torch.Tensor, loss_kwargs.teacher_logprobs)
        student_topk_logprobs = torch.gather(logits, dim=-1, index=target_token_ids)
        student_topk_logprobs = student_topk_logprobs - torch.logsumexp(logits, dim=-1, keepdim=True)

        loss_mode = cast(TopKDistillationMode, self.loss_cfg.loss_mode)

        loss, student_mass, teacher_mass = compute_topk_distillation_kl(
            student_topk_logprobs,
            teacher_logprobs,
            loss_mode,
            self.loss_cfg.log_prob_min_clamp,
            self.loss_cfg.loss_max_clamp,
        )
        student_topk_ids = torch.topk(logits.detach(), k=target_token_ids.size(-1), dim=-1).indices
        overlap = (student_topk_ids.unsqueeze(-1) == target_token_ids.detach().unsqueeze(-2)).any(dim=-1)
        return loss, student_mass, teacher_mass, overlap.float().mean(dim=-1)

    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: DistillationLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        logits = F.linear(hidden_states, head_weight, head_bias).float()
        shifted_labels = loss_kwargs.shifted_labels
        old_logprobs = cast(torch.Tensor, loss_kwargs.old_logprobs)
        policy_loss_weight = cast(torch.Tensor, loss_kwargs.policy_loss_weight)
        current_logprobs = gather_logprobs(logits, shifted_labels)

        topk_metrics: dict[str, torch.Tensor] = {}
        if self.loss_cfg.uses_sampled_token_targets:
            teacher_logprobs = cast(torch.Tensor, loss_kwargs.teacher_logprobs)
            distillation_student_logprobs = old_logprobs if self.loss_cfg.use_policy_gradient else current_logprobs
            per_token_distillation_loss = self._compute_sampled_distillation_loss(
                distillation_student_logprobs,
                teacher_logprobs,
            )
        else:
            (
                per_token_distillation_loss,
                student_selected_mass,
                teacher_selected_mass,
                overlap_fraction,
            ) = self._compute_topk_distillation_loss(logits, loss_kwargs)
            topk_metrics = {
                "reduced_topk_opd_student_selected_mass_sum": student_selected_mass.detach(),
                "reduced_topk_opd_teacher_selected_mass_sum": teacher_selected_mass.detach(),
                "reduced_topk_opd_overlap_fraction_sum": overlap_fraction.detach(),
            }

        if self.loss_cfg.use_policy_gradient:
            combined_advantages = (
                self.loss_cfg.task_adv_weight * loss_kwargs.advantages
                - self.loss_cfg.distillation_loss_weight * per_token_distillation_loss.detach()
            )
            loss = self.policy_loss_fn(
                current_logprobs,
                old_logprobs,
                combined_advantages,
                policy_loss_weight,
                self.loss_cfg.policy_loss_cfg,
            )
        else:
            distillation_loss_weight = cast(torch.Tensor, loss_kwargs.distillation_loss_weight)
            task_advantages = self.loss_cfg.task_adv_weight * loss_kwargs.advantages
            task_loss = self.policy_loss_fn(
                current_logprobs,
                old_logprobs,
                task_advantages,
                policy_loss_weight,
                self.loss_cfg.policy_loss_cfg,
            )
            distillation_loss = (per_token_distillation_loss * distillation_loss_weight).sum()
            loss = task_loss + distillation_loss
            if self.loss_cfg.uses_topk_targets:
                topk_metrics["reduced_topk_opd_loss_sum"] = distillation_loss.detach()

        valid_mask = shifted_labels != self.loss_cfg.ignore_idx
        valid_float = valid_mask.float()
        log_ratio = current_logprobs.detach() - old_logprobs.detach()
        log_ratio_safe = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio_safe)
        ratio_max = ratio.masked_fill(~valid_mask, 0.0).max()
        ratio_min = ratio.masked_fill(~valid_mask, float("inf")).min()
        extra_info = {
            "max_ratio": ratio_max,
            "reduced_train_policy_ratio_abs_dev_sum": ((ratio - 1.0).abs() * valid_float).sum(),
            "reduced_train_policy_kl1_sum": (-log_ratio * valid_float).sum(),
            "reduced_train_policy_kl3_sum": ((ratio - 1.0 - log_ratio_safe) * valid_float).sum(),
            "reduced_train_policy_valid_count": valid_float.sum(),
            "reduced_train_policy_ratio_max": ratio_max,
            "reduced_train_policy_ratio_min": ratio_min,
            "reduced_distillation_kl_sum": (per_token_distillation_loss.detach() * valid_float).sum(),
            "reduced_distillation_abs_loss_sum": (per_token_distillation_loss.detach().abs() * valid_float).sum(),
            "reduced_distillation_valid_count": valid_float.sum(),
            **{
                key: (value * valid_float).sum() if key != "reduced_topk_opd_loss_sum" else value
                for key, value in topk_metrics.items()
            },
        }
        if self.loss_cfg.loss_mode == "k1":
            extra_info["reduced_opd_reverse_kl_sum"] = (per_token_distillation_loss.detach() * valid_float).sum()
            extra_info["reduced_opd_abs_logprob_loss_sum"] = (
                per_token_distillation_loss.detach().abs() * valid_float
            ).sum()
        if self.loss_cfg.uses_topk_targets:
            extra_info["reduced_topk_opd_kl_sum"] = (per_token_distillation_loss.detach() * valid_float).sum()
            extra_info["reduced_topk_opd_valid_count"] = valid_float.sum()

        cliprange_low = self.loss_cfg.policy_loss_cfg.get("cliprange_low")
        cliprange_high = self.loss_cfg.policy_loss_cfg.get("cliprange_high")
        if cliprange_low is not None and cliprange_high is not None:
            extra_info["reduced_train_policy_clip_low_count"] = (
                ((ratio < 1 - cliprange_low) & valid_mask).float().sum()
            )
            extra_info["reduced_train_policy_clip_high_count"] = (
                ((ratio > 1 + cliprange_high) & valid_mask).float().sum()
            )

        if self.loss_cfg.use_kl_loss:
            ref_logprobs = loss_kwargs.ref_logprobs
            kl_loss_weight = loss_kwargs.kl_loss_weight
            assert ref_logprobs is not None and kl_loss_weight is not None
            loss = loss + kl_penalty(
                current_logprobs,
                ref_logprobs,
                kl_loss_weight,
                self.loss_cfg.kl_loss_type,
            )

        return loss, (logits, extra_info)


def finalize_distillation_metrics(
    extra_info_dict: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    def reduce_values(keys: tuple[str, ...]) -> dict[str, float]:
        values = torch.tensor(
            [extra_info_dict.pop(key, 0.0) for key in keys],
            dtype=torch.float32,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return dict(zip(keys, values.tolist()))

    if "reduced_distillation_valid_count" in extra_info_dict:
        distillation_keys = [
            "reduced_distillation_kl_sum",
            "reduced_distillation_abs_loss_sum",
            "reduced_distillation_valid_count",
        ]
        has_opd_metrics = "reduced_opd_reverse_kl_sum" in extra_info_dict
        if has_opd_metrics:
            distillation_keys.extend(
                [
                    "reduced_opd_reverse_kl_sum",
                    "reduced_opd_abs_logprob_loss_sum",
                ]
            )
        values = reduce_values(tuple(distillation_keys))
        valid_count = values["reduced_distillation_valid_count"]
        extra_info_dict["reduced_distillation_kl"] = (
            values["reduced_distillation_kl_sum"] / valid_count if valid_count > 0 else 0.0
        )
        extra_info_dict["reduced_distillation_abs_loss"] = (
            values["reduced_distillation_abs_loss_sum"] / valid_count if valid_count > 0 else 0.0
        )
        if has_opd_metrics:
            extra_info_dict["opd_reverse_kl"] = (
                values["reduced_opd_reverse_kl_sum"] / valid_count if valid_count > 0 else 0.0
            )
            extra_info_dict["opd_abs_logprob_loss"] = (
                values["reduced_opd_abs_logprob_loss_sum"] / valid_count if valid_count > 0 else 0.0
            )

    if "reduced_topk_opd_valid_count" in extra_info_dict:
        topk_keys = (
            "reduced_topk_opd_kl_sum",
            "reduced_topk_opd_loss_sum",
            "reduced_topk_opd_student_selected_mass_sum",
            "reduced_topk_opd_teacher_selected_mass_sum",
            "reduced_topk_opd_overlap_fraction_sum",
            "reduced_topk_opd_valid_count",
        )
        values = reduce_values(topk_keys)
        valid_count = values["reduced_topk_opd_valid_count"]
        for output_key, sum_key in {
            "reduced_topk_opd_kl": "reduced_topk_opd_kl_sum",
            "reduced_topk_opd_student_selected_mass": "reduced_topk_opd_student_selected_mass_sum",
            "reduced_topk_opd_teacher_selected_mass": "reduced_topk_opd_teacher_selected_mass_sum",
            "reduced_topk_opd_overlap_fraction": "reduced_topk_opd_overlap_fraction_sum",
        }.items():
            extra_info_dict[output_key] = values[sum_key] / valid_count if valid_count > 0 else 0.0
        extra_info_dict["reduced_topk_opd_loss"] = values["reduced_topk_opd_loss_sum"]
    return extra_info_dict
