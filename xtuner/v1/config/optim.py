import math
from abc import abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch.optim.lr_scheduler import LambdaLR
from typing_extensions import Annotated

from xtuner.v1.optim import Muon, SwapAdamW
from xtuner.v1.utils import get_logger


logger = get_logger()


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0
    skip_grad_norm_threshold: Annotated[
        float | None, Parameter(help="Gradient norm threshold for skipping optimizer step.")
    ] = None

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8
    foreach: Annotated[Optional[bool], Parameter(help="Use foreach implementation for AdamW")] = None
    swap_optimizer: Annotated[Optional[bool], Parameter(help="Swap optimizer states to host memory.")] = False

    def build(self, model):
        params = [p for p in model.parameters() if p.requires_grad]

        trainable_parameters_names = model.trainable_parameters()
        trainable_names = [name for name, _ in trainable_parameters_names]
        untrainable_names = []
        num_total_requires_grad = 0
        num_total = 0
        for name, params_ in model.named_parameters():
            num_total += params_.numel()
            num_total_requires_grad += params_.numel() if name in trainable_names else 0
            if name not in trainable_names:
                untrainable_names.append(name)

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad / 1e6:.2f}M, total parameters: {num_total / 1e6:.2f}M"
            )
            logger.info(f"Untrainable parameters names: {untrainable_names}")
        if self.swap_optimizer:
            return SwapAdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                foreach=self.foreach,
            )
        return torch.optim.AdamW(
            params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, foreach=self.foreach
        )


class MuonConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.1
    momentum: Annotated[float, Parameter(help="Momentum coefficients for Muon optimizer")] = 0.95
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for AdamW optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Muon optimizer")] = 1e-8
    adjust_lr: Annotated[
        Literal["rms_norm", "spectral_norm", "none"], Parameter(help="Method for adjusting lr in Muon")
    ] = "rms_norm"
    enable_all2all: Annotated[
        bool, Parameter(help="Allow all-to-all comm strategy; set False on topologies where all-to-all is unreliable")
    ] = True
    clip_grad_mode: Annotated[
        Literal["all", "adamw_only"],
        Parameter(help="Gradient clipping policy: clip all parameters or only the AdamW parameter groups"),
    ] = "adamw_only"

    def build(self, model):
        trainable_parameters_names = model.trainable_parameters()
        trainable_names = {name for name, _ in trainable_parameters_names}

        muon_split_sizes = {}
        for module in model.modules():
            get_muon_split_sizes = getattr(module, "get_muon_split_sizes", None)
            if callable(get_muon_split_sizes):
                muon_split_sizes.update(get_muon_split_sizes())

        untrainable_names = []
        num_total = 0
        num_total_requires_grad = 0
        num_muon_regular = 0
        num_muon_moe = 0
        num_adamw = 0

        num_experts: int
        if hasattr(model.config, "text_config"):
            num_experts = getattr(model.config.text_config, "n_routed_experts", 1) or 1
        else:
            num_experts = getattr(model.config, "n_routed_experts", 1) or 1

        is_moe_model = num_experts > 1

        # Expert parameter patterns for MoE models
        # Note: fused_w1w3 contains both w1 and w3 weights, so num_experts = 2 * n_routed_experts
        fused_w1w3_patterns = ("fused_w1w3",)
        other_expert_patterns = ("fused_w2", "fused_w1", "fused_w3")

        # Separate Muon params into regular and MoE expert params
        # fused_w1w3 has 2 * num_experts (w1 and w3 each have num_experts)
        # other expert params have num_experts
        muon_params_regular = []
        muon_params_moe_fused_w1w3 = []  # num_experts = 2 * n_routed_experts
        muon_params_moe_other = []  # num_experts = n_routed_experts
        adamw_params = []

        for name, p in model.named_parameters():
            n = p.numel()
            num_total += n
            if name in trainable_names:
                num_total_requires_grad += n
                # we want to avoid using Muon for 1D-tensors, as well as embed_tokens and lm_head.
                # effectively-1D tensors where one dimension accounts for all elements (e.g. shape [1, D]) should
                # also be excluded, and hence the `p.numel() not in p.shape` condition.
                is_muon_tensor = (
                    p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and p.numel() not in p.shape
                )
                if is_muon_tensor:
                    if is_moe_model and any(pattern in name for pattern in fused_w1w3_patterns):
                        muon_params_moe_fused_w1w3.append(p)
                        num_muon_moe += n
                        if dist.get_rank() == 0:
                            logger.info(f"Muon moe param: [{name}]: shape = {p.shape}")
                    elif is_moe_model and any(pattern in name for pattern in other_expert_patterns):
                        muon_params_moe_other.append(p)
                        num_muon_moe += n
                        if dist.get_rank() == 0:
                            logger.info(f"Muon moe param: [{name}]: shape = {p.shape}")
                    else:
                        muon_params_regular.append(p)
                        num_muon_regular += n
                        if dist.get_rank() == 0:
                            logger.info(f"Muon regular param: [{name}]: shape = {p.shape}")
                else:
                    adamw_params.append(p)
                    num_adamw += n
                    if dist.get_rank() == 0:
                        logger.info(f"AdamW param: [{name}]: shape = {p.shape}")
            else:
                untrainable_names.append(name)

        # Build parameter groups
        clip_muon_grad = self.clip_grad_mode == "all"
        param_groups = []
        if muon_params_regular:
            param_groups.append(dict(params=muon_params_regular, clip_grad=clip_muon_grad))
        # fused_w1w3: w1 and w3 are fused, so num_experts = 2 * n_routed_experts
        if muon_params_moe_fused_w1w3:
            param_groups.append(
                dict(
                    params=muon_params_moe_fused_w1w3,
                    num_experts=2 * num_experts,
                    clip_grad=clip_muon_grad,
                )
            )
        # Other expert params: num_experts = n_routed_experts
        if muon_params_moe_other:
            param_groups.append(dict(params=muon_params_moe_other, num_experts=num_experts, clip_grad=clip_muon_grad))
        param_groups.append(dict(params=adamw_params, algorithm="adamw", clip_grad=True))

        # Sanity check: ensure all trainable parameters are assigned to optimizer
        total_assigned = (
            len(muon_params_regular) + len(muon_params_moe_fused_w1w3) + len(muon_params_moe_other) + len(adamw_params)
        )
        total_trainable = len(trainable_names)
        assert total_assigned == total_trainable, (
            f"Parameter assignment mismatch: {total_assigned} assigned vs {total_trainable} trainable. "
            f"Some parameters may have their gradients not zeroed by optimizer.zero_grad()."
        )

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad / 1e6:.2f}M, total parameters: {num_total / 1e6:.2f}M"
            )
            if is_moe_model:
                logger.info(
                    f"Muon params: {(num_muon_regular + num_muon_moe) / 1e6:.2f}M "
                    f"(regular: {num_muon_regular / 1e6:.2f}M, MoE expert: {num_muon_moe / 1e6:.2f}M), "
                    f"AdamW params: {num_adamw / 1e6:.2f}M (counts by numel)"
                )
                logger.info(
                    f"Detected MoE model with {num_experts} routed experts, "
                    f"fused_w1w3 uses num_experts={2 * num_experts} (w1+w3), "
                    f"other expert params use num_experts={num_experts}"
                )
            else:
                logger.info(
                    f"Muon params: {num_muon_regular / 1e6:.2f}M, AdamW params: {num_adamw / 1e6:.2f}M (counts by numel)"
                )
            logger.info(f"Untrainable parameters names: {untrainable_names}")

        optimizer = Muon(
            param_groups,
            lr=self.lr,
            mu=self.momentum,
            betas=self.betas,
            weight_decay=self.weight_decay,
            nesterov=True,
            flatten=True,
            adjust_lr=self.adjust_lr,
            use_triton=False,
            epsilon=self.eps,
            enable_all2all=self.enable_all2all,
            muon_split_sizes=muon_split_sizes,
        )

        return optimizer


class _ResumableLambdaLR(LambdaLR):
    """A ``LambdaLR`` that reapplies its learning rate when state is loaded.

    ``LRScheduler.load_state_dict`` only refreshes the scheduler's own counters;
    it never writes the restored learning rate back to the optimizer. The next
    ``optimizer.step()`` therefore runs at whatever rate the freshly constructed
    scheduler happened to set -- 0.0 while warmup is configured -- and only the
    following ``scheduler.step()`` corrects it. Reapplying on load keeps a
    resumed run continuous.
    """

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            param_group["lr"] = lr


class LRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr_type: Annotated[Literal["cosine", "linear", "constant"], Parameter(help="Type of learning rate schedule")] = (
        "constant"
    )
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Build the learning rate scheduler for ``optimizer``.

        Warmup is always applied first, then the configured decay. When
        ``warmup_ratio`` is below 1 it is a fraction of ``total_steps``,
        otherwise it is an absolute step count.

        Implemented as a single ``LambdaLR`` rather than a ``SequentialLR`` of
        warmup and decay phases: ``SequentialLR.load_state_dict`` restores its
        internal counters without pushing the learning rate back onto the
        optimizer, so a resumed run continues from the wrong point in the
        schedule. One stateless closure over ``last_epoch`` has no such issue.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule.
            total_steps (int): Total number of scheduler steps.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: Configured scheduler.
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        if self.warmup_ratio < 1:
            warmup_steps = int(self.warmup_ratio * total_steps)
        else:
            warmup_steps = int(self.warmup_ratio)
        warmup_steps = min(warmup_steps, total_steps)

        base_lr = float(optimizer.defaults["lr"])
        min_factor = self.lr_min / base_lr if base_lr > 0 else 1.0
        decay_steps = max(total_steps - warmup_steps, 1)
        lr_type = self.lr_type

        def lr_factor(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            if lr_type == "constant":
                return 1.0
            # Clamped so the schedule holds at lr_min past total_steps rather
            # than continuing past the end of the curve.
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            if lr_type == "linear":
                return 1.0 - progress * (1.0 - min_factor)
            if lr_type == "cosine":
                return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
            raise ValueError(f"Unsupported lr type: {lr_type}")

        if lr_type not in ("constant", "linear", "cosine"):
            raise ValueError(f"Unsupported lr type: {lr_type}")

        return _ResumableLambdaLR(optimizer, lr_factor)
