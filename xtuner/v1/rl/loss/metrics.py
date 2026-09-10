from typing import Any

import torch

from .base_loss import BaseRLLossConfig, finalize_train_policy_metrics
from .distillation_loss import DistillationLossConfig, finalize_distillation_metrics


def finalize_train_metrics(
    extra_info_dict: dict[str, Any],
    device: str | torch.device,
    loss_cfg: BaseRLLossConfig,
) -> dict[str, Any]:
    """Finalize policy metrics and optional distillation metrics."""
    extra_info_dict = finalize_train_policy_metrics(extra_info_dict, device)
    if isinstance(loss_cfg, DistillationLossConfig):
        extra_info_dict = finalize_distillation_metrics(extra_info_dict, device)
    return extra_info_dict
