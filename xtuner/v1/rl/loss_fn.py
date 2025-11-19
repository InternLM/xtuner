from typing import Any, Callable

import torch
import torch.nn.functional as F


PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # loss_weights
        dict,  # config
    ],
    torch.Tensor,  # loss
]


POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        if "." in loss_name:  # try to manually import the loss function from a custom path
            try:
                import importlib

                package_name, module_name = loss_name.rsplit(".", 1)
                print(f"Importing loss function from {package_name}.{module_name}")
                module = importlib.import_module(package_name)
                loss_fn = getattr(module, module_name)
                POLICY_LOSS_REGISTRY[loss_name] = loss_fn
                return loss_fn
            except ImportError as e:
                raise ImportError(f"Failed to import loss function: {loss_name}, error: {e}")
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


def check_config(keys_needed: list[str], config: dict[str, Any]) -> None:
    """Check if the config contains all the required keys."""
    for key in keys_needed:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config. Available keys: {list(config.keys())}")


@register_policy_loss("vanilla")
def pg_loss_fn(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    loss_weights: torch.Tensor,
    policy_loss_cfg: dict,
) -> torch.Tensor:
    check_config(["cliprange_low", "cliprange_high"], policy_loss_cfg)
    cliprange_low = policy_loss_cfg["cliprange_low"]
    cliprange_high = policy_loss_cfg["cliprange_high"]
    clip_ratio_c = policy_loss_cfg.get("clip_ratio_c", 3.0)
    log_prob_diff_min = policy_loss_cfg.get("log_prob_diff_min", -20.0)
    log_prob_diff_max = policy_loss_cfg.get("log_prob_diff_max", 20.0)
    advantages = advantages.to(log_prob.dtype)
    negative_approx_kl = log_prob - old_log_prob.detach()
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=log_prob_diff_min, max=log_prob_diff_max)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -clip_ratio_c * advantages
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    loss = (pg_losses * loss_weights.to(pg_losses.dtype)).sum()
    return loss

@register_policy_loss("cispo")
def pg_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_weights: torch.Tensor,
    policy_loss_cfg: dict,
) -> torch.Tensor:
    check_config(["cliprange_low", "cliprange_high"], policy_loss_cfg)
    cliprange_low = policy_loss_cfg["cliprange_low"]
    cliprange_high = policy_loss_cfg["cliprange_high"]

    advantages = advantages.to(logprobs.dtype)
    negative_approx_kl = logprobs.detach() - old_logprobs.detach()
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high) * logprobs
    loss = (pg_losses * loss_weights.to(pg_losses.dtype)).sum()
    return loss

def sft_loss_fn(
    logits: torch.Tensor,  # [1, seq_len, vocab_size]
    shifted_labels: torch.Tensor,  # [1, seq_len]
    loss_weights: torch.Tensor,  # [1, seq_len]
    ignore_idx: int = -100,
) -> torch.Tensor:
    sft_loss = F.cross_entropy(
        logits.squeeze(),
        shifted_labels.squeeze(),
        reduction="none",
        ignore_index=ignore_idx,
    ).unsqueeze(0)  # [1, seq_len]
    sft_loss = (sft_loss * loss_weights.to(sft_loss.dtype)).sum()
    return sft_loss


def kl_penalty(
    logprobs: torch.Tensor, ref_logprobs: torch.Tensor, loss_weights: torch.Tensor, kl_penalty
) -> torch.Tensor:
    """
    Modified from https://github.com/volcengine/verl/blob/313366fd85e95ad43d567a808dd647089723a255/verl/trainer/ppo/core_algos.py#L1272
    Compute KL divergence given logprobs and ref_logprobs.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty in ("kl", "k1"):
        loss = logprobs - ref_logprobs
        # return logprobs - ref_logprobs
    elif kl_penalty == "abs":
        loss = (logprobs - ref_logprobs).abs()
    elif kl_penalty in ("mse", "k2"):
        loss = 0.5 * (logprobs - ref_logprobs).square()
    elif kl_penalty in ("low_var_kl", "k3"):
        # J. Schulman. Approximating kl divergence, 2020.
        # # URL http://joschu.net/blog/kl-approx.html.
        kl = ref_logprobs - logprobs
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        loss = torch.clamp(kld, min=-10, max=10)
    else:
        raise NotImplementedError

    return (loss * loss_weights.to(loss.dtype)).sum()
