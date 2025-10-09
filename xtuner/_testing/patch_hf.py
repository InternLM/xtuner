from xtuner.v1.module import RMSNorm
import torch.nn as nn


def patch_hf_rms_norm(module: nn.Module) -> None:
    for submodule in module.modules():
        if "RMSNorm" in submodule.__class__.__name__ and isinstance(submodule, nn.Module):
            submodule.__class__.forward = RMSNorm.forward

