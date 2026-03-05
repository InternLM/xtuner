from xtuner.v1.module import RMSNorm
import torch.nn as nn


def patch_hf_rms_norm(module: nn.Module) -> None:
    replacements = []
    for name, submodule in module.named_modules():
        if "RMSNorm" in submodule.__class__.__name__ and isinstance(submodule, nn.Module):
            dim = submodule.weight.shape
            device = submodule.weight.device
            eps = submodule.variance_epsilon
            new_submodule = RMSNorm(hidden_size=dim, eps=eps).to(device)
            new_submodule.load_state_dict(submodule.state_dict())
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], new_submodule))

    for parent, attr_name, new_submodule in replacements:
        setattr(parent, attr_name, new_submodule)