from hashlib import new
import types

import torch.nn as nn

from xtuner.v1.module import RMSNorm
from xtuner.v1.module.rope.rope import Qwen3VLTextRotaryEmbedding, RopeScalingConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_vision import Qwen3VLVisionModel
from pydantic import BaseModel


def patch_hf_rope(module: nn.Module) -> None:
    class FakeXTunerConfig(BaseModel):
        rope_scaling_cfg: RopeScalingConfig
        rope_theta: float = 100000.0
        max_position_embeddings: int
        hidden_size: int
        num_attention_heads: int
        head_dim: int

    replacements = []
    import torch

    for name, submodule in module.named_modules():
        if "Qwen3VLTextRotaryEmbedding" in submodule.__class__.__name__ and isinstance(submodule, nn.Module):
            hf_config = submodule.config
            config = FakeXTunerConfig(
                rope_theta=hf_config.rope_theta,
                max_position_embeddings=hf_config.max_position_embeddings,
                hidden_size=hf_config.hidden_size,
                head_dim=hf_config.head_dim,
                num_attention_heads=hf_config.num_attention_heads,
                rope_scaling_cfg=RopeScalingConfig(
                    mrope_section=hf_config.rope_scaling["mrope_section"],
                ),
            )
            new_submodule = Qwen3VLTextRotaryEmbedding(config)
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], new_submodule))

        if "Qwen3VLVisionModel" in submodule.__class__.__name__ and isinstance(submodule, nn.Module):
            submodule.__class__.fast_pos_embed_interpolate = Qwen3VLVisionModel.fast_pos_embed_interpolate

    for parent, attr_name, new_submodule in replacements:
        setattr(parent, attr_name, new_submodule)


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
