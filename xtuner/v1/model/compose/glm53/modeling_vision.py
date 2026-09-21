# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash vision tower, see doc/xtuner_glm5p3flash_design.md F2 §4/§5.

``vision_tower`` (this file) covers HF `Glm5NextVisionModel` minus its `downsample`/`merger`,
which XTuner assigns to ``Glm53Projector`` instead (§5.2 module boundary; the HF checkpoint key
mapping in §6.1 confirms this is a deliberate XTuner-side split, not an HF module boundary).
"""

from typing import Callable, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model import BaseModel
from xtuner.v1.module import AttnOutputs, RMSNorm
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.ops.attn_imp import AttnOpOutputs, get_attn_impl_fn
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module
from xtuner.v1.utils.init_weight import default_init_weights

from .glm53_config import Glm53VisionConfig


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def init_world_mesh() -> DeviceMesh:
    world_size = dist.get_world_size()
    return init_device_mesh(DEVICE, (world_size,))


def _get_vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Merge-block-major (h, w) position ids per patch; video rows repeat over
    `t` (§2.2/§4).

    Native port of `transformers.vision_utils.get_vision_position_ids(include_temporal=False)` --
    kept in-tree per the "native implementation, not an HF training-time dependency" principle
    (design doc §3.2), verified against that function bitwise in tests/model/test_glm53_vision.py.
    """
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        block_shape = (h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def _get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Per-tubelet attention boundaries (`merge_temporal=False`): each temporal
    slice of each image/video row is an independent ViT attention sequence
    (§2.2/§4)."""
    seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = seqlens.cumsum(dim=0).to(torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


class Glm53VisionRotaryEmbedding(nn.Module):
    """Axial 2D RoPE, no time axis: `[h, w, h, w]` frequency layout covering
    the whole head dim.

    Native port of `Glm5NextVisionRotaryEmbedding` (direct outer product per position, no
    embedding-table lookup -- GLM doesn't use qwen3_vl's freq-table + gather trick).
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (N, 2), row = (h, w) index of that patch.
        freqs = position_ids[..., None].float() * self.inv_freq.float()  # (N, 2, spatial_dim // 2)
        cos, sin = self._recompose(freqs.cos()), self._recompose(freqs.sin())
        return cos, sin

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Glm53VisionMLP(nn.Module):
    def __init__(self, config: Glm53VisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.act_fn = get_act_fn(config.hidden_act)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_state).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_state).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm53VisionAttention(nn.Module):
    def __init__(self, config: Glm53VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=config.attention_bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=config.attention_bias)
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_impl_func: Callable[..., AttnOpOutputs] = get_attn_impl_fn(config.attn_impl)  # type: ignore[assignment]

    def get_muon_split_sizes(self) -> dict[nn.Parameter, tuple[int, ...]]:
        return {cast(nn.Parameter, self.qkv.weight): (self.dim, self.dim, self.dim)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> AttnOutputs:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            query_states = ulysses_all_to_all(query_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh)
            key_states = ulysses_all_to_all(key_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh)
            value_states = ulysses_all_to_all(value_states, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh)

        attn_op_outputs = self.attn_impl_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scale,
            causal=False,
            deterministic=XTUNER_DETERMINISTIC,
        )

        raw_output = attn_op_outputs["raw_output"]
        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            raw_output = ulysses_all_to_all(raw_output, scatter_dim=1, gather_dim=2, mesh=sequence_parallel_mesh)

        raw_output = raw_output[0].reshape(seq_length, -1).contiguous()
        projected_output = self.proj(raw_output)
        return {"projected_output": projected_output, **attn_op_outputs}


class Glm53VisionBlock(nn.Module):
    def __init__(self, config: Glm53VisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm53VisionAttention(config)
        self.mlp = Glm53VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states
            + self.attn(
                self.norm1(hidden_states), cu_seqlens, max_seqlen, position_embeddings, sequence_parallel_mesh
            )["projected_output"]
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Glm53VisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm53VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Glm53VisionModel(BaseModel):
    config: Glm53VisionConfig

    def __init__(self, config: Glm53VisionConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.spatial_merge_size = config.spatial_merge_size
        head_dim = config.hidden_size // config.num_heads

        self.patch_embed = Glm53VisionPatchEmbed(config)
        self.rotary_pos_emb = Glm53VisionRotaryEmbedding(head_dim, theta=config.rope_parameters["rope_theta"])
        self.blocks = nn.ModuleList([Glm53VisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._hf_prefix = "model.visual."
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    @torch.no_grad()
    def init_weights(self) -> None:
        initialized = default_init_weights(self)
        if missing := {name for name, _ in self.named_parameters()} - initialized:
            raise RuntimeError(f"{missing} is not initialized")

    @override
    def fully_shard(self, fsdp_config: FSDPConfig):
        self.fsdp_config = fsdp_config
        self.fsdp_mesh = init_world_mesh()

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        setattr(module, p_name, torch.nn.Parameter(param.to(dtype=torch.float32)))
        else:
            for param in self.parameters():
                param.requires_grad = False

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype, cast_forward_inputs=False
        )
        if self.config.fully_shard:
            for block in self.blocks:
                self._fully_shard(
                    mesh=self.fsdp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=True,
                    offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
                    module=block,
                )
            self._fully_shard(
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        # `grid_thw` must already be expanded (video rows are [1, h, w], see vision_utils.py /
        # F1.d); the tower is t-aware regardless (repeats position ids/cu_seqlens over t), but
        # mixing representations silently changes num_img_tokens/feature-grouping downstream.
        position_ids = _get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _get_vision_cu_seqlens(grid_thw)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        hidden_states = self.patch_embed(hidden_states)
        cos, sin = self.rotary_pos_emb(position_ids.to(hidden_states.device))
        position_embeddings = (cos.to(hidden_states.device), sin.to(hidden_states.device))

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, position_embeddings, sequence_parallel_mesh)

        return self.post_layernorm(hidden_states)
