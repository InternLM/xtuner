from typing import List
from xtuner.v1.ops.act_fn import get_act_fn
import torch.nn as nn
import torch
from typing_extensions import override
from .qwen3_vl_config import Qwen3VLVisionConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module, init_params
from xtuner.v1.ops.attn_imp import attn_impl_mapping
import torch.nn.functional as F
from pathlib import Path
from xtuner.v1.model import BaseModel
from functools import partial
from xtuner.v1.config import FSDPConfig
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers.models.llama.modeling_llama import repeat_kv
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.model.utils.checkpointing import checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def init_world_mesh():
    device = DEVICE
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scale = self.head_dim ** -0.5
        self.config = config
        self.attention_dropout = 0.0
        self.attn_impl_func = attn_impl_mapping[config.attn_impl]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sequence_parallel_mesh: DeviceMesh | None = None,
    ):
        seq_length = hidden_states.shape[0]  # s, d
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            sp_size = sequence_parallel_mesh.size()
            num_kv_heads = key_states.size(1)
            if sp_size > num_kv_heads:
                assert sp_size % num_kv_heads == 0
                key_states = repeat_kv(key_states, sp_size // num_kv_heads)
                value_states = repeat_kv(value_states, sp_size // num_kv_heads)

            query_states = ulysses_all_to_all(
                query_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=sequence_parallel_mesh,
            )
            key_states = ulysses_all_to_all(
                key_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=sequence_parallel_mesh,
            )
            value_states = ulysses_all_to_all(
                value_states,
                scatter_dim=1,
                gather_dim=2,
                mesh=sequence_parallel_mesh,
            )

        attn_output: torch.Tensor = self.attn_impl_func(  # type: ignore
            query_states,  # [b, n_head, seq, head_dim]
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scale,
            causal=False,
            deterministic=XTUNER_DETERMINISTIC
        )  # [b, seq, n_head, head_dim]

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            attn_output = ulysses_all_to_all(
                attn_output,
                scatter_dim=1,
                gather_dim=2,
                mesh=sequence_parallel_mesh,
            )

        attn_output = attn_output[0].reshape(seq_length, -1).contiguous()  # s, d
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionLayer(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    # @maybe_compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        sequence_parallel_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            sequence_parallel_mesh=sequence_parallel_mesh,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(BaseModel):
    config: Qwen3VLVisionConfig

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__(config)  # type: ignore[arg-type]
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.rotary_pos_emb = self.build_rotary_embedding(config)

        self.blocks = nn.ModuleList([Qwen3VLVisionLayer(config) for _ in range(config.depth)])

        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self._hf_prefix = "model.visual."
        self._init_load_spec()

    def build_rotary_embedding(self, config: Qwen3VLVisionConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        return Qwen3VLVisionRotaryEmbedding(head_dim // 2)

    @torch.no_grad()
    def init_weights(self):
        # If the model is trained from scratch, this will be triggered. It has not been strictly tested.
        initialized_params: set[str] = set()

        init_params(self.patch_embed.proj.weight,
                    partial(torch.nn.init.normal_, mean=0.0, std=self.config.initializer_range))
        initialized_params.add(f"patch_embed.proj.weight")
        if self.patch_embed.proj.bias is not None:
            init_params(self.patch_embed.proj.bias, torch.nn.init.zeros_)
            initialized_params.add(f"patch_embed.proj.bias")

        init_params(self.pos_embed.weight,
                    partial(torch.nn.init.normal_, mean=0.0, std=self.config.initializer_range))
        initialized_params.add(f"pos_embed.weight")

        for layer_idx, layer in enumerate(self.blocks):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    init_params(module.weight,
                                partial(torch.nn.init.normal_, mean=0.0, std=self.config.initializer_range))
                    initialized_params.add(f"blocks.{layer_idx}.{name}.weight")
                    if module.bias is not None:
                        init_params(module.bias, torch.nn.init.zeros_)
                        initialized_params.add(f"blocks.{layer_idx}.{name}.bias")

                elif isinstance(module, nn.LayerNorm):
                    init_params(module.weight, torch.nn.init.ones_)
                    init_params(module.bias, torch.nn.init.zeros_)
                    initialized_params.add(f"blocks.{layer_idx}.{name}.weight")
                    initialized_params.add(f"blocks.{layer_idx}.{name}.bias")

        expected_param_name = {self._clean_param_name(name) for name, _ in self.named_parameters()}
        if (missing := expected_param_name - initialized_params):
            raise RuntimeError(f"{missing} is not initialized")

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    @override
    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple:
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict)
        # If model is built on meta device, we need to rebuild rotary embedding since from_hf will not
        # load the `inv_freq` of RotaryEmbedding which is a inpersisitent buffer.
        self.rotary_pos_emb = self.build_rotary_embedding(self.config)
        return loaded_keys, unloaded_keys, missing_keys

    @override
    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        self.fsdp_config = fsdp_config
        assert float8_handler is None

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )

        # NOTE: 在 cpu_offload 模式下，mesh 应该是 cuda 的，在 meta fully_shard 后在调用 .to_empty(device=cpu)
        self.fsdp_mesh = init_world_mesh()
        assert self.fsdp_mesh is not None

        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        self.rotary_pos_emb = self.build_rotary_embedding(self.config)

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        recompute_ratio = 1.0
        num_recompute_layers = int(len(self.blocks) * recompute_ratio)
        for layer_idx in tqdm(list(range(len(self.blocks))), desc="[Vision Fully Shard]"):
            layer = self.blocks[layer_idx]

            if layer_idx < num_recompute_layers:
                layer = checkpoint_wrapper(layer,
                                           preserve_rng_state=checkpoint_preserve_rng_state,
                                           checkpoint_impl=CheckpointImpl.REENTRANT)
            if self.compile_cfg:
                layer.forward = torch.compile(layer.forward, fullgraph=True)

            self.blocks[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy()
                if fsdp_config.cpu_offload
                else None,
            )

        for layer_cur, layer_next in zip(self.blocks[:-1],  self.blocks[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor,
                sequence_parallel_mesh: DeviceMesh | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            # To ensure that the sequence length after sp split is divisible by 4,
            # we require that the sequence length before sp split is also divisible by 4.
            assert max_seqlen % 4 == 0, f"max_seqlen {max_seqlen} must be divisible by 4. Please check dataset setting."
            div_num = sequence_parallel_mesh.size() * 4
            hidden_states = pad_to_multiple_of(hidden_states, 0, div_num, 0)
            hidden_states = split_for_sequence_parallel(hidden_states, dim=0, sp_mesh=sequence_parallel_mesh)
            pos_embeds = pad_to_multiple_of(pos_embeds, 0, div_num, 0)
            pos_embeds = split_for_sequence_parallel(pos_embeds, dim=0, sp_mesh=sequence_parallel_mesh)
            rotary_pos_emb = pad_to_multiple_of(rotary_pos_emb, 0, div_num, 0)
            rotary_pos_emb = split_for_sequence_parallel(rotary_pos_emb, dim=0, sp_mesh=sequence_parallel_mesh)

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1).to(hidden_states.device)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                position_embeddings,
                sequence_parallel_mesh
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = hidden_states
                deepstack_feature_lists.append(deepstack_feature)
     
        return hidden_states, deepstack_feature_lists
