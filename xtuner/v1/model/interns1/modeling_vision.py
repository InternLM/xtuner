from torch import nn
import torch
from typing import cast, Union, Optional
import numpy as np
from pathlib import Path

# TODO: 等 interns1 合入后全部换成 interns1 的实现
from transformers.models.internvl.modeling_internvl import InternVLVisionRMSNorm, \
    InternVLVisionEmbeddings, InternVLVisionMLP, NORM2FN, ACT2FN
from transformers.modeling_outputs import BaseModelOutput

try:
    from timm.layers import DropPath

    has_timm = True
except:
    has_timm = False
from tqdm import tqdm
from flash_attn import flash_attn_varlen_func
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_torch_device_module
from xtuner.v1.model import BaseModel
from xtuner.v1.config import FSDPConfig, InternS1VisionConfig, InternS1Config
from xtuner.v1.float8.float8_handler import Float8Handler
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from xtuner.v1.utils.compile import maybe_compile
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def init_world_mesh(fsdp_config: FSDPConfig):
    device = DEVICE if not fsdp_config.cpu_offload else "cpu"
    world_size = dist.get_world_size()

    # TODO: Support hsdp_sharding_size
    fsdp_mesh = init_device_mesh(device, (world_size,))
    return fsdp_mesh


class InternS1VisionAttention(nn.Module):
    def __init__(self, config: InternS1VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.q_norm = InternVLVisionRMSNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.k_norm = InternVLVisionRMSNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor
    ):
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).flatten(0, 1)
        key_states = key_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim).flatten(0, 1)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).flatten(0, 1)

        cu_seq_lens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len,
                                   dtype=torch.int32,
                                   device=hidden_states.device)
        attn_output: torch.Tensor = cast(
            torch.Tensor,
            flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seq_lens,
                cu_seqlens_k=cu_seq_lens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                causal=False,
                deterministic=XTUNER_DETERMINISTIC,
            ),
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output


class InternS1VisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: InternS1VisionConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.seq_len_dim = 1
        self.attention = InternS1VisionAttention(config)
        self.mlp = InternVLVisionMLP(config)

        self.layernorm_before = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        assert has_timm, 'timm is not installed, please install it to use DropPath'
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    @maybe_compile(fullgraph=True)
    def attention_pre_forward(self, hidden_states):
        attention_output = self.attention(self.layernorm_before(hidden_states))
        attention_output = self.lambda_1 * attention_output
        return attention_output

    @maybe_compile(fullgraph=True)
    def attention_post_forward(self, hidden_states):
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.mlp(layer_output)
        return layer_output

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        attention_output = self.attention_pre_forward(hidden_states)
        hidden_states = self.drop_path1(attention_output) + hidden_states

        layer_output = self.attention_post_forward(hidden_states)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        layer_output = self.drop_path2(layer_output) + hidden_states

        return layer_output


class InternS1VisionEncoder(nn.Module):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.config = config
        dpr = np.linspace(0.0, float(config.drop_path_rate), int(config.num_hidden_layers))
        self.layer = nn.ModuleList([
            InternS1VisionLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_hidden_states: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states
        )


class InternS1VisionModel(BaseModel):
    config: InternS1VisionConfig

    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = InternVLVisionEmbeddings(config)
        self.encoder = InternS1VisionEncoder(config)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        self._hf_prefix = "model.vision_tower."
        self._init_load_spec()

    def forward(
            self,
            pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output, _ = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix+key]

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        assert float8_handler is None

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        device = "cpu" if fsdp_config.cpu_offload else str(DEVICE)

        self.fsdp_mesh = init_world_mesh(fsdp_config)
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

        recompute_ratio = 1.0
        num_recompute_layers = int(len(self.encoder.layer) * recompute_ratio)
        for layer_idx in tqdm(list(range(len(self.encoder.layer))), desc="[Vision Fully Shard]"):
            layer = self.encoder.layer[layer_idx]
            layer.to_empty(device=device)

            if layer_idx < num_recompute_layers:
                layer = ptd_checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.REENTRANT)

            self.encoder.layer[layer_idx] = layer

            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
                offload_policy=CPUOffloadPolicy()
                if fsdp_config.cpu_offload
                else None,
            )

        for layer_cur, layer_next in zip(self.encoder.layer[:-1], self.encoder.layer[1:]):
            layer_cur.set_modules_to_forward_prefetch([layer_next])

        self.embeddings.to_empty(device=device)
        self.layernorm.to_empty(device=device)

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

        return self


class InternS1MultiModalProjector(BaseModel):
    config: InternS1Config

    def __init__(self, config: InternS1Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.text_config.hidden_size
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

        self._hf_prefix = "model.multi_modal_projector."
        self._init_load_spec()

    @maybe_compile(fullgraph=True)
    def forward(self, image_features):
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def to_hf_key_list(self, key: str) -> list[str]:
        return [self._hf_prefix + key]

    def fully_shard(
        self,
        fsdp_config: FSDPConfig,
        float8_handler: Float8Handler | None = None,
    ):
        assert float8_handler is None
        self._init_load_spec()

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype
        )
        device = "cpu" if fsdp_config.cpu_offload else str(DEVICE)

        self.fsdp_mesh = init_world_mesh(fsdp_config)
        assert self.fsdp_mesh is not None

        # self.checkpoint_wrapped = ptd_checkpoint_wrapper(self, checkpoint_impl=CheckpointImpl.REENTRANT)
        if fsdp_config.requires_grad:
            for module in self.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in self.parameters():
                param.requires_grad = False

        self.to_empty(device=device)

        fully_shard(
            self,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )
        return self

