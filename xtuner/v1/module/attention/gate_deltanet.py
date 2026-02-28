# Copyright (c) OpenMMLab. All rights reserved.

from typing import Annotated, Callable, Literal, cast

import torch
from cyclopts import Parameter
from mmengine import is_installed
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import overload
import torch.nn.functional as F

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_device, get_logger

from ..linear import build_linear
from .attn_outputs import AttnOutputs

from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from causal_conv1d import causal_conv1d_fn

logger = get_logger()


class GateDeltaNetConfig(BaseModel):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="forbid")
    num_value_heads: Annotated[int, Parameter(group="attention")]
    num_key_heads: Annotated[int, Parameter(group="attention")]
    key_head_dim: Annotated[int, Parameter(group="attention")]
    value_head_dim: Annotated[int, Parameter(group="attention")]
    conv_kernel_dim: Annotated[int, Parameter(group="attention")]
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    rms_norm_eps: Annotated[float, Parameter(group="attention")]

    def build(
        self,
        hidden_size: int,
        float8_cfg: Float8Config | None = None,
        **kwargs,
    ) -> "GateDeltaNet":
        return GateDeltaNet(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
        )


class GateDeltaNet(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_value_heads: int, 
                 num_key_heads: int, 
                 key_head_dim: int, 
                 value_head_dim: int, 
                 conv_kernel_dim: int, 
                 hidden_act: str, 
                 rms_norm_eps: float,
                 layer_idx: int = 0,
                 float8_cfg: Float8Config | None = None) -> None:
        super().__init__()
        self.name = f"layers.{layer_idx}.gate_deltanet"
        self.float8_cfg = float8_cfg

        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = hidden_act
        self.rms_norm_eps = rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.causal_conv1d_fn = causal_conv1d_fn
        self.chunk_gated_delta_rule = chunk_gated_delta_rule

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.rms_norm_eps,
            activation=self.activation
        )

        self.out_proj = build_linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            float8_cfg=self.float8_cfg,
        )

        self.in_proj_qkv = build_linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_z = build_linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_b = build_linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_a = build_linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, # not used
    ) -> AttnOutputs:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size==1, "Only batch size of 1 is supported for now in GateDeltaNet"
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        num_tokens = seq_ctx.seq_lens_q
        seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, 
                                        device=hidden_states.device) for i, s in enumerate(num_tokens)], dim=0)[None]
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        
        core_attn_out, _ = self.chunk_gated_delta_rule( # TODO: packed sequence support
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )
        
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        attn_outputs: AttnOutputs = {
            "projected_output": output,
        }
        return attn_outputs
    
    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> AttnOutputs: ...

    __call__ = nn.Module.__call__
    
