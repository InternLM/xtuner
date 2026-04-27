# Copyright (c) OpenMMLab. All rights reserved.

from typing import Annotated, cast

import torch
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import get_logger

from ..linear import build_linear
from .attn_outputs import AttnOutputs


# Temporary solution: use separate function objects for each call site, Dynamo will cache them separately
def _all_to_all_conv_pre_qk(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_v(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_gb(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_out(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    class FusedRMSNormGated(FLA_FusedRMSNormGated):
        def forward(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
        ) -> torch.Tensor:
            weight = self.weight
            if isinstance(weight, DTensor):
                weight = weight.to_local()

            return rms_norm_gated(
                x,
                g,
                weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

except ImportError:
    FusedRMSNormGated = None  # type: ignore
    chunk_gated_delta_rule = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

logger = get_logger()


class GatedDeltaNetConfig(BaseModel):
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
    ) -> "GatedDeltaNet":
        return GatedDeltaNet(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
        )


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str,
        rms_norm_eps: float,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
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

        assert causal_conv1d_fn is not None, (
            "causal_conv1d_fn is not available. Please install causal-conv1d to use GatedDeltaNet by `https://github.com/Dao-AILab/causal-conv1d`."
        )
        self.causal_conv1d_fn = causal_conv1d_fn
        assert chunk_gated_delta_rule is not None, (
            "chunk_gated_delta_rule is not available. Please install fla to use GatedDeltaNet by `pip install flash-linear-attention`."
        )
        self.chunk_gated_delta_rule = chunk_gated_delta_rule
        assert FusedRMSNormGated is not None, (
            "FusedRMSNormGated is not available. Please install fla to use GatedDeltaNet by `pip install flash-linear-attention`."
        )
        self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.rms_norm_eps, activation=self.activation)

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
        self.in_proj_b = build_linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = build_linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward_for_sp(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
    ) -> AttnOutputs:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        # TODO: If full_graph mode is supported in the future, it needs to be modified to custom_op
        if seq_ctx.seq_idx is None:
            seq_idx = torch.cat(
                [
                    torch.full((s,), i, dtype=torch.int32, device=mixed_qkv.device)
                    for i, s in enumerate(seq_ctx.seq_lens_q)
                ],
                dim=0,
            )[None]
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        query, key, value = torch.split(
            mixed_qkv,  # (1, L/sp_size, 8192)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        # (1, L, 8192/sp_size)
        query = query.transpose(1, 2)  # (1, dim, L/sp_size)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = _all_to_all_conv_pre_qk(
            query,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        key = _all_to_all_conv_pre_qk(
            key,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        value = _all_to_all_conv_pre_v(
            value,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )

        # query =  (1, dim/sp_size, L)
        query_weight, key_weight, value_weight = torch.split(
            weight,  # (8192, 4)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )

        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_rank = seq_ctx.sequence_parallel_mesh.get_local_rank()
        sp_size = seq_ctx.sequence_parallel_mesh.size()
        query_weight = query_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        key_weight = key_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        value_weight = value_weight.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(seq_ctx.sequence_parallel_mesh.size(), dim=0)[sp_rank]

        query = query.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn
        key = key.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn
        value = value.transpose(1, 2).contiguous().transpose(1, 2)  # make it contiguous for causal_conv1d_fn
        query = self.causal_conv1d_fn(  # query (batch, dim, seqlen)
            x=query,  # need non contiguous
            weight=query_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        key = self.causal_conv1d_fn(
            x=key,  # need non contiguous
            weight=key_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        value = self.causal_conv1d_fn(
            x=value,  # need non contiguous
            weight=value_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        # (1,key_dim/sp_size, L)
        query = query.transpose(1, 2).reshape(
            batch_size, seq_len * sp_size, -1, self.head_k_dim
        )  # (1, L, num_k_heads/sp_size, head_k_dim)
        key = key.transpose(1, 2).reshape(
            batch_size, seq_len * sp_size, -1, self.head_k_dim
        )  # (1, L, num_k_heads/sp_size, head_k_dim)
        value = value.transpose(1, 2).reshape(
            batch_size, seq_len * sp_size, -1, self.head_v_dim
        )  # (1, L, num_v_heads/sp_size, head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)

            g = _all_to_all_gb(
                g,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            beta = _all_to_all_gb(
                beta,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
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

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            core_attn_out = _all_to_all_out(
                core_attn_out,  # (1, L, num_v_head/sp_size, head_dim)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
    ) -> AttnOutputs:
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx, position_embeddings)

        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        # TODO: If full_graph mode is supported in the future, it needs to be modified to custom_op
        if seq_ctx.seq_idx is None:
            seq_idx = torch.cat(
                [
                    torch.full((s,), i, dtype=torch.int32, device=mixed_qkv.device)
                    for i, s in enumerate(seq_ctx.seq_lens_q)
                ],
                dim=0,
            )[None]
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        # TODO: due to the limitation of scatter_dim=1 in ulysses_all_to_all,
        # the implementation is very inelegant and inefficient, and needs to be refactored in the future.
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,  # need non contiguous
            weight=weight,
            bias=bias,
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
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
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
