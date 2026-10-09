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

from ...ops.gated_deltanet import get_causal_conv1d_fn, get_chunk_gated_delta_rule_fn, get_rms_norm_gated_cls
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


def _repeat_qk_heads(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat Q/K heads while preserving the backend-native storage layout.

    The logical input and output layout is always ``[B, T, H, K]``. Supported
    physical layouts are:

    - contiguous time-major ``[B, T, H, K]``: repeat logical head dim 2;
    - contiguous head-first ``[B, H, T, K]`` exposed through a
      ``[B, T, H, K]`` transpose view: repeat native head dim 1 and return the
      corresponding time-major view;
    - any other strided ``[B, T, H, K]`` view: fall back to repeating logical
      head dim 2, which materializes a time-major output.
    """
    if repeats == 1:
        return x

    head_first = x.transpose(1, 2)
    time_major_contiguous = x.is_contiguous()
    head_first_contiguous = head_first.is_contiguous()
    if head_first_contiguous and (not time_major_contiguous or x.stride(2) != x.shape[-1]):
        return head_first.repeat_interleave(repeats, dim=1).transpose(1, 2)
    return x.repeat_interleave(repeats, dim=2)


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
        if num_value_heads % num_key_heads != 0:
            raise ValueError(
                "GatedDeltaNet requires num_value_heads to be an integer multiple of num_key_heads, "
                f"got num_value_heads={num_value_heads} and num_key_heads={num_key_heads}"
            )
        if value_head_dim != key_head_dim:
            raise ValueError(
                "GatedDeltaNet requires value_head_dim to equal key_head_dim, "
                f"got value_head_dim={value_head_dim} and key_head_dim={key_head_dim}"
            )
        self.name = f"layers.{layer_idx}.gate_deltanet"
        self.float8_cfg = float8_cfg

        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.qk_head_repeat = self.num_v_heads // self.num_k_heads
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

        # Resolved at build time so `XTUNER_HF_IMPL` selects the HF-native fla call patterns
        # (same convention as `get_attn_impl_fn`).
        self.causal_conv1d_fn = get_causal_conv1d_fn()
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule_fn()
        rms_norm_gated_cls = get_rms_norm_gated_cls()
        self.norm = rms_norm_gated_cls(self.head_v_dim, eps=self.rms_norm_eps, activation=self.activation)

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

    def get_muon_split_sizes(self) -> dict[nn.Parameter, tuple[int, ...]]:
        """Return the logical Q, K, and V row blocks for MuonSplit."""
        return {
            cast(nn.Parameter, self.in_proj_qkv.weight): (
                self.key_dim,
                self.key_dim,
                self.value_dim,
            )
        }

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

        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_rank = seq_ctx.sp_rank
        sp_size = seq_ctx.sequence_parallel_mesh.size()
        if self.num_k_heads % sp_size != 0 or self.num_v_heads % sp_size != 0:
            raise ValueError(
                "GatedDeltaNet requires num_key_heads and num_value_heads to be divisible by the sequence "
                f"parallel size, got num_key_heads={self.num_k_heads}, num_value_heads={self.num_v_heads}, "
                f"sp_size={sp_size}"
            )

        query, key, value = torch.split(
            mixed_qkv,  # (1, L/sp_size, 8192)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
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

        query_weight, key_weight, value_weight = torch.split(
            weight,  # (8192, 4)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )
        query_weight = query_weight.chunk(sp_size, dim=0)[sp_rank]
        key_weight = key_weight.chunk(sp_size, dim=0)[sp_rank]
        value_weight = value_weight.chunk(sp_size, dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(sp_size, dim=0)[sp_rank]

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        query = query.reshape(batch_size, seq_len * sp_size, self.num_k_heads // sp_size, self.head_k_dim)
        key = key.reshape(batch_size, seq_len * sp_size, self.num_k_heads // sp_size, self.head_k_dim)
        value = value.reshape(batch_size, seq_len * sp_size, self.num_v_heads // sp_size, self.head_v_dim)
        gdn_metadata = seq_ctx.gdn_metadata
        seq_idx = gdn_metadata.seq_idx if gdn_metadata is not None else None
        cu_seqlens_int64 = gdn_metadata.cu_seqlens_int64 if gdn_metadata is not None else None
        chunk_indices = gdn_metadata.chunk_indices if gdn_metadata is not None else None
        chunk_indices_list = gdn_metadata.chunk_indices_list if gdn_metadata is not None else None
        query = self.causal_conv1d_fn(
            x=query,
            weight=query_weight,
            bias=bias,
            activation=self.activation,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            seq_idx=seq_idx,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
        )
        key = self.causal_conv1d_fn(
            x=key,
            weight=key_weight,
            bias=bias,
            activation=self.activation,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            seq_idx=seq_idx,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
        )
        value = self.causal_conv1d_fn(
            x=value,
            weight=value_weight,
            bias=bias,
            activation=self.activation,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            seq_idx=seq_idx,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
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

        query = query.reshape(batch_size, seq_len * sp_size, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len * sp_size, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len * sp_size, -1, self.head_v_dim)

        if self.qk_head_repeat > 1:
            query = _repeat_qk_heads(query, self.qk_head_repeat)
            key = _repeat_qk_heads(key, self.qk_head_repeat)

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
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
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

        mixed_qkv = mixed_qkv.reshape(
            batch_size,
            seq_len,
            2 * self.num_k_heads + self.num_v_heads,
            self.head_k_dim,
        )
        gdn_metadata = seq_ctx.gdn_metadata
        seq_idx = gdn_metadata.seq_idx if gdn_metadata is not None else None
        cu_seqlens_int64 = gdn_metadata.cu_seqlens_int64 if gdn_metadata is not None else None
        chunk_indices = gdn_metadata.chunk_indices if gdn_metadata is not None else None
        chunk_indices_list = gdn_metadata.chunk_indices_list if gdn_metadata is not None else None
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=weight,
            bias=bias,
            activation=self.activation,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            seq_idx=seq_idx,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            dim=2,
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

        if self.qk_head_repeat > 1:
            query = _repeat_qk_heads(query, self.qk_head_repeat)
            key = _repeat_qk_heads(key, self.qk_head_repeat)

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
            cu_seqlens_list=seq_ctx.cu_seq_lens_q_list,
            cu_seqlens_int64=cu_seqlens_int64,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
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
