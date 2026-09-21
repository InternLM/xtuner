# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Delta Attention (KDA) for GLM-5.3-Flash.

Module structure and parameter names follow the published checkpoint layout (see
``doc/xtuner_glm5p3flash_design.md`` section 3.1/3.5.1): separate ``q/k/v_conv1d``, a
low-rank ``f_a_proj -> f_b_proj`` forget gate with fp32 ``A_log``/``dt_bias``, and a
low-rank ``g_a_proj -> g_b_proj`` output gate.

The installed ``fla`` package's ``chunk_kda`` (0.4.2) does **not** accept ``A_log`` /
``dt_bias`` / ``use_beta_sigmoid_in_kernel`` -- passing them (as e.g. an Automodel-style call
convention does) silently drops them into ``**kwargs`` and the gate ends up computed wrong
without raising. The gate must be computed *before* calling the kernel via
``fla.ops.kda.gate.fused_kda_gate``, and ``beta`` must be sigmoid'd explicitly. This matches
HF's own reference ``Glm5NextTextLinearAttention.forward``, which also precomputes the gate
via a separate ``forget_gate`` module before calling the (kernelizable) chunk/recurrent fns.
"""

from __future__ import annotations

from typing import Annotated, Any

import torch
import torch.nn as nn
from cyclopts import Parameter
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch.distributed.tensor import DTensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all

from ..linear import build_linear
from .attn_outputs import AttnOutputs


# Separate call-site wrappers so Dynamo caches each SP collective independently.
def _all_to_all_conv_pre_q(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_k(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_v(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_g(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_beta(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_out(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _to_local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if isinstance(param, DTensor) else param


# Sequences at or below this length use the recurrent kernel (matches Automodel's dispatch).
_CHUNK_KERNEL_MIN_SEQ_LEN = 64

_fla_kda_import_error: BaseException | None = None

try:
    from fla.modules import FusedRMSNormGated as _FLAFusedRMSNormGated
    from fla.modules import ShortConvolution as _FLAShortConvolution
    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
    from fla.ops.kda import chunk_kda as _chunk_kda
    from fla.ops.kda import fused_recurrent_kda as _fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate as _fused_kda_gate

    class FusedRMSNormGated(_FLAFusedRMSNormGated):
        pass

    class KDAShortConvolution(_FLAShortConvolution):
        """Adds an explicit ``weight``/``bias`` override so SP can run the same
        conv entry with a per-rank channel slice instead of this module's own
        (full) parameters."""

        def materialize_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
            weight = rearrange(_to_local(self.weight), "d 1 w -> d w")
            bias = _to_local(self.bias) if self.bias is not None else None
            return weight, bias

        def forward(  # type: ignore[override]
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            weight: torch.Tensor | None = None,
            bias: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if weight is None:
                weight, bias = self.materialize_weight_bias()
            return _fla_causal_conv1d(
                x=x,
                weight=weight,
                bias=bias,
                activation=self.activation,
                backend=self.backend,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

    chunk_kda = _chunk_kda
    fused_recurrent_kda = _fused_recurrent_kda
    fused_kda_gate = _fused_kda_gate
except (ImportError, ModuleNotFoundError) as e:
    has_fla_kda = False
    chunk_kda = None  # type: ignore[assignment]
    fused_recurrent_kda = None  # type: ignore[assignment]
    fused_kda_gate = None  # type: ignore[assignment]
    FusedRMSNormGated = None  # type: ignore[assignment,misc]
    KDAShortConvolution = None  # type: ignore[assignment,misc]
    _fla_kda_import_error = e
else:
    has_fla_kda = True
    _fla_kda_import_error = None


class KDAConfig(BaseModel):
    """Kimi Delta Attention config.

    Field names match the published checkpoint 1:1.
    """

    model_config = ConfigDict(title="Kimi Delta Attention config", extra="forbid")
    num_heads: Annotated[int, Parameter(group="attention")]
    head_dim: Annotated[int, Parameter(group="attention")]
    conv_kernel_size: Annotated[int, Parameter(group="attention")] = 4
    # GLM-5.3-Flash uses a low-rank g_a_proj -> g_b_proj output gate.
    use_full_rank_gate: Annotated[bool, Parameter(group="attention")] = False
    gate_lower_bound: Annotated[float | None, Parameter(group="attention")] = -5.0
    rms_norm_eps: Annotated[float, Parameter(group="attention")] = 1e-5

    def build(
        self,
        hidden_size: int,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> KimiDeltaAttention:
        del kwargs  # layer_type / rope_scaling_cfg / generate_config: KDA is NoPE, unused.
        return KimiDeltaAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
            layer_idx=layer_idx,
        )


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention matching HF ``Glm5NextTextLinearAttention``
    numerically.

    Parameter names match the published checkpoint's ``self_attn.*`` keys, which differ from
    HF's internal module layout (HF fuses ``q/k/v_conv1d`` into one ``conv1d``, and moves the
    forget-gate params under a ``forget_gate`` submodule); transformers' built-in conversion
    mapping bridges the two on load, so XTuner doesn't need a bridging module (design doc
    section 3.1).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        use_full_rank_gate: bool = False,
        gate_lower_bound: float | None = -5.0,
        rms_norm_eps: float = 1e-5,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        if not has_fla_kda:
            assert _fla_kda_import_error is not None
            raise ImportError("Please install fla (`pip install -U fla-core`)") from _fla_kda_import_error
        assert KDAShortConvolution is not None
        assert FusedRMSNormGated is not None
        assert chunk_kda is not None
        assert fused_recurrent_kda is not None
        assert fused_kda_gate is not None

        self.name = f"layers.{layer_idx}.self_attn"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound
        self.rms_norm_eps = rms_norm_eps
        self.layer_idx = layer_idx
        self.float8_cfg = float8_cfg

        projection_size = head_dim * num_heads
        self.q_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        self.k_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        self.v_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)

        self.q_conv1d = KDAShortConvolution(
            hidden_size=projection_size, kernel_size=conv_kernel_size, activation="silu"
        )
        self.k_conv1d = KDAShortConvolution(
            hidden_size=projection_size, kernel_size=conv_kernel_size, activation="silu"
        )
        self.v_conv1d = KDAShortConvolution(
            hidden_size=projection_size, kernel_size=conv_kernel_size, activation="silu"
        )

        self.f_a_proj = build_linear(hidden_size, head_dim, bias=False, float8_cfg=float8_cfg)
        self.f_b_proj = build_linear(head_dim, projection_size, bias=False, float8_cfg=float8_cfg)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.b_proj = build_linear(hidden_size, num_heads, bias=False, float8_cfg=float8_cfg)

        if use_full_rank_gate:
            self.g_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        else:
            self.g_a_proj = build_linear(hidden_size, head_dim, bias=False, float8_cfg=float8_cfg)
            self.g_b_proj = build_linear(head_dim, projection_size, bias=False, float8_cfg=float8_cfg)

        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = build_linear(projection_size, hidden_size, bias=False, float8_cfg=float8_cfg)

    def _select_kernel(self, seq_len: int, cp_context: Any | None):
        # Automodel's dispatch: short (unpacked) sequences use the recurrent kernel; long
        # sequences, or anything running under context parallel, use the chunked kernel.
        if cp_context is not None or seq_len > _CHUNK_KERNEL_MIN_SEQ_LEN:
            return chunk_kda
        return fused_recurrent_kda

    def _compute_gate_and_beta(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).view(batch_size, seq_len, self.num_heads, self.head_dim)
        gate = fused_kda_gate(
            g_raw,
            _to_local(self.A_log),
            dt_bias=_to_local(self.dt_bias),
            lower_bound=self.gate_lower_bound,
        )
        beta = self.b_proj(hidden_states).float().sigmoid()
        return gate, beta

    def _gate_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_full_rank_gate:
            return self.g_proj(hidden_states)
        return self.g_b_proj(self.g_a_proj(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        **kwargs: Any,
    ) -> AttnOutputs:
        # Decoder layers may still pass rotary kwargs used by MLA; KDA is NoPE and ignores them.
        del kwargs
        if seq_ctx.sequence_parallel_mesh is not None and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx)

        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "KDA currently supports packed batch size 1"

        cu_seqlens = seq_ctx.cu_seq_lens_q
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cu_seqlens=cu_seqlens)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        gate, beta = self._compute_gate_and_beta(hidden_states)

        kernel = self._select_kernel(seq_len, cp_context=None)
        o, _ = kernel(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            transpose_state_layout=True,
            safe_gate=(self.gate_lower_bound is not None),
            cu_seqlens=cu_seqlens,
        )

        gate_out = self._gate_output(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        raw_output = self.o_norm(o, gate_out).reshape(batch_size, seq_len, -1)
        projected_output = self.o_proj(raw_output)
        return {"raw_output": raw_output, "projected_output": projected_output, "softmax_lse": None}

    def forward_for_sp(self, hidden_states: torch.Tensor, seq_ctx: SequenceContext) -> AttnOutputs:
        """Ulysses SP: local-seq/all-heads -> all-to-all -> full-seq/head-shard, run KDA
        rank-local (no cross-rank recurrent state needed), then all-to-all back.

        Precondition (enforced by ``SequenceContext.split``, see design doc section 3.6.1):
        shards must be contiguous and non-empty. A future zigzag/load-balanced SP split would
        silently break both the short convolution and the recurrent state.
        """
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "KDA currently supports packed batch size 1"
        sp_mesh = seq_ctx.sequence_parallel_mesh
        assert sp_mesh is not None
        sp_size = sp_mesh.size()
        sp_rank = seq_ctx.sp_rank
        assert self.num_heads % sp_size == 0, (
            f"KDA num_heads ({self.num_heads}) must be divisible by sp_size ({sp_size})"
        )

        cu_seqlens = seq_ctx.cu_seq_lens_q
        projection_size = self.num_heads * self.head_dim

        q = self._sp_short_conv(
            self.q_conv1d, self.q_proj(hidden_states), _all_to_all_conv_pre_q, sp_mesh, sp_rank, sp_size, cu_seqlens
        )
        k = self._sp_short_conv(
            self.k_conv1d, self.k_proj(hidden_states), _all_to_all_conv_pre_k, sp_mesh, sp_rank, sp_size, cu_seqlens
        )
        v = self._sp_short_conv(
            self.v_conv1d, self.v_proj(hidden_states), _all_to_all_conv_pre_v, sp_mesh, sp_rank, sp_size, cu_seqlens
        )

        # Local seq, full heads -> all_to_all -> full seq, head shard (same as GDN g/beta).
        g_raw = self.f_b_proj(self.f_a_proj(hidden_states))
        g_raw = g_raw.view(batch_size, seq_len, projection_size).transpose(1, 2)  # (B, H*D, L/sp)
        g_raw = _all_to_all_g(g_raw, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        g_raw = g_raw.transpose(1, 2).view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)

        beta = self.b_proj(hidden_states).float().transpose(1, 2)  # (B, H, L/sp)
        beta = _all_to_all_beta(beta, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        beta = beta.transpose(1, 2).sigmoid()  # (B, L, H/sp)

        q = q.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)
        k = k.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)
        v = v.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)

        a_log = _to_local(self.A_log).chunk(sp_size, dim=0)[sp_rank]
        dt_bias = (
            _to_local(self.dt_bias).view(self.num_heads, self.head_dim).chunk(sp_size, dim=0)[sp_rank].reshape(-1)
        )
        gate = fused_kda_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=self.gate_lower_bound)

        kernel = self._select_kernel(seq_len * sp_size, cp_context=None)
        o, _ = kernel(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            transpose_state_layout=True,
            safe_gate=(self.gate_lower_bound is not None),
            cu_seqlens=cu_seqlens,
        )

        # (B, L, H/sp, D) -> (B, L/sp, H, D)
        o = _all_to_all_out(o, scatter_dim=1, gather_dim=2, mesh=sp_mesh)

        gate_out = self._gate_output(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        raw_output = self.o_norm(o, gate_out).reshape(batch_size, seq_len, -1)
        projected_output = self.o_proj(raw_output)
        return {"raw_output": raw_output, "projected_output": projected_output, "softmax_lse": None}

    def _sp_short_conv(
        self,
        conv: KDAShortConvolution,
        x_local: torch.Tensor,
        all_to_all_fn,
        sp_mesh,
        sp_rank: int,
        sp_size: int,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gather the full sequence for a channel shard, then run the same conv
        entry as non-SP."""
        x = x_local.transpose(1, 2)
        x = all_to_all_fn(x, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        x = x.transpose(1, 2).contiguous()

        weight, bias = conv.materialize_weight_bias()
        weight = weight.chunk(sp_size, dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(sp_size, dim=0)[sp_rank]
        out, _ = conv(x, cu_seqlens=cu_seqlens, weight=weight, bias=bias)
        return out
