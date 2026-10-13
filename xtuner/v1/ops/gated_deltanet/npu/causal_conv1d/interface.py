"""Ascend causal convolution autograd interface."""

from typing import Mapping, Optional

import torch

from ..metadata import get_npu_causal_conv1d_block_sizes, prepare_npu_metadata
from .causal_conv1d_triton_ascend import causal_conv1d_bwd_impl, causal_conv1d_fwd_impl


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
        output_final_state: bool = False,
    ):
        weight = weight.transpose(-1, -2).contiguous()
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.H = H

        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            H=H,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
            output_final_state=output_final_state,
        )
        ctx.final_state = final_state

        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        # The public time-major transpose is outside this autograd function.
        # Its backward normally restores head-first strides; materialize only
        # for callers whose downstream graph returns an incompatible view.
        if not dy.is_contiguous():
            dy = dy.contiguous()
        activation = ctx.activation
        cu_seqlens = ctx.cu_seqlens
        chunk_indices = ctx.chunk_indices
        H = ctx.H

        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=dy,
            H=H,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
        )

        return dx, dw.transpose(0, 1).contiguous(), None, db, dr, dh0, None, None, None, None


def causal_conv1d_triton_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the NPU-native causal convolution.

    Args:
        x: Input tensor of shape ``[B, T, D]``.
        weight: Weight tensor of shape ``[D, W]``.
        H: Number of heads in the output view.
        bias: Optional bias tensor of shape ``[D]``.
        residual: Optional residual tensor of shape ``[B, T, D]``.
        initial_state: Optional initial state for sequence processing.
        activation: Optional activation function name.
        cu_seqlens: Optional cumulative lengths for packed sequences.
        chunk_indices: Optional precomputed chunk indices keyed by block size.
        output_final_state: Whether to return the final convolution state.

    Returns:
        A contiguous output of shape ``[B, H, T, D / H]`` and the optional
        final state.
    """
    return CausalConv1dFunction.apply(
        x, weight, H, bias, residual, initial_state, activation, cu_seqlens, chunk_indices, output_final_state
    )


@torch.compiler.disable
def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: Optional[list[int]] = None,
    seq_idx: Optional[torch.Tensor] = None,
    cu_seqlens_int64: Optional[torch.Tensor] = None,
    chunk_indices: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Apply NPU causal-conv with a ``[B, T, H, K]`` public layout."""
    if x.ndim != 4:
        raise ValueError(f"causal-conv input must have shape [B, T, H, K], got {tuple(x.shape)}")
    batch_size, seq_len, num_heads, head_dim = x.shape
    if cu_seqlens_list is None:
        raise ValueError("NPU causal-conv requires cu_seq_lens_q_list")

    total_tokens = batch_size * seq_len
    forward_block_size, backward_block_size = get_npu_causal_conv1d_block_sizes(total_tokens)
    required_keys = {str(forward_block_size), str(backward_block_size)}
    if cu_seqlens_int64 is None or chunk_indices is None or not required_keys.issubset(chunk_indices):
        fallback_metadata = prepare_npu_metadata(
            cu_seqlens=cu_seqlens_list,
            device=x.device,
            total_tokens=total_tokens,
            block_sizes={forward_block_size, backward_block_size},
        )
        cu_seqlens_int64 = fallback_metadata.cu_seqlens_int64
        chunk_indices = fallback_metadata.chunk_indices
    assert cu_seqlens_int64 is not None
    assert chunk_indices is not None

    native, _ = causal_conv1d_triton_native(
        x=x.reshape(batch_size, seq_len, num_heads * head_dim),
        weight=weight,
        H=num_heads,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens_int64,
        chunk_indices=chunk_indices,
        output_final_state=False,
    )
    return native.transpose(1, 2)
