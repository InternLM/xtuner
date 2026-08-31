import torch
from torch import nn

from xtuner.v1.ops.moe.sonicmoe import SonicMoEOp
from xtuner.v1.ops.moe.token_rounding import build_token_rounding_metadata

from .config import SonicMoEBackendConfig


class SonicMoEBackend(nn.Module):
    """Adapter for the official ``sonicmoe`` general-routing API.

    Parameters remain owned by XTuner. This object only translates XTuner's
    routing result and expert-weight layout to SonicMoE's functional API, so
    state-dict and HuggingFace checkpoint keys remain unchanged.

    Args:
        config (SonicMoEBackendConfig): SonicMoE routing configuration.
    """

    def __init__(self, config: SonicMoEBackendConfig):
        super().__init__()
        self.config = config
        self._op = SonicMoEOp()

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_weights: torch.Tensor | None = None,
        fused_w1w3: torch.Tensor,
        fused_w2: torch.Tensor,
        fused_w1w3_bias: torch.Tensor | None = None,
        fused_w2_bias: torch.Tensor | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a complete routed expert MLP through SonicMoE.

        Args:
            hidden_states (torch.Tensor): Unpermuted token states with shape ``[T, H]``.
            topk_ids (torch.Tensor): Selected global expert ids with shape ``[T, K]``.
            topk_weights (torch.Tensor): Selected routing weights with shape ``[T, K]``.
            router_weights (torch.Tensor | None): Full router weights with shape ``[T, E]``. Required
                by token rounding and ignored by general routing.
            fused_w1w3 (torch.Tensor): XTuner concatenated gate/up weights ``[E, 2I, H]``.
            fused_w2 (torch.Tensor): XTuner down-projection weights ``[E, H, I]``.
            fused_w1w3_bias (torch.Tensor | None): Optional gate/up bias.
            fused_w2_bias (torch.Tensor | None): Optional down-projection bias.
            training (bool): Whether to enable SonicMoE's training path.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Combined expert output and executed token count per expert.
        """
        if not hidden_states.is_cuda:
            raise RuntimeError("The SonicMoE backend is CUDA-only.")
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"The initial SonicMoE integration supports bfloat16 only, got {hidden_states.dtype}.")
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must have shape [T, H], got {tuple(hidden_states.shape)}.")
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError(
                "topk_ids and topk_weights must have the same [T, K] shape, "
                f"got {tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}."
            )
        if topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("The token dimension of routing results must match hidden_states.")
        if fused_w1w3.ndim != 3 or fused_w2.ndim != 3:
            raise ValueError("SonicMoE expert weights must be logical [E, 2I, H] and [E, H, I] tensors.")

        tensors = (topk_ids, topk_weights, fused_w1w3, fused_w2)
        if any(tensor.device != hidden_states.device for tensor in tensors):
            raise ValueError("All SonicMoE inputs and expert weights must be on the same CUDA device.")
        if fused_w1w3.dtype != torch.bfloat16 or fused_w2.dtype != torch.bfloat16:
            raise TypeError("The initial SonicMoE integration requires bfloat16 expert weights.")

        num_experts, gate_up_features, hidden_size = fused_w1w3.shape
        if num_experts > 32768:
            raise ValueError(f"SonicMoE supports at most 32768 experts, got {num_experts}.")
        if gate_up_features % 2 != 0:
            raise ValueError("SwiGLU gate/up output dimension must be even.")
        intermediate_size = gate_up_features // 2
        if fused_w2.shape != (num_experts, hidden_size, intermediate_size):
            raise ValueError(
                "Incompatible SonicMoE expert weight shapes: "
                f"w1w3={tuple(fused_w1w3.shape)}, w2={tuple(fused_w2.shape)}."
            )
        if hidden_states.shape[1] != hidden_size:
            raise ValueError("hidden_states hidden dimension does not match the expert weights.")
        if (fused_w1w3_bias is None) != (fused_w2_bias is None):
            raise ValueError("SonicMoE requires both expert biases to be present or both to be absent.")
        if fused_w1w3_bias is not None and fused_w2_bias is not None:
            if fused_w1w3_bias.shape != (num_experts, gate_up_features):
                raise ValueError(
                    f"fused_w1w3_bias must have shape {(num_experts, gate_up_features)}, "
                    f"got {tuple(fused_w1w3_bias.shape)}."
                )
            if fused_w2_bias.shape != (num_experts, hidden_size):
                raise ValueError(
                    f"fused_w2_bias must have shape {(num_experts, hidden_size)}, "
                    f"got {tuple(fused_w2_bias.shape)}."
                )
            if fused_w1w3_bias.device != hidden_states.device or fused_w2_bias.device != hidden_states.device:
                raise ValueError("SonicMoE expert biases must be on the same CUDA device as hidden_states.")
            if fused_w1w3_bias.dtype != torch.bfloat16 or fused_w2_bias.dtype != torch.bfloat16:
                raise TypeError("The initial SonicMoE integration requires bfloat16 expert biases.")

        if hidden_states.shape[0] == 0:
            zero = hidden_states.sum() + fused_w1w3.sum() + fused_w2.sum()
            zero = zero + topk_weights.sum().to(hidden_states.dtype)
            if router_weights is not None:
                zero = zero + router_weights.sum().to(hidden_states.dtype)
            if fused_w1w3_bias is not None and fused_w2_bias is not None:
                zero = zero + fused_w1w3_bias.sum() + fused_w2_bias.sum()
            expert_frequency = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
            return hidden_states + zero * 0, expert_frequency

        if self.config.routing_mode == "token_rounding":
            if router_weights is None:
                raise ValueError("router_weights are required when SonicMoE token rounding is enabled.")
            if router_weights.device != hidden_states.device:
                raise ValueError("router_weights must be on the same CUDA device as hidden_states.")
            router_scores, token_indices, expert_indices = build_token_rounding_metadata(
                router_weights,
                topk_ids,
                num_experts=num_experts,
                rounding_quantum=self.config.rounding_quantum,
                rounding_mode=self.config.rounding_mode,
            )
        else:
            num_experts_per_token = topk_ids.shape[1]
            token_indices = torch.arange(
                hidden_states.shape[0], device=hidden_states.device, dtype=torch.int32
            ).repeat_interleave(num_experts_per_token)
            expert_indices = topk_ids.reshape(-1).to(dtype=torch.int32).contiguous()
            router_scores = topk_weights.reshape(-1).to(dtype=torch.float32).contiguous()

        if router_scores.numel() == 0:
            zero = fused_w1w3.sum() + fused_w2.sum() + router_scores.sum().to(hidden_states.dtype)
            if fused_w1w3_bias is not None and fused_w2_bias is not None:
                zero = zero + fused_w1w3_bias.sum() + fused_w2_bias.sum()
            output = hidden_states * 0 + zero * 0
            expert_frequency = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
            return output, expert_frequency

        # SonicMoE expects [2I, H, E] and [H, I, E]. XTuner stores the
        # checkpoint-compatible concatenated layout [E, 2I, H] / [E, H, I].
        # Permuted views preserve the original Parameters and their gradients.
        # Do not materialize contiguous copies here: SonicMoE requires the
        # stride order produced by permuting contiguous [E, O, I] parameters.
        sonic_w1 = fused_w1w3.permute(1, 2, 0)
        sonic_w2 = fused_w2.permute(1, 2, 0)
        output, expert_frequency = self._op(
            hidden_states,
            router_scores,
            token_indices,
            expert_indices,
            sonic_w1,
            fused_w1w3_bias,
            sonic_w2,
            fused_w2_bias,
            num_experts,
            training=training,
        )
        return output, expert_frequency
