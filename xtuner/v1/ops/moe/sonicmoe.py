import importlib
from importlib import metadata
from typing import Callable

import torch


SonicMoEForward = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class SonicMoEOp:
    """Low-level adapter for the official SonicMoE functional kernel.

    The optional dependency is resolved once during construction, outside any
    compiled forward graph. The call itself is intentionally a Dynamo graph
    boundary: SonicMoE already owns the optimized CUDA/CuTe kernels and custom
    backward, while XTuner can compile the surrounding decoder computation.
    """

    def __init__(self) -> None:
        self._forward, self.activation_type = self._resolve_official_api()

    @torch.compiler.disable
    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_scores: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        num_experts: int,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the official SonicMoE general-routing kernel.

        Args:
            hidden_states (torch.Tensor): Unpermuted token states.
            router_scores (torch.Tensor): Score for every routed assignment.
            token_indices (torch.Tensor): Token id for every assignment.
            expert_indices (torch.Tensor): Expert id for every assignment.
            w1 (torch.Tensor): SonicMoE up-projection weight view.
            b1 (torch.Tensor | None): Optional up-projection bias.
            w2 (torch.Tensor): SonicMoE down-projection weight view.
            b2 (torch.Tensor | None): Optional down-projection bias.
            num_experts (int): Number of routed experts.
            training (bool): Whether to enable the official training path.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Combined expert output and
            executed token count per expert.
        """
        return self._forward(
            hidden_states,
            router_scores,
            token_indices,
            expert_indices,
            w1,
            b1,
            w2,
            b2,
            num_experts,
            None,
            self.activation_type,
            not training,
            concat_layout=True,
        )

    @staticmethod
    def _resolve_official_api() -> tuple[SonicMoEForward, object]:
        try:
            functional = importlib.import_module("sonicmoe.functional")
            enums = importlib.import_module("sonicmoe.enums")
        except Exception as e:  # noqa: BLE001 - binary dependency import errors are not stable
            raise ImportError(
                "The SonicMoE expert backend requires the official 'sonic-moe' package. "
                "Install XTuner's sonicmoe optional dependency or run `pip install sonic-moe`."
            ) from e

        forward = getattr(functional, "moe_general_routing_inputs", None)
        activation_type = getattr(enums, "ActivationType", None)
        if not callable(forward) or activation_type is None or not hasattr(activation_type, "SWIGLU"):
            try:
                version = metadata.version("sonic-moe")
            except metadata.PackageNotFoundError:
                version = "unknown"
            raise ImportError(
                "The installed sonic-moe package does not expose the supported official API "
                f"(version={version!r}). Expected sonicmoe.functional.moe_general_routing_inputs "
                "and sonicmoe.enums.ActivationType.SWIGLU."
            )
        return forward, activation_type.SWIGLU
