from typing import Literal, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial
from torch.nn import functional as F

from transformers.activations import ACT2FN
from xtuner.v1.config.base_model import BaseAttnConfig, BaseRouterConfig, Float8Config, GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import MultiHeadAttention, MultiLatentAttention, RMSNorm, RouterResults
from xtuner.v1.module.dispatcher import PrefillingDispatchResult, build_dispatcher
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear
from xtuner.v1.module.router import GreedyRouter, NoAuxRouter
from xtuner.v1.utils import ForwardState
from xtuner.v1.utils.compile import maybe_compile

from ..linear.linear import build_linear


class MoEMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_shared_experts: int,
        moe_intermediate_size: int,
        hidden_act: str,
        mlp_bias: bool = False,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size * n_shared_experts
        self.gate_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.up_proj = build_linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.down_proj = build_linear(self.intermediate_size, self.hidden_size, bias=mlp_bias, float8_cfg=float8_cfg)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_config: BaseRouterConfig[GreedyRouter | NoAuxRouter],
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts

        self.gating_dim = hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.router = router_config.build(
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def forward(self, hidden_states: torch.Tensor) -> RouterResults:
        _, _, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local(grad_placements=(Partial("avg"),))
        else:
            weight = self.weight
        logits = F.linear(hidden_states.float(), weight.float(), None)

        return self.router(logits)


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        ep_mesh: DeviceMesh | None = None,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.num_routed_experts = n_routed_experts

        self.ep_mesh = ep_mesh
        # self.fused_w1 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        # self.fused_w3 = GroupedLinear(self.hidden_size, self.intermediate_size, self.num_routed_experts, ep_mesh)
        self.fused_w1w3 = build_grouped_linear(
            self.hidden_size,
            2 * self.intermediate_size,
            self.num_routed_experts,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )
        self.fused_w2 = build_grouped_linear(
            self.intermediate_size,
            self.hidden_size,
            self.num_routed_experts,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )

    @maybe_compile(fullgraph=True)
    def forward(self, x, tokens_per_expert, decoding):
        gate_up_out = self.fused_w1w3(x, tokens_per_expert, decoding)
        gate_out, up_out = gate_up_out.chunk(2, dim=-1)
        # up_out = self.fused_w1(x, tokens_per_expert, decoding)
        # gate_out = self.fused_w3(x, tokens_per_expert, decoding)
        gate_out = F.silu(gate_out)
        out = gate_out * up_out

        res = self.fused_w2(out, tokens_per_expert, decoding)
        return res


class MoEDecoderLayer(nn.Module):
    """MoE decoder layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        mlp_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        hidden_factor: float = 1.0,
        attention_config: BaseAttnConfig[MultiHeadAttention | MultiLatentAttention],
        generate_config: GenerateConfig | None = None,
        router_config: BaseRouterConfig[GreedyRouter | NoAuxRouter],
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        dispatcher: Literal["deepep", "all2all"] | None,
        ep_mesh: DeviceMesh | None = None,
    ):
        super().__init__()
        self.ep_mesh = ep_mesh
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.hidden_factor = hidden_factor

        self.self_attn = attention_config.build(
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            generate_config=generate_config,
            float8_cfg=float8_cfg,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.shared_experts: MoEMLP | None

        if n_shared_experts > 0:
            self.shared_experts = MoEMLP(
                hidden_size=hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
            )
        else:
            self.shared_experts = None

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_config=router_config,
        )
        self.experts = MoEBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            ep_mesh=ep_mesh,
            float8_cfg=float8_cfg,
        )
        # TODO: (yehaochen) Maybe should be replaced by build_dispatcher
        process_group = ep_mesh.get_group() if ep_mesh is not None else None
        self.dispatcher = build_dispatcher(
            dispatcher=dispatcher,
            n_routed_experts=n_routed_experts,
            ep_group=process_group,
            training_dtype="fp8" if float8_cfg is not None else "bf16",
            generate_dtype=generate_config.dtype if generate_config is not None else "bf16",
        )

    @maybe_compile(fullgraph=True)
    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, RouterResults]:
        residual, hidden_states, router_results = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
        )

        origin_shape = hidden_states.shape

        # reshape hidden_states to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=False,
        )  # type: ignore[call-overload]
        experts_out: torch.Tensor = self.experts(
            dispatched["hidden_states"],
            dispatched["tokens_per_experts"],
            decoding=False,
        )

        dispatched = cast(PrefillingDispatchResult, dispatched)
        combined = self.dispatcher.combine(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            decoding=False,
        )
        hidden_states = self.dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )
        hidden_states = hidden_states.view(*origin_shape)

        hidden_states = self._post_moe_forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return hidden_states, router_results

    @torch.no_grad  # type: ignore[call-arg]
    def prefilling(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[list[torch.Tensor]] | None = None,
    ):
        residual, hidden_states, router_results = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.PREFILLING,
            past_key_values=past_key_values,
        )
        origin_shape = hidden_states.shape

        # reshape hidden_states to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=False,
        )  # type: ignore[call-overload]
        experts_out: torch.Tensor = self.experts(
            dispatched["hidden_states"],
            dispatched["tokens_per_experts"],
            decoding=False,
        )
        combined = self.dispatcher.combine(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            decoding=False,
        )
        hidden_states = self.dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )
        hidden_states = hidden_states.view(*origin_shape)

        hidden_states = self._post_moe_forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return hidden_states, router_results

    @torch.no_grad  # type: ignore[call-arg]
    def decoding(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[list[torch.Tensor]] | None = None,
    ):
        residual, hidden_states, router_results = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.DECODING,
            past_key_values=past_key_values,
        )
        origin_shape = hidden_states.shape

        # reshape hidden_states to (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            decoding=True,
        )  # type: ignore[call-overload]
        experts_out: torch.Tensor = self.experts(
            dispatched["hidden_states"],
            dispatched["tokens_per_experts"],
            decoding=True,
        )
        combined = self.dispatcher.combine(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            decoding=True,
        )
        hidden_states = self.dispatcher.combine_post_process(
            pre_dispatched=pre_dispatched,
            dispatch_result=dispatched,
            combine_result=combined,
        )
        hidden_states = hidden_states.view(*origin_shape)

        hidden_states = self._post_moe_forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return hidden_states, router_results

    @maybe_compile(fullgraph=True)
    def _pre_moe_forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        state: ForwardState,
        past_key_values: list[list[torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RouterResults]:
        # NOTE: In order to allow `torch.compile` to compile the ops before and after attention as much as possible,
        # attention, post-layernorm and gate are implemented in one function
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if state == ForwardState.TRAINING:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
        elif state == ForwardState.PREFILLING:
            assert past_key_values is not None, "past_key_values should be provided in pre-filling state"
            hidden_states = self.self_attn.prefilling(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )
        elif state == ForwardState.DECODING:
            assert past_key_values is not None, "past_key_values should be provided in decoding state"
            hidden_states = self.self_attn.decoding(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
                past_key_values=past_key_values,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_results: RouterResults = self.gate(hidden_states)
        return residual, hidden_states, router_results

    @maybe_compile(fullgraph=True)
    def _post_moe_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        # This part can be fullgraph compiled
        if self.n_shared_experts > 0:
            assert self.shared_experts is not None, "Shared experts should be initialized when n_shared_experts > 0"
            shared_experts_out = self.shared_experts(hidden_states)
            hidden_states += shared_experts_out

        hidden_states = residual + hidden_states * self.hidden_factor
        return hidden_states

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )
