from typing import Literal, TypeAlias

import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from transformers.activations import ACT2FN
from xtuner.v1.config.base_model import BaseAttnConfig, BaseRouterConfig, GenerateConfig
from xtuner.v1.config.float8 import Float8Config
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import MultiHeadAttention, MultiLatentAttention, RMSNorm, RouterResults
from xtuner.v1.module.dispatcher import (
    CombineResult,
    DispatchResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    build_dispatcher,
)
from xtuner.v1.module.dispatcher.base import PostCombineResult
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear
from xtuner.v1.module.router import GreedyRouter, NoAuxRouter
from xtuner.v1.ops import swiglu
from xtuner.v1.utils import ForwardState
from xtuner.v1.utils.compile import maybe_compile

from ..linear.linear import build_linear


RouterLogits: TypeAlias = torch.Tensor
HiddenStates: TypeAlias = torch.Tensor


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
            weight = self.weight.to_local()
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
        out = swiglu(gate_up_out, split_dim=-1)
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
        self.layer_idx = layer_idx

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
        *hidden_states: torch.Tensor,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[HiddenStates, RouterResults] | tuple[torch.Tensor, ...]:
        """Forward pass of the MoE decoder layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            seq_ctx (SequenceContext): Sequence context.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Position embeddings.
            past_key_values (list[list[torch.Tensor]], optional): Past key values for pre-filling or decoding.

        Returns:
            tuple[torch.Tensor, RouterResults]: Output hidden states and router results.
        """
        if len(hidden_states) == 1:
            assert isinstance(seq_ctx, SequenceContext), (
                f"seq_ctx should be a SequenceContext instance but got {seq_ctx}"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a tuple of two tensors (position_ids, position_embeds)"
            )
            return self._forward(
                hidden_states=hidden_states[0],
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
            )
        else:
            assert isinstance(seq_ctx, list) and len(seq_ctx) == len(hidden_states), (
                "seq_ctx should be a list of SequenceContext instances with the same length as hidden_states"
            )
            assert isinstance(position_embeddings, list) and len(position_embeddings) == len(hidden_states), (
                "position_embeddings should be a list of tuples with the same length as hidden_states"
            )

            return self._micro_batch_forward(
                hidden_states_list=list(hidden_states),
                seq_ctx_list=seq_ctx,
                position_embeddings_list=position_embeddings,
            )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[HiddenStates, RouterLogits]:
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
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=router_results["topk_weights"],
            decoding=False,
        )  # type: ignore[call-overload]
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            decoding=False,
        )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            decoding=False,
        )

        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        hidden_states = post_combined["hidden_states"]
        hidden_states = hidden_states.view(*origin_shape)

        hidden_states = self._post_moe_forward(
            hidden_states=hidden_states,
            residual=residual,
        )
        return hidden_states, router_results["logits"]

    def _micro_batch_forward(
        self,
        hidden_states_list: list[torch.Tensor],
        seq_ctx_list: list[SequenceContext],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, ...]:  # (HiddenStates, HiddenStates, RouterLogits, RouterLogits)
        origin_shape = hidden_states_list[0].shape
        assert all(hidden_states.shape == origin_shape for hidden_states in hidden_states_list), (
            "All hidden states should have the same shape"
        )
        residual_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []

        pre_dispatched_list: list[PreDispatchResult] = []
        dispatched_list: list[DispatchResult] = []

        # Attention + gate + pre-dispatch
        for (
            hidden_states,
            seq_ctx,
            position_embeddings,
        ) in zip(
            hidden_states_list,
            seq_ctx_list,
            position_embeddings_list,
        ):
            residual, hidden_states, router_results = self._pre_moe_forward(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                state=ForwardState.TRAINING,
            )
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            pre_dispatched = self.dispatcher.dispatch_preprocess(
                hidden_states=hidden_states,
                topk_ids=router_results["topk_ids"],
                async_op=True,
            )
            pre_dispatched_list.append(pre_dispatched)
            residual_list.append(residual)
            router_results_list.append(router_results)

        post_dispatched_list: list[PostDispatchResult] = []
        experts_out_list: list[torch.Tensor] = []
        pre_combined_list: list[PreCombineResult] = []
        combined_list: list[CombineResult] = []

        # dispatch + experts + pre-combine
        for router_results, pre_dispatched in zip(
            router_results_list,
            pre_dispatched_list,
        ):
            dispatched = self.dispatcher.dispatch(
                pre_dispatched=pre_dispatched,
                topk_weights=router_results["topk_weights"],
                async_op=True,
            )
            # wait for pre-dispatch event
            post_dispatched = self.dispatcher.dispatch_postprocess(
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                async_op=True,
            )
            experts_out = self.experts(
                post_dispatched["hidden_states"],
                post_dispatched["tokens_per_expert"],
                decoding=False,
            )

            pre_combined = self.dispatcher.combine_preprocess(
                hidden_states=experts_out,
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                async_op=True,
            )

            post_dispatched_list.append(post_dispatched)
            experts_out_list.append(experts_out)
            dispatched_list.append(dispatched)
            pre_combined_list.append(pre_combined)

        post_combined_list: list[PostCombineResult] = []

        for pre_combined, pre_dispatched, dispatched, post_dispatched in zip(
            pre_combined_list,
            pre_dispatched_list,
            dispatched_list,
            post_dispatched_list,
        ):
            combined = self.dispatcher.combine(
                pre_combined=pre_combined,
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                async_op=True,
            )
            combined_list.append(combined)

        hidden_states_out_list: list[torch.Tensor] = []
        for combine_result, residual in zip(post_combined_list, residual_list):
            hidden_states = self._post_moe_forward(
                hidden_states=combine_result["hidden_states"],
                residual=residual,
            )
            hidden_states_out_list.append(hidden_states)

        router_logits = [router_results["logits"] for router_results in router_results_list]
        return tuple(hidden_states_out_list + router_logits)

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
        # hidden_states =

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
            return (hidden_states + shared_experts_out) * self.hidden_factor + residual
        else:
            return hidden_states * self.hidden_factor + residual

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.build_kv_cache(
            max_batch_size=max_batch_size,
            max_length=max_length,
            block_size=block_size,
        )


class _BackwardSync(Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        previous_backward_event: torch.cuda.Event | None = None,
        finished_backward_event: torch.cuda.Event | None = None,
        name=None,
    ) -> torch.Tensor:
        ctx.previous_backward_event = previous_backward_event
        ctx.finished_backward_event = finished_backward_event
        ctx.name = name
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        current_stream = torch.cuda.current_stream()

        # if ctx.name == "pre_dispatched":
        #     torch.cuda.synchronize()
        #
        # if ctx.name == "dispatched":
        #     torch.cuda.synchronize()
        #
        # if ctx.name == "pre_combined":
        #     torch.cuda.synchronize()
        #
        if ctx.previous_backward_event is not None:
            current_stream.wait_event(ctx.previous_backward_event)
        if ctx.finished_backward_event is not None:
            current_stream.record_event(ctx.finished_backward_event)

        return grad_output, None, None, None


backward_sync = _BackwardSync.apply


# class _DebugBackward(Function):
#     @staticmethod
#     def forward(
#         ctx,
#         input_tensor: torch.Tensor,
#         name: str
#     ) -> torch.Tensor:
#         ctx.name = name
#         return input_tensor
#
#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor):
#         print(ctx.name)
#         return grad_output, None
