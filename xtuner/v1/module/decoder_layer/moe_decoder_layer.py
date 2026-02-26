from functools import partial
from typing import Literal, Protocol, TypeAlias, cast

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.autograd.function import Function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.config.generate import GenerateConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8 import Float8Config
from xtuner.v1.module import (
    AttnOutputs,
    GreedyRouterConfig,
    MHAConfig,
    MLAConfig,
    NoAuxRouterConfig,
    RMSNorm,
    RouterResults,
)
from xtuner.v1.module.dispatcher import (
    CombineResult,
    DispatchResult,
    PostDispatchResult,
    PreCombineResult,
    PreDispatchResult,
    build_dispatcher,
)
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.ops.act_fn import get_act_fn
from xtuner.v1.utils import ForwardState

from ..linear import build_linear


RouterLogits: TypeAlias = torch.Tensor
RouterWeights: TypeAlias = torch.Tensor
HiddenStates: TypeAlias = torch.Tensor


class MoEActFnProtocol(Protocol):
    def __call__(self, fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor: ...


class MoEActFnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    act_type: Literal["clipped_swiglu", "swiglu"] = "swiglu"

    clip_alpha: float | None = None
    clip_limit: float | None = None

    def build(self) -> MoEActFnProtocol:
        act_fn = get_act_fn(self.act_type)

        if self.act_type == "clipped_swiglu":
            act_fn = partial(act_fn, alpha=self.clip_alpha, limit=self.clip_limit)
        return act_fn


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
        self.act_fn = get_act_fn(hidden_act)

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
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        gate_bias: bool = False,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts

        self.gating_dim = hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

        self.router = router_config.build(
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        self.gate_bias = gate_bias
        if self.gate_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(
        self, hidden_states: torch.Tensor, rollout_routed_experts: torch.Tensor | None = None
    ) -> RouterResults:
        _, _, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)

        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        bias = None
        if self.gate_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            bias = bias.float()

        logits = F.linear(hidden_states.float(), weight.float(), bias)

        return self.router(logits, rollout_routed_experts)


class MoEBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
        float8_cfg: Float8Config | None = None,
        moe_act_fn_cfg: MoEActFnConfig,
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
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )
        self.fused_w2 = build_grouped_linear(
            self.intermediate_size,
            self.hidden_size,
            self.num_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=self.ep_mesh,
            float8_cfg=float8_cfg,
        )
        self.moe_act = moe_act_fn_cfg.build()

    def forward(self, x, tokens_per_expert, decoding):
        gate_up_out = self.fused_w1w3(x, tokens_per_expert, decoding)
        out = self.moe_act(gate_up_out, split_dim=-1)
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
        gate_bias: bool = False,
        moe_bias: bool = False,
        hidden_act: str,
        rms_norm_eps: float = 1e-6,
        rms_norm_type: Literal['default', 'zero_centered'] = 'default',
        num_experts_per_tok: int,
        n_routed_experts: int,
        n_shared_experts: int,
        with_shared_expert_gate: bool = False,
        hidden_factor: float = 1.0,
        attention_config: MHAConfig | MLAConfig,
        rope_scaling_cfg: RopeScalingConfig | None = None,
        layer_type: Literal["full_attention", "sliding_attention"] | None = None,
        generate_config: GenerateConfig | None = None,
        router_config: GreedyRouterConfig | NoAuxRouterConfig,
        moe_act_fn_cfg: MoEActFnConfig,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        dispatcher: Literal["deepep", "all2all", "agrs"] | None,
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
            rope_scaling_cfg=rope_scaling_cfg,
            layer_type=layer_type,
            float8_cfg=float8_cfg,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.shared_experts: MoEMLP | None
        self.layer_idx = layer_idx
        
        self.with_shared_expert_gate = with_shared_expert_gate
        if n_shared_experts > 0:
            self.shared_experts = MoEMLP(
                hidden_size=hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
            )
            if with_shared_expert_gate:
                self.shared_expert_gate = build_linear(hidden_size, 1, bias=False, float8_cfg=float8_cfg)
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

        self.gate = MoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_config=router_config,
            gate_bias=gate_bias,
        )
        self.experts = MoEBlock(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            moe_bias=moe_bias,
            ep_mesh=ep_mesh,
            float8_cfg=float8_cfg,
            moe_act_fn_cfg=moe_act_fn_cfg,
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
    ) -> tuple[HiddenStates, RouterLogits, RouterWeights]:
        residual, hidden_states, router_results = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
        )

        origin_shape = hidden_states.shape

        # reshape hidden_states to (batch_size * seq_len, hidden_size)
        # ProberList.before_dispatch(
        #     self.layer_idx, hidden_states, router_results["topk_ids"], router_results["topk_weights"]
        # )
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states.view(-1, hidden_states.shape[-1]),
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
        # ProberList.after_dispatch(
        #     self.layer_idx,
        #     post_dispatched["hidden_states"],
        #     post_dispatched["tokens_per_expert"],
        #     post_dispatched.get("row_ids_map"),  # type: ignore[arg-type]
        #     dispatched["topk_weights"],
        # )
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            decoding=False,
        )
        # ProberList.before_combine(
        #     self.layer_idx,
        #     experts_out,
        #     post_dispatched.get("row_ids_map"),  # type: ignore[arg-type]
        #     dispatched["topk_weights"],
        # )
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
        combined_hidden_states = post_combined["hidden_states"]
        combined_hidden_states = combined_hidden_states.view(*origin_shape)
        # ProberList.after_combine(self.layer_idx, combined_hidden_states)

        if self.n_shared_experts > 0:
            shared_experts_out = self._shared_experts_forward(hidden_states=hidden_states)
        else:
            shared_experts_out = None

        hidden_states = self._post_moe_forward(
            combined_hidden_states=combined_hidden_states,
            residual=residual,
            shared_experts_out=shared_experts_out,
        )
        return hidden_states, router_results["logits"], router_results["router_weights"]

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
        intra_layer_micro_batch = len(hidden_states_list)
        residual_list: list[torch.Tensor] = []
        router_results_list: list[RouterResults] = []

        pre_dispatched_list: list[PreDispatchResult] = []
        dispatched_list: list[DispatchResult] = []
        pre_moe_forward_out_list: list[torch.Tensor] = []

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
            pre_moe_forward_out_list.append(hidden_states)
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

        shared_experts_out_list: list[torch.Tensor | None]

        if self.n_shared_experts > 0:
            shared_experts_out_list = []
            for pre_moe_forward_out in pre_moe_forward_out_list:
                shared_experts_out = self._shared_experts_forward(
                    hidden_states=pre_moe_forward_out,
                )
                shared_experts_out_list.append(shared_experts_out)
        else:
            shared_experts_out_list = [None] * intra_layer_micro_batch

        hidden_states_out_list: list[torch.Tensor] = []
        for i in range(intra_layer_micro_batch):
            post_combined = self.dispatcher.combine_postprocess(
                pre_dispatched=pre_dispatched_list[i],
                dispatched=dispatched_list[i],
                post_dispatched=post_dispatched_list[i],
                pre_combined=pre_combined_list[i],
                combined=combined_list[i],
                async_op=True,
            )
            hidden_states = self._post_moe_forward(
                # hidden_states=pre_moe_forward_out_list[i],
                combined_hidden_states=post_combined["hidden_states"].view(*pre_moe_forward_out_list[i].shape),
                residual=residual_list[i],
                shared_experts_out=shared_experts_out_list[i],
            )
            hidden_states_out_list.append(hidden_states)

        router_logits = [router_results["logits"] for router_results in router_results_list]
        router_weights = [router_results["router_weights"] for router_results in router_results_list]
        return tuple(hidden_states_out_list + router_logits + router_weights)

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
            attn_outputs: AttnOutputs = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
            hidden_states = attn_outputs["projected_output"]
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

        if seq_ctx.rollout_routed_experts is not None:
            rollout_routed_experts = seq_ctx.rollout_routed_experts[:, self.layer_idx, :]  # seq_l, expert
        else:
            rollout_routed_experts = None
        router_results: RouterResults = self.gate(hidden_states, rollout_routed_experts)
        return residual, hidden_states, router_results

    def _shared_experts_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.shared_experts is not None, "Shared experts should be initialized when n_shared_experts > 0"
        shared_experts_out = self.shared_experts(hidden_states)

        if self.with_shared_expert_gate:
            shared_experts_out = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_experts_out
            
        return shared_experts_out

    def _post_moe_forward(
        self,
        combined_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        shared_experts_out: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.n_shared_experts > 0:
            shared_experts_out = cast(torch.Tensor, shared_experts_out)
            combined_hidden_states = combined_hidden_states + shared_experts_out
        return combined_hidden_states * self.hidden_factor + residual

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
