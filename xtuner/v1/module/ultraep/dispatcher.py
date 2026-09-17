"""Composite UltraEP dispatcher: control plane around an inner transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import Function
from typing_extensions import override

from xtuner.v1.module.dispatcher.base import ExpertWeightLayout, GenericDispatcher

from .runtime import UltraEPLayerRuntime, UltraEPModelRuntime


@dataclass
class _UltraEPLayerCallState:
    """Per layer × microbatch token. Decoder treats this as opaque."""

    virtual_layer_id: int
    weight_sync_event: object | None = None


class _UltraEPGradReduceStart(Function):
    """Start replica-gradient reduction after expert and dispatch backward."""

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, runtime: UltraEPLayerRuntime, virtual_layer_id: int):
        ctx.runtime = runtime
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        ctx.runtime.start_grad_reduce(ctx.virtual_layer_id)
        return grad_output, None, None


class _UltraEPGradReduceJoin(Function):
    """Join replica-gradient reduction after attention backward."""

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, runtime: UltraEPLayerRuntime, virtual_layer_id: int):
        ctx.runtime = runtime
        ctx.virtual_layer_id = virtual_layer_id
        return hidden_states

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        ctx.runtime.finish_grad_reduce(ctx.virtual_layer_id)
        return grad_output, None, None


class _UltraEPWeightRestoreStart(Function):
    """Launch mutable replica-weight restore at combine-backward entry.

    The identity is attached to the combined MoE output, after combine
    forward has completed but before the residual/post-MoE path. During
    backward its callback therefore runs before DeepEP combine backward and
    can overlap the restore copy/communication with that work.
    """

    @staticmethod
    def forward(
        ctx,
        combined_hidden_states: torch.Tensor,
        runtime: UltraEPLayerRuntime,
        virtual_layer_id: int,
    ) -> torch.Tensor:
        ctx.runtime = runtime
        ctx.virtual_layer_id = virtual_layer_id
        return combined_hidden_states

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        ctx.runtime.start_weight_restore(ctx.virtual_layer_id)
        return grad_output, None, None


class _UltraEPWeightRestoreJoin(Function):
    """Join a restore immediately before expert DGrad runs.

    Replica weights are reusable communication buffers, not model parameters.
    A later layer can overwrite them after this layer's forward. This identity
    node sits immediately after expert compute, so its backward waits for the
    restore launched at combine-backward entry before grouped-GEMM DGrad reads
    the mutable slots.
    """

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,
        runtime: UltraEPLayerRuntime,
        virtual_layer_id: int,
    ) -> torch.Tensor:
        ctx.runtime = runtime
        ctx.virtual_layer_id = virtual_layer_id
        return expert_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        ctx.runtime.finish_weight_restore(ctx.virtual_layer_id)
        return grad_output, None, None


def _call_state(pre_dispatched: Any) -> _UltraEPLayerCallState:
    state = pre_dispatched.get("_ultraep_call") if isinstance(pre_dispatched, dict) else None
    if not isinstance(state, _UltraEPLayerCallState):
        raise RuntimeError("UltraEP stages require layer_state from prepare_layer_inputs")
    return state


class UltraEPDispatcher(
    GenericDispatcher[Any, Any, Any, Any, Any, Any],
):
    """Layer-static UltraEP policy. Each call carries a private vid token.

    The inner transport is constructed lazily so unit tests can inject a fake
    without importing ``deep_ep``. Production first use builds the transport
    named by ``model_runtime.inner_dispatcher`` with ``E + R×ep`` physical
    experts.
    """

    def __init__(
        self,
        *,
        model_runtime: UltraEPModelRuntime,
        layer_runtime: UltraEPLayerRuntime,
        inner: GenericDispatcher | None = None,
    ) -> None:
        super().__init__(
            n_routed_experts=model_runtime.num_dispatch_experts,
            process_group=model_runtime.group,
            training_dtype=model_runtime.training_dtype,
            generate_dtype=model_runtime.generate_dtype,
        )
        self._model_runtime = model_runtime
        self._layer = layer_runtime
        self._inner = inner

    def _get_inner(self) -> GenericDispatcher:
        if self._inner is None:
            from xtuner.v1.module.dispatcher import build_dispatcher

            # Omit ep_runtime / layer_id / projections so this call builds
            # token transport and does not bind UltraEP again.
            # Inner-name support is gated at model construction.
            self._inner = build_dispatcher(  # type: ignore[assignment]
                self._model_runtime.inner_dispatcher,
                n_routed_experts=self._model_runtime.num_dispatch_experts,
                ep_group=self._model_runtime.group,
                training_dtype=self._model_runtime.training_dtype,
                generate_dtype=self._model_runtime.generate_dtype,
            )
        return self._inner

    @override
    def prepare_layer_inputs(
        self,
        layer_inputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[object | None]]:
        self._layer.validate_microbatch_capacity(len(layer_inputs))
        states: list[object | None] = []
        for _ in layer_inputs:
            virtual_layer_id = self._layer.allocate_virtual_layer_id()
            states.append(_UltraEPLayerCallState(virtual_layer_id=virtual_layer_id))
        return layer_inputs, states

    @override
    def prepare_microbatch_input(
        self,
        hidden_states: torch.Tensor,
        layer_state: object | None,
    ) -> torch.Tensor:
        if not isinstance(layer_state, _UltraEPLayerCallState):
            raise RuntimeError("UltraEP microbatch input requires layer_state from prepare_layer_inputs")
        # Match the original decoder schedule: create each Join immediately
        # before that microbatch's attention/Start nodes. Hoisting every Join
        # before the loop lets two Starts claim the shared staging pair.
        return _UltraEPGradReduceJoin.apply(hidden_states, self._layer, layer_state.virtual_layer_id)

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer_state: object | None = None,
        async_op: bool = False,
    ) -> Any:
        if not isinstance(layer_state, _UltraEPLayerCallState):
            raise RuntimeError("UltraEP dispatch_preprocess requires layer_state from prepare_layer_inputs")
        virtual_layer_id = layer_state.virtual_layer_id
        self._layer.update_placement(topk_ids, virtual_layer_id)
        layer_state.weight_sync_event = self._layer.sync_weights(virtual_layer_id, async_finish=True)
        physical_topk_ids = self._layer.reroute(topk_ids, virtual_layer_id)
        dispatch_hidden_states = _UltraEPGradReduceStart.apply(hidden_states, self._layer, virtual_layer_id)
        pre_dispatched = dict(
            self._get_inner().dispatch_preprocess(
                hidden_states=dispatch_hidden_states,
                topk_ids=physical_topk_ids,
                topk_weights=topk_weights,
                async_op=async_op,
            )
        )
        pre_dispatched["_ultraep_call"] = layer_state
        return pre_dispatched

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: Any,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> Any:
        return self._get_inner().dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=topk_weights,
            async_op=async_op,
            decoding=decoding,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: Any,
        dispatched: Any,
        async_op: bool = False,
    ) -> Any:
        post_dispatched = self._get_inner().dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            async_op=async_op,
        )
        call = _call_state(pre_dispatched)
        if call.weight_sync_event is not None:
            # Replica slots are first used by expert GEMMs. Deferring this
            # wait overlaps their async refresh with DeepEP dispatch work.
            with torch.profiler.record_function("UltraEP::forward_weight_sync_wait"):
                call.weight_sync_event.current_stream_wait()  # type: ignore[attr-defined]
        self._layer.bind_virtual_layer_slot(call.virtual_layer_id)
        post_dispatched = dict(post_dispatched)
        # Empty envelope: replica slots stay on the module after bind.
        post_dispatched["expert_weight_layout"] = ExpertWeightLayout()
        return post_dispatched

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: Any,
        dispatched: Any,
        post_dispatched: Any,
        async_op: bool = False,
        decoding: bool = False,
    ) -> Any:
        call = _call_state(pre_dispatched)
        hidden_states = _UltraEPWeightRestoreJoin.apply(hidden_states, self._layer, call.virtual_layer_id)
        return self._get_inner().combine_preprocess(
            hidden_states=hidden_states,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            async_op=async_op,
            decoding=decoding,
        )

    @override
    def combine(
        self,
        *,
        pre_dispatched: Any,
        dispatched: Any,
        post_dispatched: Any,
        pre_combined: Any,
        async_op: bool = False,
        decoding: bool = False,
    ) -> Any:
        return self._get_inner().combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            async_op=async_op,
            decoding=decoding,
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: Any,
        dispatched: Any,
        post_dispatched: Any,
        pre_combined: Any,
        combined: Any,
        async_op: bool = False,
    ) -> Any:
        post_combined = dict(
            self._get_inner().combine_postprocess(
                pre_dispatched=pre_dispatched,
                dispatched=dispatched,
                post_dispatched=post_dispatched,
                pre_combined=pre_combined,
                combined=combined,
                async_op=async_op,
            )
        )
        call = _call_state(pre_dispatched)
        post_combined["hidden_states"] = _UltraEPWeightRestoreStart.apply(
            post_combined["hidden_states"],
            self._layer,
            call.virtual_layer_id,
        )
        return post_combined


__all__ = [
    "UltraEPDispatcher",
    "_UltraEPGradReduceJoin",
    "_UltraEPGradReduceStart",
    "_UltraEPLayerCallState",
    "_UltraEPWeightRestoreJoin",
    "_UltraEPWeightRestoreStart",
]
