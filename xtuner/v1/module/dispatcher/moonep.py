"""MoonEP's model-scoped XTuner integration.

The backend import lives in this module and remains lazy: importing XTuner or
building another dispatcher must not require MoonEP.  The three stateful
classes added here are intentionally deep modules. ``MoonEPRuntime`` owns
model resources, ``MoonEPDispatcher`` owns one layer's behavior, and
``_MoonEPInvocationState`` is the lifecycle token for one dispatch/combine pair.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import TypedDict, override

from xtuner.v1.ops.moe.cuda.route_weight import route_weight_rows_backward
from xtuner.v1.utils import log_rank0

from .base import ExpertWeightLayout, GenericDispatcher, PostDispatchResult, ProjectionPair
from .fsdp_vmm_landing import (
    accumulate_fsdp_unsharded_expert_gradients,
    fsdp_current_unsharded_expert_parameters,
    install_fsdp_vmm_landing,
    uninstall_fsdp_vmm_landing,
)
from .moonep_workspace import _ExpertVMMWorkspace


_INTEGRATION_API_VERSION = 3
_MOONEP_IMPORT_ERROR: ImportError | None

try:
    import moonep as _moonep_backend
except ImportError as exc:
    _moonep_backend = None
    _MOONEP_IMPORT_ERROR = exc
else:
    _MOONEP_IMPORT_ERROR = None


def require_moonep_backend() -> Any:
    """Validate the optional MoonEP-mod package when MoonEP is selected."""
    if _moonep_backend is None:
        raise RuntimeError("dispatcher='moonep' requires the MoonEP-mod integration package") from _MOONEP_IMPORT_ERROR

    source = getattr(_moonep_backend, "__file__", "<unknown>")
    if getattr(_moonep_backend, "XTUNER_INTEGRATION_API_VERSION", None) != _INTEGRATION_API_VERSION:
        raise RuntimeError(
            f"incompatible MoonEP integration API; expected {_INTEGRATION_API_VERSION}; loaded module: {source}"
        )
    return _moonep_backend


class MoonEPRuntime:
    """Own the resources shared by all routed layers in one model/EP group."""

    def __init__(
        self,
        *,
        ep_group: dist.ProcessGroup,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        intra_layer_micro_batch: int,
        staging_reference: bool,
        num_sms: int = 64,
    ) -> None:
        require_moonep_backend()
        if intra_layer_micro_batch < 1:
            raise ValueError("intra_layer_micro_batch must be positive")

        # MoonEP keeps token counts device-resident. Triton already satisfies
        # that contract; grouped_gemm does so only with its CUTLASS backend.
        # Validate this once here instead of branching in every GMM call.
        from xtuner.v1.module.grouped_linear import moe_group_linear
        from xtuner.v1.ops.moe.cuda import cutlass_group_gemm

        if cutlass_group_gemm is not None and moe_group_linear.group_gemm is cutlass_group_gemm:
            from grouped_gemm import backend as grouped_gemm_backend

            if os.environ.get("GROUPED_GEMM_USE_CUTLASS") != "1" or not grouped_gemm_backend.use_cutlass:
                raise RuntimeError(
                    "MoonEP with grouped_gemm requires GROUPED_GEMM_USE_CUTLASS=1 before importing grouped_gemm"
                )

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._intra_layer_micro_batch = intra_layer_micro_batch
        self._staging_reference = staging_reference
        self._num_sms = num_sms

        self._buffer: Any | None = None
        self._workspace: _ExpertVMMWorkspace | None = None
        self._comm_stream: torch.cuda.Stream | None = None
        self._fsdp_params: tuple[Any, ...] = ()
        self._layers: list[tuple[str, tuple[nn.Module, nn.Module]]] = []
        self._fixed_tokens_per_rank: int | None = None
        self._closed = False

    def build_dispatcher(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> MoonEPDispatcher:
        """Register one physical routed layer in FSDP execution order."""
        if any(registered_fqn == layer_fqn for registered_fqn, _ in self._layers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        layer_id = len(self._layers)
        self._layers.append((layer_fqn, projections))
        return MoonEPDispatcher(runtime=self, layer_id=layer_id)

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        """Validate the build-time FSDP policy without retaining its config."""
        if fsdp_config.param_dtype is not torch.bfloat16 or fsdp_config.reduce_dtype is not torch.bfloat16:
            raise ValueError("MoonEP requires BF16 FSDP param and reduce dtypes")
        if fsdp_config.cpu_offload:
            raise ValueError("MoonEP VMM weights cannot use FSDP CPU offload")
        if not fsdp_config.requires_grad:
            raise ValueError("MoonEP v1 requires trainable FSDP parameters")
        if not fsdp_config.reshard_after_forward:
            raise ValueError("MoonEP requires reshard_after_forward=True")

    def install_after_fsdp(self, *, fsdp_root: nn.Module) -> None:
        """Allocate execution resources after native FSDP has been
        installed."""
        if self._workspace is not None:
            raise RuntimeError("MoonEP FSDP resources are already installed")
        if not self._layers:
            raise TypeError("MoonEP requires at least one physical routed-expert layer")
        if self._staging_reference:
            log_rank0.warning(
                "moonep_staging_reference=True copies complete BF16 home expert "
                "weights after every FSDP AllGather; it is a numerical reference, "
                "not the production performance path."
            )
        workspace = _ExpertVMMWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            gradient_slots=self._intra_layer_micro_batch,
        )
        # Keep MoonEP collectives in FSDP's device-side launch order. A
        # separate high-priority stream forms an orthogonal progress wave with
        # NCCL and stalls at MoonEP's rank barriers under a full model.
        comm_stream = torch.cuda.current_stream()
        if not self._staging_reference:
            try:
                self._fsdp_params = install_fsdp_vmm_landing(
                    fsdp_root=fsdp_root,
                    targets=tuple(
                        (
                            layer_fqn,
                            projections,
                            workspace.landing(layer_id % 2),
                        )
                        for layer_id, (layer_fqn, projections) in enumerate(self._layers)
                    ),
                )
            except Exception:
                workspace.destroy()
                raise
        self._workspace = workspace
        self._comm_stream = comm_stream

    def _validate_tokens_per_rank(self, tokens_per_rank: int) -> None:
        if self._closed:
            raise RuntimeError("MoonEP runtime was closed")
        if self._fixed_tokens_per_rank is None:
            self._fixed_tokens_per_rank = tokens_per_rank
        elif tokens_per_rank != self._fixed_tokens_per_rank:
            raise RuntimeError(f"MoonEP fixed S changed: {self._fixed_tokens_per_rank} -> {tokens_per_rank}")

    def _buffer_for(self, tokens_per_rank: int) -> Any:
        self._validate_tokens_per_rank(tokens_per_rank)
        if self._workspace is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before forward")
        if self._buffer is None:
            assert _moonep_backend is not None
            self._buffer = _moonep_backend.Buffer(
                S=tokens_per_rank,
                H=self._hidden_size,
                K=self._top_k,
                E=self._num_experts,
                num_ep_ranks=self._ep_group.size(),
                group=self._ep_group,
                explicitly_destroy=True,
                num_sms=self._num_sms,
            )
        return self._buffer

    def _enqueue(self, operation, *, inputs: tuple[torch.Tensor | None, ...] = ()):
        """Run one MoonEP transaction on XTuner's stream and return its
        event."""
        stream = self._comm_stream
        if stream is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before execution")
        caller_stream = torch.cuda.current_stream()
        stream.wait_event(caller_stream.record_event())
        for tensor in inputs:
            if tensor is not None:
                tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            result = operation()
            done = stream.record_event()
        return result, done

    def close(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._closed:
            return
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
            self._comm_stream = None
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        if self._fsdp_params:
            uninstall_fsdp_vmm_landing(self._fsdp_params)
            self._fsdp_params = ()
        if self._workspace is not None:
            self._workspace.destroy()
            self._workspace = None
        self._layers.clear()
        self._closed = True


class MoonEPPreDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    tokens_per_expert: torch.Tensor


class MoonEPDispatchResult(TypedDict):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    cu_seqlens: torch.Tensor
    _moonep_invocation: _MoonEPInvocationState


class MoonEPPostDispatchResult(PostDispatchResult): ...


class MoonEPPreCombineResult(TypedDict):
    hidden_states: torch.Tensor
    _moonep_invocation: _MoonEPInvocationState


class MoonEPCombineResult(TypedDict):
    hidden_states: torch.Tensor


class MoonEPPostCombineResult(TypedDict):
    hidden_states: torch.Tensor


@dataclass
class _MoonEPInvocationState:
    """State owned exclusively by one forward/backward invocation."""

    grad_slot: int
    plan: Any | None = None
    dispatch_done: Any | None = None
    weights_ready: Any | None = None
    combine_done: Any | None = None
    local_weights: ProjectionPair | None = None
    gradient_targets: ProjectionPair | None = None
    fallback_gradient_targets: ProjectionPair | None = None
    home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
    local_gradient_parameters: list[nn.Parameter | None] = field(default_factory=lambda: [None, None])


class _DispatchAutograd(torch.autograd.Function):
    """Concrete MoonEP dispatch forward paired with combine backward."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        dispatcher: MoonEPDispatcher,
        state: _MoonEPInvocationState,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.dispatcher = dispatcher
        ctx.state = state
        hidden_nvsh, route_weights_nvs, cu_seqlens = dispatcher._dispatch_forward(
            state,
            source_hidden,
            topk_ids,
            tokens_per_expert,
            source_route_weights,
            async_op=async_op,
        )
        ctx.mark_non_differentiable(cu_seqlens)
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    @staticmethod
    def backward(
        ctx: Any,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
        grad_cu_seqlens: None,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None, None]:
        del grad_cu_seqlens
        grad_hidden, grad_route_weights = ctx.dispatcher._dispatch_backward(
            ctx.state,
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None, None


class _CombineAutograd(torch.autograd.Function):
    """Fused route-scaled combine forward paired with plan-reuse dispatch."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        dispatcher: MoonEPDispatcher,
        state: _MoonEPInvocationState,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.dispatcher = dispatcher
        ctx.state = state
        ctx.save_for_backward(expert_output, route_weights)
        return dispatcher._combine_forward(state, expert_output, route_weights, async_op=async_op)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        grad_weighted, replay_done = ctx.dispatcher._combine_backward(ctx.state, grad_output)
        expert_output, route_weights = ctx.saved_tensors
        grad_expert, grad_route_weights = route_weight_rows_backward(
            grad_weighted,
            expert_output,
            route_weights,
        )
        # The next autograd node immediately reads duplicated weights.
        replay_done.wait()
        return grad_expert, grad_route_weights, None, None, None


class MoonEPDispatcher(
    GenericDispatcher[
        MoonEPPreDispatchResult,
        MoonEPDispatchResult,
        MoonEPPostDispatchResult,
        MoonEPPreCombineResult,
        MoonEPCombineResult,
        MoonEPPostCombineResult,
    ]
):
    """Adapt MoonEP plans/VMM state to XTuner's six-stage dispatcher API."""

    def __init__(
        self,
        *,
        runtime: MoonEPRuntime,
        layer_id: int,
    ) -> None:
        super().__init__(
            n_routed_experts=runtime._num_experts,
            process_group=runtime._ep_group,
        )
        self._runtime = runtime
        self._layer_id = layer_id
        self._next_gradient_slot = 0

    def _current_home_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return current FSDP leaves, staging values only in reference
        mode."""
        runtime = self._runtime
        layer_fqn, projections = runtime._layers[self._layer_id]
        if not runtime._staging_reference:
            return fsdp_current_unsharded_expert_parameters(projections)

        workspace = runtime._workspace
        assert workspace is not None
        landings = workspace.landing(self._layer_id % 2)
        parameters: list[nn.Parameter] = []
        for linear, landing in zip(projections, landings, strict=True):
            weight = cast(torch.Tensor, linear.weight)
            if not isinstance(weight, nn.Parameter):
                raise RuntimeError(f"{layer_fqn} staging expected an unsharded expert Parameter")
            source = weight.to_local() if isinstance(weight, DTensor) else weight
            if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
                raise RuntimeError(f"{layer_fqn} staging expected an unsharded BF16 expert weight")
            with torch.no_grad():
                landing.copy_(source.view_as(landing))
            parameters.append(weight)
        return parameters[0], parameters[1]

    def _dispatch_forward(
        self,
        state: _MoonEPInvocationState,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        *,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        runtime = self._runtime
        buffer = runtime._buffer_for(source_hidden.shape[0])
        workspace = runtime._workspace
        assert workspace is not None

        def dispatch_and_prefetch():
            # The staging copy precedes dispatch's device barrier, so every
            # rank publishes this FSDP generation before remote prefetch.
            state.home_parameters = self._current_home_parameters()
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
                source_hidden,
                route_weights_sk=source_route_weights,
                topk_experts_sk=topk_ids,
                tokens_per_expert=tokens_per_expert,
                async_finish=False,
                zero_copy=False,
            )
            assert route_weights_nvs is not None and cu_seqlens is not None
            state.plan = plan
            state.dispatch_done = torch.cuda.current_stream().record_event()
            state.local_weights, state.gradient_targets = workspace.prefetch_weights(
                buffer=buffer,
                plan=plan,
                generation=self._layer_id % 2,
                grad_slot=state.grad_slot,
            )
            return hidden_nvsh, route_weights_nvs, cu_seqlens

        with torch.profiler.record_function("MoonEP::dispatch_forward"):
            result, state.weights_ready = runtime._enqueue(
                dispatch_and_prefetch,
                inputs=(source_hidden, source_route_weights, topk_ids, tokens_per_expert),
            )
        assert state.dispatch_done is not None
        if not async_op:
            state.dispatch_done.wait()
        return result

    def _prepare_experts(
        self,
        state: _MoonEPInvocationState,
        dispatched: MoonEPDispatchResult,
    ) -> MoonEPPostDispatchResult:
        workspace = self._runtime._workspace
        assert workspace is not None
        assert state.weights_ready is not None
        assert state.local_weights is not None and state.gradient_targets is not None

        with torch.profiler.record_function("MoonEP::prepare_experts"):
            # Event.wait adds a dependency to the caller stream; it never
            # waits on the host. Counts and expert GEMMs can now consume both
            # dispatch outputs and the prefetched BF16 weight aliases.
            state.weights_ready.wait()
            local_counts = workspace.local_token_counts(dispatched["cu_seqlens"])
            covered_rows = local_counts.sum()
            row_is_covered = (
                torch.arange(
                    dispatched["hidden_states"].shape[0],
                    device=dispatched["hidden_states"].device,
                )
                < covered_rows
            )
            hidden_states = dispatched["hidden_states"] * row_is_covered.unsqueeze(-1)
            local_counts = torch.cat(
                (
                    local_counts[:-1],
                    local_counts[-1:] + dispatched["hidden_states"].shape[0] - covered_rows,
                )
            )

        local_weights = state.local_weights
        gradient_targets = state.gradient_targets
        state.local_weights = None
        state.gradient_targets = None
        differentiable_weights: ProjectionPair
        trainable_wgrad_outs: ProjectionPair | None = None
        if torch.is_grad_enabled():
            # Leaf Parameters let AccumulateGrad hand the preallocated VMM
            # WGrad targets to grouped GEMM without a full-gradient copy.
            differentiable_weights = (
                nn.Parameter(local_weights[0]),
                nn.Parameter(local_weights[1]),
            )
            for projection, parameter in enumerate(differentiable_weights):
                parameter.register_post_accumulate_grad_hook(
                    lambda completed, projection=projection: self._record_parameter_gradient(
                        state, projection, completed
                    )
                )
            trainable_wgrad_outs = gradient_targets
        else:
            differentiable_weights = local_weights
        return MoonEPPostDispatchResult(
            hidden_states=hidden_states,
            tokens_per_expert=local_counts,
            expert_weight_layout=ExpertWeightLayout(
                trainable_weights=differentiable_weights,
                trainable_wgrad_outs=trainable_wgrad_outs,
            ),
        )

    def _dispatch_backward(
        self,
        state: _MoonEPInvocationState,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
    ) -> ProjectionPair:
        runtime = self._runtime
        buffer = runtime._buffer
        assert state.plan is not None and buffer is not None
        grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
        grad_route_weights_nvs = grad_route_weights_nvs.contiguous()

        def combine_gradients():
            grad_hidden, grad_route_weights, no_event = buffer.combine(
                plan=state.plan,
                hidden_nvsh=grad_hidden_nvsh,
                route_weights_nvs=grad_route_weights_nvs,
                async_finish=False,
                zero_copy=False,
            )
            assert grad_route_weights is not None and no_event is None
            return grad_hidden, grad_route_weights

        with torch.profiler.record_function("MoonEP::dispatch_backward"):
            result, done = runtime._enqueue(
                combine_gradients,
                inputs=(grad_hidden_nvsh, grad_route_weights_nvs),
            )
            done.wait()
        return result

    def _combine_forward(
        self,
        state: _MoonEPInvocationState,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        async_op: bool,
    ) -> torch.Tensor:
        runtime = self._runtime
        buffer = runtime._buffer
        assert state.plan is not None and buffer is not None

        def combine_output():
            output, gathered_weights, no_event = buffer.combine(
                plan=state.plan,
                hidden_nvsh=expert_output,
                hidden_scales_nvs=route_weights,
                route_weights_nvs=None,
                async_finish=False,
                zero_copy=False,
            )
            assert gathered_weights is None and no_event is None
            return output

        with torch.profiler.record_function("MoonEP::combine_forward"):
            output, state.combine_done = runtime._enqueue(
                combine_output,
                inputs=(expert_output, route_weights),
            )
        if not async_op:
            state.combine_done.wait()
        return output

    def _combine_backward(
        self,
        state: _MoonEPInvocationState,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        runtime = self._runtime
        buffer = runtime._buffer
        workspace = runtime._workspace
        assert state.plan is not None and buffer is not None and workspace is not None
        grad_output = grad_output.contiguous()

        def dispatch_gradient_and_prefetch():
            # FSDP pre-backward has restored this generation. Stage it before
            # dispatch's barrier, then replay remote weights on the same stream.
            replay_home_parameters = self._current_home_parameters()
            if state.home_parameters is not None and any(
                replay is not forward
                for replay, forward in zip(replay_home_parameters, state.home_parameters, strict=True)
            ):
                raise RuntimeError("MoonEP backward observed a different FSDP unsharded Parameter")
            grad_weighted, no_weights, no_cu, reused_plan = buffer.dispatch(
                grad_output,
                plan=state.plan,
                async_finish=False,
                zero_copy=False,
            )
            assert no_weights is None and no_cu is None and reused_plan is state.plan
            gradient_dispatch_done = torch.cuda.current_stream().record_event()
            _, state.fallback_gradient_targets = workspace.prefetch_weights(
                buffer=buffer,
                plan=state.plan,
                generation=self._layer_id % 2,
                grad_slot=state.grad_slot,
            )
            return grad_weighted, gradient_dispatch_done

        with torch.profiler.record_function("MoonEP::combine_backward"):
            (grad_weighted, gradient_dispatch_done), replay_done = runtime._enqueue(
                dispatch_gradient_and_prefetch,
                inputs=(grad_output,),
            )
            # Route-scale backward overlaps weight replay but cannot read the
            # dispatched gradient until this device event is satisfied.
            gradient_dispatch_done.wait()
        return grad_weighted, replay_done

    def _complete_weight_gradients(
        self,
        state: _MoonEPInvocationState,
        local_grads: ProjectionPair,
    ) -> ProjectionPair:
        runtime = self._runtime
        workspace = runtime._workspace
        assert state.plan is not None and runtime._buffer is not None and workspace is not None
        assert state.fallback_gradient_targets is not None
        reduction_grads: list[torch.Tensor] = []
        for source, target in zip(local_grads, state.fallback_gradient_targets, strict=True):
            # Extra Tensor references may force AccumulateGrad to copy. The
            # original direct-output target already contains the full dW.
            reduction_grads.append(source if source.data_ptr() == target.data_ptr() else target)

        with torch.profiler.record_function("MoonEP::gradient_handoff"):
            home_grads, done = runtime._enqueue(
                lambda: workspace.complete_gradients(
                    buffer=runtime._buffer,
                    plan=state.plan,
                    local_grads=tuple(reduction_grads),
                    grad_slot=state.grad_slot,
                ),
                inputs=tuple(reduction_grads),
            )
            # FSDP post-backward is downstream on the caller stream, so this
            # device wait guarantees ReduceScatter sees the returned BF16 dW.
            done.wait()
        state.fallback_gradient_targets = None
        return home_grads

    def _record_parameter_gradient(
        self,
        state: _MoonEPInvocationState,
        projection: int,
        parameter: nn.Parameter,
    ) -> None:
        if parameter.grad is None:
            raise RuntimeError("MoonEP local expert Parameter completed without a gradient")
        state.local_gradient_parameters[projection] = parameter
        if any(item is None for item in state.local_gradient_parameters):
            return

        local_parameters = cast(list[nn.Parameter], state.local_gradient_parameters)
        local_grads = cast(ProjectionPair, tuple(parameter.grad for parameter in local_parameters))
        assert state.home_parameters is not None and all(gradient is not None for gradient in local_grads)
        home_grads = self._complete_weight_gradients(state, local_grads)
        accumulate_fsdp_unsharded_expert_gradients(state.home_parameters, home_grads)
        for local_parameter in local_parameters:
            local_parameter.grad = None
        state.home_parameters = None
        state.local_gradient_parameters = [None, None]

    @staticmethod
    def _finish_forward_only(state: _MoonEPInvocationState) -> None:
        # Python ownership can be dropped after device waits are enqueued;
        # route-dependent state is never queried or synchronized on the host.
        state.plan = None
        state.dispatch_done = None
        state.weights_ready = None
        state.combine_done = None
        state.local_weights = None
        state.gradient_targets = None
        state.fallback_gradient_targets = None
        state.home_parameters = None
        state.local_gradient_parameters = [None, None]

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        self._runtime._validate_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            tokens_per_expert=tokens_per_expert.to(dtype=torch.int32).contiguous(),
        )

    @override
    def dispatch(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        topk_weights: torch.Tensor,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPDispatchResult:
        if decoding:
            raise NotImplementedError("MoonEP fixed-S training dispatch does not implement decoding")
        grad_slot = self._next_gradient_slot
        self._next_gradient_slot = (grad_slot + 1) % self._runtime._intra_layer_micro_batch
        state = _MoonEPInvocationState(grad_slot=grad_slot)
        hidden_nvsh, weights_nvs, cu_seqlens = _DispatchAutograd.apply(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            pre_dispatched["tokens_per_expert"],
            topk_weights.to(dtype=torch.float32).contiguous(),
            self,
            state,
            async_op,
        )
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=weights_nvs,
            cu_seqlens=cu_seqlens,
            _moonep_invocation=state,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPPostDispatchResult:
        del pre_dispatched, async_op, decoding
        return self._prepare_experts(dispatched["_moonep_invocation"], dispatched)

    @override
    def combine_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPPreCombineResult:
        del pre_dispatched, post_dispatched, async_op, decoding
        return MoonEPPreCombineResult(
            hidden_states=hidden_states,
            _moonep_invocation=dispatched["_moonep_invocation"],
        )

    @override
    def combine(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        async_op: bool = False,
        decoding: bool = False,
    ) -> MoonEPCombineResult:
        del pre_dispatched, post_dispatched, decoding
        state = pre_combined["_moonep_invocation"]
        return MoonEPCombineResult(
            hidden_states=_CombineAutograd.apply(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
                self,
                state,
                async_op,
            )
        )

    @override
    def combine_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        post_dispatched: MoonEPPostDispatchResult,
        pre_combined: MoonEPPreCombineResult,
        combined: MoonEPCombineResult,
        async_op: bool = False,
    ) -> MoonEPPostCombineResult:
        del pre_dispatched, dispatched, post_dispatched
        state = pre_combined["_moonep_invocation"]
        if async_op:
            assert state.combine_done is not None
            state.combine_done.wait()
        result = MoonEPPostCombineResult(hidden_states=combined["hidden_states"])
        if not torch.is_grad_enabled():
            self._finish_forward_only(state)
        return result


__all__ = [
    "MoonEPDispatcher",
    "MoonEPRuntime",
    "MoonEPPreDispatchResult",
    "MoonEPDispatchResult",
    "MoonEPPostDispatchResult",
    "MoonEPPreCombineResult",
    "MoonEPCombineResult",
    "MoonEPPostCombineResult",
    "require_moonep_backend",
]
