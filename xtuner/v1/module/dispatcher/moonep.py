"""MoonEP's model-scoped XTuner integration.

The backend import remains lazy so unrelated dispatchers do not require
MoonEP. ``MoonEPModelRuntime`` owns model resources and ``MoonEPDispatcher``
owns one routed layer's static policy. One dispatch/combine call is a pure
data ``_MoonEPLayerCallState`` record advanced by the module-level transaction
functions (``dispatch_forward``, ``prepare_experts``, ``combine_forward``,
``combine_backward``, gradient completion, and the layer-Join handoff). The
private VMM workspace remains the deep module for physical expert layout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _MoonEPLayer:
    """One physical routed layer's identity, stored once.

    ``ordinal`` is the FSDP execution-order position; the home generation is
    issued from it by the workspace, so it is not stored here.
    """

    fqn: str
    projections: tuple[nn.Module, nn.Module]
    ordinal: int


@dataclass(frozen=True)
class _MoonEPResources:
    """The only non-``None`` state after ``install_after_fsdp``.

    The call state borrows this record instead of the runtime; ``buffer_for``
    and ``enqueue`` are its two methods. ``_buffer_box`` is a one-slot mutable
    box holding ``(Buffer, Fixed-S)`` because the activation Buffer is still
    built lazily on the first forward once Fixed-S is known.
    """

    workspace: _ExpertVMMWorkspace
    landing: ExpertLandingAdapter
    comm_stream: torch.cuda.Stream
    ep_group: dist.ProcessGroup
    num_experts: int
    experts_per_rank: int
    top_k: int
    hidden_size: int
    gradient_slots: int
    num_sms: int
    _buffer_box: list

    def buffer_for(self, tokens_per_rank: int) -> Any:
        """Dispatch entry: build the Fixed-S Buffer once, then check S."""
        if not self._buffer_box:
            assert _moonep_backend is not None
            buffer = _moonep_backend.Buffer(
                S=tokens_per_rank,
                H=self.hidden_size,
                K=self.top_k,
                E=self.num_experts,
                num_ep_ranks=self.ep_group.size(),
                group=self.ep_group,
                explicitly_destroy=True,
                num_sms=self.num_sms,
            )
            self._buffer_box.append((buffer, tokens_per_rank))
        buffer, fixed_s = self._buffer_box[0]
        if tokens_per_rank != fixed_s:
            raise RuntimeError(f"MoonEP fixed S changed: {fixed_s} -> {tokens_per_rank}")
        return buffer

    @property
    def buffer(self) -> Any:
        """Combine/backward entry: the Buffer this call's dispatch built."""
        if not self._buffer_box:
            raise RuntimeError("MoonEP Buffer must be created by dispatch first")
        return self._buffer_box[0][0]

    def expect_tokens_per_rank(self, tokens_per_rank: int) -> None:
        """Reject a changed Fixed-S before the first Dispatcher/VMM op."""
        if self._buffer_box and tokens_per_rank != self._buffer_box[0][1]:
            raise RuntimeError(f"MoonEP fixed S changed: {self._buffer_box[0][1]} -> {tokens_per_rank}")

    def enqueue(
        self,
        operation: Callable[[], Any],
        *,
        inputs: tuple[torch.Tensor | None, ...] = (),
    ) -> tuple[Any, torch.cuda.Event]:
        """Run one MoonEP transaction on XTuner's stream and return ``(result,
        done event)``."""
        caller_stream = torch.cuda.current_stream()
        self.comm_stream.wait_event(caller_stream.record_event())
        for tensor in inputs:
            if tensor is not None:
                tensor.record_stream(self.comm_stream)
        with torch.cuda.stream(self.comm_stream):
            result = operation()
            done = self.comm_stream.record_event()
        return result, done

    def home_generation(self, layer: _MoonEPLayer) -> int:
        """The workspace owns the physical chunks and issues the generation."""
        return self.workspace.generation_for(layer.ordinal)


class ExpertLandingAdapter:
    """Make one generation's home weights ready and hand back its Parameters.

    ``DirectVMMLanding`` is the production path (FSDP unpacks straight into VMM
    home rows); ``StagingReferenceLanding`` is the numerical-reference path
    (FSDP lands normal storage, then copies into the VMM home rows). Both are
    the two Adapters of one Seam, so the runtime and transaction functions
    only call ``prepare`` and never branch on a ``staging_reference`` flag.
    """

    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        raise NotImplementedError

    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        raise NotImplementedError

    def uninstall(self) -> None:
        raise NotImplementedError


class DirectVMMLanding(ExpertLandingAdapter):
    """FSDP's final per-parameter unpack lands directly in the VMM home
    rows."""

    def __init__(self) -> None:
        self._fsdp_params: tuple[Any, ...] = ()

    @override
    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        self._fsdp_params = install_fsdp_vmm_landing(
            fsdp_root=fsdp_root,
            targets=tuple(
                (layer.fqn, layer.projections, workspace.landing(workspace.generation_for(layer.ordinal)))
                for layer in layers
            ),
        )

    @override
    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        # "Ready" is one check: FSDP has already materialized the weight in
        # the VMM landing, so there is no copy.
        del generation
        return fsdp_current_unsharded_expert_parameters(layer.projections)

    @override
    def uninstall(self) -> None:
        if self._fsdp_params:
            uninstall_fsdp_vmm_landing(self._fsdp_params)
            self._fsdp_params = ()


class StagingReferenceLanding(ExpertLandingAdapter):
    """FSDP lands normal storage first, then this copies it into the VMM home
    rows."""

    def __init__(self) -> None:
        log_rank0.warning(
            "moonep_staging_reference=True copies complete BF16 home expert "
            "weights after every FSDP AllGather; it is a numerical reference, "
            "not the production performance path."
        )
        self._workspace: _ExpertVMMWorkspace | None = None

    @override
    def install(
        self, *, fsdp_root: nn.Module, workspace: _ExpertVMMWorkspace, layers: tuple[_MoonEPLayer, ...]
    ) -> None:
        # No FSDP binding is installed: that is what distinguishes the two
        # Adapters. The copy happens in ``prepare``.
        del fsdp_root, layers
        self._workspace = workspace

    @override
    def prepare(self, *, layer: _MoonEPLayer, generation: int) -> tuple[nn.Parameter, nn.Parameter]:
        assert self._workspace is not None
        parameters: list[nn.Parameter] = []
        for linear, landing in zip(layer.projections, self._workspace.landing(generation), strict=True):
            weight = cast(torch.Tensor, linear.weight)
            if not isinstance(weight, nn.Parameter):
                raise RuntimeError(f"{layer.fqn} staging expected an unsharded expert Parameter")
            source = weight.to_local() if isinstance(weight, DTensor) else weight
            if source.dtype is not torch.bfloat16 or source.numel() != landing.numel():
                raise RuntimeError(f"{layer.fqn} staging expected an unsharded BF16 expert weight")
            with torch.no_grad():
                landing.copy_(source.view_as(landing))
            parameters.append(weight)
        return parameters[0], parameters[1]

    @override
    def uninstall(self) -> None:
        self._workspace = None


def build_landing_adapter(staging_reference: bool) -> ExpertLandingAdapter:
    if staging_reference:
        return StagingReferenceLanding()
    return DirectVMMLanding()


class MoonEPModelRuntime:
    """Own the lifecycle and ordered layer registry for one model/EP group.

    Construction takes no CUDA resources. ``build_dispatcher`` registers one
    physical routed layer per call in construction order; ``install_after_fsdp``
    allocates the workspace, cross-checks registration order against FSDP
    execution order, and installs the landing Adapter.
    """

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
        # Config-level capability validation (backend version, EP geometry,
        # dtype, grouped-GEMM backend, ...) lives in ``moonep_capability`` and
        # runs at meta model build. Keep only the optional-backend gate here so
        # direct construction still fails fast.
        require_moonep_backend()

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._num_sms = num_sms
        self._gradient_slots = intra_layer_micro_batch
        self._landing = build_landing_adapter(staging_reference)

        # Physical routed layers in registration (construction) order.
        self._layers: list[_MoonEPLayer] = []
        self._resources: _MoonEPResources | None = None
        self._closed = False

    def bind_layer(
        self,
        *,
        layer_fqn: str,
        projections: tuple[nn.Module, nn.Module],
    ) -> MoonEPDispatcher:
        """Register one physical routed layer and return its Dispatcher."""
        if any(layer.fqn == layer_fqn for layer in self._layers):
            raise ValueError(f"duplicate MoonEP routed layer: {layer_fqn}")
        layer = _MoonEPLayer(fqn=layer_fqn, projections=projections, ordinal=len(self._layers))
        self._layers.append(layer)
        return MoonEPDispatcher(runtime=self, layer=layer)

    def validate_before_fsdp(self, fsdp_config: Any) -> None:
        # The build-time FSDP policy checks moved to ``moonep_capability``.
        # This boundary stays because the Protocol needs it and a future
        # backend may have its own FSDP preconditions.
        del fsdp_config

    def install_after_fsdp(self, *, fsdp_root: nn.Module, execution_order: list[str]) -> None:
        """Allocate execution resources after native FSDP has been
        installed."""
        if self._resources is not None:
            raise RuntimeError("MoonEP FSDP resources are already installed")
        if not self._layers:
            raise TypeError("MoonEP requires at least one physical routed-expert layer")

        # Registration order vs FSDP execution order, checked once in the only
        # place that can see both. ``moe.py`` hands over the ordered list
        # rather than an adapter reading FSDP private structure.
        registered = [layer.fqn for layer in self._layers]
        if registered != execution_order:
            raise RuntimeError(
                f"MoonEP registration order does not match FSDP execution order: {registered} != {execution_order}"
            )

        workspace = _ExpertVMMWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            gradient_slots=self._gradient_slots,
            home_generations=2,
        )
        # Keep MoonEP collectives in FSDP's device-side launch order. A
        # separate high-priority stream forms an orthogonal progress wave with
        # NCCL and stalls at MoonEP's rank barriers under a full model.
        comm_stream = torch.cuda.current_stream()
        try:
            self._landing.install(fsdp_root=fsdp_root, workspace=workspace, layers=tuple(self._layers))
        except Exception:
            workspace.destroy()
            raise
        self._resources = _MoonEPResources(
            workspace=workspace,
            landing=self._landing,
            comm_stream=comm_stream,
            ep_group=self._ep_group,
            num_experts=self._num_experts,
            experts_per_rank=self._num_experts // self._ep_group.size(),
            top_k=self._top_k,
            hidden_size=self._hidden_size,
            gradient_slots=self._gradient_slots,
            num_sms=self._num_sms,
            _buffer_box=[],
        )

    @property
    def resources(self) -> _MoonEPResources:
        """One place decides "is MoonEP installed"."""
        if self._closed:
            raise RuntimeError("MoonEP runtime was closed")
        if self._resources is None:
            raise RuntimeError("MoonEP FSDP resources must be installed before forward")
        return self._resources

    def close(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._closed:
            return
        if self._resources is not None:
            resources = self._resources
            resources.comm_stream.synchronize()
            for buffer, _ in resources._buffer_box:
                buffer.destroy()
            resources.landing.uninstall()
            resources.workspace.destroy()
            self._resources = None
        self._layers.clear()
        self._closed = True


# Dispatcher shape legend:
#   S: source tokens on this EP rank, K: routed experts per token,
#   NvS: MoonEP's padded VM-group rows, E: global experts, B=E/R: home
#   experts per EP rank, H: hidden size.


class MoonEPPreDispatchResult(TypedDict):
    """Stage 1: device-resident router-space inputs normalized for MoonEP."""

    hidden_states: torch.Tensor  # [S, H], BF16 source-token order.
    topk_ids: torch.Tensor  # [S, K], contiguous int32 global expert IDs.
    tokens_per_expert: torch.Tensor  # [E], contiguous int32 source histogram.
    # The call state is opaque control state for the remaining five stages;
    # it never crosses into the compiled tensor-only expert block.
    _moonep_call: _MoonEPLayerCallState


class MoonEPDispatchResult(TypedDict):
    """Stage 2: global dispatch outputs.

    The call state travels on ``MoonEPPreDispatchResult`` only; every later
    stage already receives ``pre_dispatched`` and reads ``_moonep_call`` there.
    """

    hidden_states: torch.Tensor  # [NvS, H], BF16 physical VM-group order.
    topk_weights: torch.Tensor  # [NvS], FP32 weights in the same row order.
    # [E+B], int32 padded group ends; stays on device and is non-differentiable.
    cu_seqlens: torch.Tensor


class MoonEPPostDispatchResult(PostDispatchResult):
    """Stage 3: tensor-only local ``[2B]`` expert-compute bundle.

    ``hidden_states`` is ``[NvS, H]``; ``tokens_per_expert`` is device int32
    ``[2B]`` for home then duplicate groups; ``expert_weight_layout`` holds the
    projection-paired ``[2B, O_p, I_p]`` call-local weight aliases.
    """


class MoonEPPreCombineResult(TypedDict):
    """Stage 4: expert outputs before route scaling and reverse transport."""

    hidden_states: torch.Tensor  # [NvS, H], physical VM-group order.


class MoonEPCombineResult(TypedDict):
    """Stage 5: fused route-scaled output restored to source-token order."""

    hidden_states: torch.Tensor  # [S, H].


class MoonEPPostCombineResult(TypedDict):
    """Stage 6: final tensor bundle returned through the generic interface."""

    hidden_states: torch.Tensor  # [S, H].


@dataclass
class _MoonEPLayerGradients:
    """One FSDP call's initialization state, never shared across replay calls.

    Storage belongs to the workspace. Only the first backward producer clears it; forward/checkpoint replay must not
    touch a different call's live H.
    """

    initialized: bool = False


@dataclass(eq=False)
class _MoonEPLayerCallState:
    """One routed-layer call's pure lifecycle state for the transaction
    functions.

    The record borrows the installed ``_MoonEPResources`` and the ``_MoonEPLayer``
    identity, plus the call-local plan/event/weight/gradient handles that the
    module-level transaction functions read and advance. It owns no behavior
    and never references ``MoonEPDispatcher``. Identity, not field equality,
    distinguishes two calls, so instances stay hashable by ``id``.
    """

    resources: _MoonEPResources
    layer: _MoonEPLayer
    generation: int
    grad_slot: int
    layer_gradients: _MoonEPLayerGradients

    # One MoonEP communication plan and its device-side dependency chain. Each
    # event is recorded once its named producer has been enqueued.
    plan: Any | None = None
    dispatch_done: Any | None = None
    weights_ready: Any | None = None
    combine_done: Any | None = None

    # Borrowed local [2B, O_p, I_p] weight aliases for this call.
    local_weights: ProjectionPair | None = None

    # Current FSDP unsharded home Parameters [B, O_p, I_p] receive the returned
    # BF16 home gradients after both local projections complete.
    home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
    # Completed home views and the event covering the pair reduction.
    gradient_completion: tuple[ProjectionPair, Any] | None = None


# --- Transaction functions --------------------------------------------------
#
# Each function covers one complete device-side sequence for a single call and
# advances ``_MoonEPLayerCallState`` in place. The autograd Functions and the
# Dispatcher are the only callers; the pair backward must reuse the forward
# plan and gradient slot recorded on the call state.


def finish_combine(state: _MoonEPLayerCallState, combined: torch.Tensor, *, async_op: bool) -> torch.Tensor:
    """Establish the final device dependency for an async combine."""
    if async_op:
        assert state.combine_done is not None
        state.combine_done.wait()
    return combined


def prepare_experts(state: _MoonEPLayerCallState, dispatched: MoonEPDispatchResult) -> MoonEPPostDispatchResult:
    """Wait at the first weight consumer and expose the tensor-only layout."""
    resources = state.resources
    assert state.weights_ready is not None

    with torch.profiler.record_function("MoonEP::prepare_experts"):
        # This inserts a device dependency; it never waits on the host.
        state.weights_ready.wait()
        # Both halves of the ``[E+B] -> [2B]`` contract now live in the
        # workspace: the caller cannot receive counts it must still fix up.
        hidden_states, local_counts = resources.workspace.local_compute_view(
            hidden_nvsh=dispatched["hidden_states"],
            cu_seqlens=dispatched["cu_seqlens"],
        )

    local_weights = state.local_weights
    state.local_weights = None
    assert local_weights is not None
    # Join activation and both weight edges so staging precedes upstream
    # activation backward; a weight-only hook cannot establish that dependency.
    # The bridge already makes the aliases require grad, so grouped GEMM
    # returns their dW without a leaf ``nn.Parameter`` wrapper.
    hidden_states, w13, w2 = _MoonEPExpertGradBridge.apply(hidden_states, local_weights[0], local_weights[1], state)
    return MoonEPPostDispatchResult(
        hidden_states=hidden_states,
        tokens_per_expert=local_counts,
        expert_weight_layout=ExpertWeightLayout(
            trainable_weights=(w13, w2),
        ),
    )


def dispatch_forward(
    state: _MoonEPLayerCallState,
    source_hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    source_route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch on a fresh plan and start both projection weight prefetches."""
    resources = state.resources
    buffer = resources.buffer_for(source_hidden.shape[0])

    def dispatch_and_prefetch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Landing.prepare makes this generation's home weights ready: a check
        # for the direct Adapter, the staging copy for the reference Adapter.
        # Its call position (before dispatch's device barrier) is part of this
        # sequence, not an implicit precondition of a mode branch.
        state.home_parameters = resources.landing.prepare(layer=state.layer, generation=state.generation)
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
        # Only local weights are returned now; the gradient slot views are
        # created inside ``return_expert_gradients`` when they are needed.
        state.local_weights = resources.workspace.prefetch_weights(
            buffer=buffer,
            plan=plan,
            generation=state.generation,
        )
        return hidden_nvsh, route_weights_nvs, cu_seqlens

    with torch.profiler.record_function("MoonEP::dispatch_forward"):
        result, state.weights_ready = resources.enqueue(
            dispatch_and_prefetch,
            inputs=(source_hidden, source_route_weights, topk_ids, tokens_per_expert),
        )
    assert state.dispatch_done is not None
    if not async_op:
        state.dispatch_done.wait()
    return result


def dispatch_backward(
    state: _MoonEPLayerCallState,
    grad_hidden_nvsh: torch.Tensor,
    grad_route_weights_nvs: torch.Tensor,
) -> ProjectionPair:
    """Combine activation and route-weight gradients on the saved plan."""
    resources = state.resources
    buffer = resources.buffer
    grad_hidden_nvsh = grad_hidden_nvsh.contiguous()
    grad_route_weights_nvs = grad_route_weights_nvs.contiguous()

    def combine_gradients() -> tuple[torch.Tensor, torch.Tensor]:
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
        result, done = resources.enqueue(
            combine_gradients,
            inputs=(grad_hidden_nvsh, grad_route_weights_nvs),
        )
        done.wait()
    return result


def combine_forward(
    state: _MoonEPLayerCallState,
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    async_op: bool,
) -> torch.Tensor:
    """Fuse route scaling into the combine boundary on the saved plan."""
    resources = state.resources
    buffer = resources.buffer

    def combine_output() -> torch.Tensor:
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
        output, state.combine_done = resources.enqueue(
            combine_output,
            inputs=(expert_output, route_weights),
        )
    if not async_op:
        state.combine_done.wait()
    return output


def combine_backward(state: _MoonEPLayerCallState, grad_output: torch.Tensor) -> tuple[torch.Tensor, Any]:
    """Replay duplicated weights on the saved plan and return weighted grad."""
    resources = state.resources
    buffer = resources.buffer
    grad_output = grad_output.contiguous()

    def dispatch_gradient_and_prefetch() -> tuple[torch.Tensor, torch.cuda.Event]:
        # FSDP pre-backward has restored this generation; same Adapter call,
        # same sequence.
        replay_home_parameters = resources.landing.prepare(layer=state.layer, generation=state.generation)
        if state.home_parameters is None:
            raise RuntimeError("MoonEP backward has no forward home Parameters")
        if any(
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
        resources.workspace.prefetch_weights(buffer=buffer, plan=state.plan, generation=state.generation)
        return grad_weighted, gradient_dispatch_done

    with torch.profiler.record_function("MoonEP::combine_backward"):
        (grad_weighted, gradient_dispatch_done), replay_done = resources.enqueue(
            dispatch_gradient_and_prefetch,
            inputs=(grad_output,),
        )
        # Route-scale backward overlaps weight replay but cannot read the
        # dispatched gradient before this device event.
        gradient_dispatch_done.wait()
    return grad_weighted, replay_done


def start_gradient_completion(state: _MoonEPLayerCallState, gradients: ProjectionPair) -> None:
    """Hand the allocation-return dW to the workspace for the home return."""
    if state.gradient_completion is not None:
        raise RuntimeError("MoonEP gradient completion was started twice")
    resources = state.resources

    with torch.profiler.record_function("MoonEP::gradient_handoff"):
        # The workspace owns the ``B`` split, the home-prefix zero-or-add, the
        # duplicate-suffix copy, and the EP exact-sum reduction. ``initialize``
        # is the call-local flag the Dispatcher owns (ADR-0027).
        home_grads, done = resources.enqueue(
            lambda: resources.workspace.return_expert_gradients(
                buffer=resources.buffer,
                plan=state.plan,
                gradients=gradients,
                grad_slot=state.grad_slot,
                initialize=not state.layer_gradients.initialized,
            ),
            inputs=gradients,
        )
    state.layer_gradients.initialized = True
    state.gradient_completion = (home_grads, done)


def finish_gradient_completion(
    state: _MoonEPLayerCallState,
) -> tuple[tuple[nn.Parameter, nn.Parameter], ProjectionPair]:
    """Wait on the device event; the layer Join owns the single handoff."""
    completion = state.gradient_completion
    if completion is None:
        raise RuntimeError("MoonEP gradient completion was not started")
    home_grads, done = completion
    done.wait()

    if state.home_parameters is None:
        raise RuntimeError("MoonEP gradient completion has no home Parameters")
    home_parameters = state.home_parameters

    state.home_parameters = None
    state.gradient_completion = None
    return home_parameters, home_grads


class _MoonEPExpertGradBridge(torch.autograd.Function):
    """Consume both dWs before releasing the expert activation gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        call_state: _MoonEPLayerCallState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        return hidden_states, w13, w2

    @staticmethod
    def backward(
        ctx: Any, grad_hidden: torch.Tensor, dw13: torch.Tensor, dw2: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        start_gradient_completion(cast(_MoonEPLayerCallState, ctx.call_state), (dw13, dw2))
        # dW is now owned by MoonEP; do not also accumulate it on anchor leaves.
        return grad_hidden, None, None, None


class _MoonEPLayerGradJoin(torch.autograd.Function):
    """Join every microbatch before the native FSDP input backward hook."""

    @staticmethod
    def forward(
        ctx: Any,
        call_states: tuple[_MoonEPLayerCallState, ...],
        *layer_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.call_states = call_states
        return layer_inputs

    @staticmethod
    def backward(ctx: Any, *grad_inputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        # Event.wait inserts a dependency into the current CUDA stream; it does
        # not block the Python host or poll event readiness. Every call state's
        # returned views alias the same home H, so completing all of them and
        # publishing the last pair once is correct; publishing each would count
        # the whole sum repeatedly. Native FSDP consumes it once via copy-in
        # before upstream calls may reuse H; RS itself may remain asynchronous.
        home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
        home_grads: ProjectionPair | None = None
        for call_state in ctx.call_states:
            home_parameters, home_grads = finish_gradient_completion(call_state)
        assert home_parameters is not None and home_grads is not None
        accumulate_fsdp_unsharded_expert_gradients(home_parameters, home_grads)
        return (None, *grad_inputs)


class _DispatchAutograd(torch.autograd.Function):
    """Bridge the dispatch/combine pair into PyTorch autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.call_state = call_state
        hidden_nvsh, route_weights_nvs, cu_seqlens = dispatch_forward(
            call_state,
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
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, None, None]:
        del grad_cu_seqlens
        grad_hidden, grad_route_weights = dispatch_backward(
            cast(_MoonEPLayerCallState, ctx.call_state),
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None


class _CombineAutograd(torch.autograd.Function):
    """Bridge fused combine and saved-plan dispatch into autograd."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        call_state: _MoonEPLayerCallState,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.call_state = call_state
        ctx.save_for_backward(expert_output, route_weights)
        return combine_forward(call_state, expert_output, route_weights, async_op=async_op)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_weighted, replay_done = combine_backward(cast(_MoonEPLayerCallState, ctx.call_state), grad_output)
        expert_output, route_weights = ctx.saved_tensors
        grad_expert, grad_route_weights = route_weight_rows_backward(
            grad_weighted,
            expert_output,
            route_weights,
        )
        # The next autograd node immediately reads duplicated weights.
        replay_done.wait()
        return grad_expert, grad_route_weights, None, None


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
    """Adapt one routed layer to XTuner's six-stage dispatcher interface.

    This class owns only layer-static policy. Every dispatch creates a fresh
    ``_MoonEPLayerCallState`` that the module-level transaction functions
    advance with the call's plan, event, weight, and gradient state.
    """

    def __init__(
        self,
        *,
        runtime: MoonEPModelRuntime,
        layer: _MoonEPLayer,
    ) -> None:
        super().__init__(
            n_routed_experts=runtime._num_experts,
            process_group=runtime._ep_group,
        )
        self._runtime = runtime
        self._layer = layer
        self._next_gradient_slot = 0

    def _new_call_state(self, layer_gradients: _MoonEPLayerGradients) -> _MoonEPLayerCallState:
        """Allocate the next call-local slot and call-state token."""
        resources = self._runtime.resources
        grad_slot = self._next_gradient_slot
        self._next_gradient_slot = (grad_slot + 1) % resources.gradient_slots
        return _MoonEPLayerCallState(
            resources=resources,
            layer=self._layer,
            generation=resources.home_generation(self._layer),
            grad_slot=grad_slot,
            layer_gradients=layer_gradients,
        )

    @override
    def prepare_layer_inputs(
        self,
        layer_inputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[object | None]]:
        """Create one call-local Join and one plan/duplicate slot per
        branch."""
        gradients = _MoonEPLayerGradients()
        call_states = tuple(self._new_call_state(gradients) for _ in layer_inputs)
        if len(call_states) > self._runtime.resources.gradient_slots:
            raise ValueError("MoonEP layer width exceeds the gradient slot ring")
        # No-grad original forwards build no backward node and never clear H.
        return list(_MoonEPLayerGradJoin.apply(call_states, *layer_inputs)), list(call_states)

    @override
    def dispatch_preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        layer_state: object | None = None,
        async_op: bool = False,
    ) -> MoonEPPreDispatchResult:
        del topk_weights, async_op
        if layer_state is None:
            raise RuntimeError("MoonEP dispatch_preprocess requires layer_state from prepare_layer_inputs")
        if not isinstance(layer_state, _MoonEPLayerCallState):
            raise TypeError("MoonEP layer_state must be a _MoonEPLayerCallState")
        layer_state.resources.expect_tokens_per_rank(hidden_states.shape[0])
        return MoonEPPreDispatchResult(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
            tokens_per_expert=tokens_per_expert.to(dtype=torch.int32).contiguous(),
            _moonep_call=layer_state,
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
        # Create the activation autograd edge and start weight prefetch.
        hidden_nvsh, topk_weights_nvs, cu_seqlens = _DispatchAutograd.apply(
            pre_dispatched["hidden_states"],
            pre_dispatched["topk_ids"],
            pre_dispatched["tokens_per_expert"],
            topk_weights.to(dtype=torch.float32).contiguous(),
            pre_dispatched["_moonep_call"],
            async_op,
        )
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=topk_weights_nvs,
            cu_seqlens=cu_seqlens,
        )

    @override
    def dispatch_postprocess(
        self,
        *,
        pre_dispatched: MoonEPPreDispatchResult,
        dispatched: MoonEPDispatchResult,
        async_op: bool = False,
    ) -> MoonEPPostDispatchResult:
        del async_op
        return prepare_experts(pre_dispatched["_moonep_call"], dispatched)

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
        del pre_dispatched, dispatched, post_dispatched, async_op, decoding
        return MoonEPPreCombineResult(hidden_states=hidden_states)

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
        del post_dispatched, decoding
        # Create the fused route-scaled combine autograd edge.
        return MoonEPCombineResult(
            hidden_states=_CombineAutograd.apply(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
                pre_dispatched["_moonep_call"],
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
        del post_dispatched, pre_combined
        return MoonEPPostCombineResult(
            hidden_states=finish_combine(
                pre_dispatched["_moonep_call"],
                combined["hidden_states"],
                async_op=async_op,
            )
        )


__all__ = [
    "MoonEPDispatcher",
    "MoonEPModelRuntime",
    "MoonEPPreDispatchResult",
    "MoonEPDispatchResult",
    "MoonEPPostDispatchResult",
    "MoonEPPreCombineResult",
    "MoonEPCombineResult",
    "MoonEPPostCombineResult",
    "require_moonep_backend",
]
