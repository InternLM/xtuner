"""MoonEP's model-scoped XTuner integration.

The backend import lives in this module and remains lazy: importing XTuner or
building another dispatcher must not require MoonEP.  The three stateful
classes added here are intentionally deep modules.  ``MoonEPRuntime`` owns
model resources, ``_MoonEPInvocation`` owns one dispatch/combine pairing, and
``MoonEPDispatcher`` adapts that state to XTuner's six-stage dispatcher seam.
"""

from __future__ import annotations

import importlib
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


_INTEGRATION_API_VERSION = 2
_TARGET_TORCH_VERSION = "2.12.1+cu132"


def require_moonep_backend() -> Any:
    """Load and validate the optional MoonEP-mod package on first selection."""
    try:
        backend = importlib.import_module("moonep")
    except ImportError as exc:
        raise RuntimeError("dispatcher='moonep' requires the MoonEP-mod integration package") from exc

    source = getattr(backend, "__file__", "<unknown>")
    if getattr(backend, "XTUNER_INTEGRATION_API_VERSION", None) != _INTEGRATION_API_VERSION:
        raise RuntimeError(
            f"incompatible MoonEP integration API; expected {_INTEGRATION_API_VERSION}; loaded module: {source}"
        )

    workspace = getattr(backend, "ExpertVMMWorkspace", None)
    if (
        not hasattr(backend, "Buffer")
        or workspace is None
        or not hasattr(workspace, "validate")
        or not hasattr(workspace, "allocate")
    ):
        raise RuntimeError(f"MoonEP-mod XTuner capabilities are missing: {source}")
    if torch.__version__ != _TARGET_TORCH_VERSION:
        raise RuntimeError(
            f"MoonEP integration requires torch {_TARGET_TORCH_VERSION}, "
            f"got {torch.__version__}; loaded module: {source}"
        )
    return backend


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
        self._backend = require_moonep_backend()
        if intra_layer_micro_batch < 1:
            raise ValueError("intra_layer_micro_batch must be positive")

        self._ep_group = ep_group
        self._hidden_size = hidden_size
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._top_k = top_k
        self._intra_layer_micro_batch = intra_layer_micro_batch
        self._staging_reference = staging_reference
        self._num_sms = num_sms

        # This is deliberately the complete meta-build action.  The backend
        # validates kernel metadata here but cannot create CUDA/VMM/socket
        # resources until native FSDP has finished mutating parameters.
        self._backend.ExpertVMMWorkspace.validate(
            projection_shapes=(
                (2 * intermediate_size, hidden_size),
                (hidden_size, intermediate_size),
            ),
            num_experts=num_experts,
            ep_size=ep_group.size(),
            top_k=top_k,
            dtype=torch.bfloat16,
            home_generations=2,
            gradient_slots=intra_layer_micro_batch,
        )

        self._buffer: Any | None = None
        self._workspace: Any | None = None
        self._fsdp_params: tuple[Any, ...] = ()
        self._layers: list[tuple[str, tuple[nn.Module, nn.Module]]] = []
        self._fixed_tokens_per_rank: int | None = None
        self._closed = False

    def bind_dispatcher(
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
        workspace = self._backend.ExpertVMMWorkspace.allocate(
            projection_shapes=(
                (2 * self._intermediate_size, self._hidden_size),
                (self._hidden_size, self._intermediate_size),
            ),
            num_experts=self._num_experts,
            ep_group=self._ep_group,
            top_k=self._top_k,
            dtype=torch.bfloat16,
            home_generations=2,
            gradient_slots=self._intra_layer_micro_batch,
        )
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
            self._buffer = self._backend.Buffer(
                S=tokens_per_rank,
                H=self._hidden_size,
                K=self._top_k,
                E=self._num_experts,
                num_ep_ranks=self._ep_group.size(),
                group=self._ep_group,
                explicitly_destroy=True,
                num_sms=self._num_sms,
                # FSDP collectives use the caller stream.  Giving MoonEP the
                # same device-side launch order prevents orthogonal EP/FSDP
                # progress waves without introducing a host synchronization.
                use_caller_stream=True,
            )
        return self._buffer

    def close(self) -> None:
        """Release Buffer before VMM workspace at a coordinated boundary."""
        if self._closed:
            return
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
    _moonep_invocation: _MoonEPInvocation


class MoonEPPostDispatchResult(PostDispatchResult): ...


class MoonEPPreCombineResult(TypedDict):
    hidden_states: torch.Tensor
    _moonep_invocation: _MoonEPInvocation


class MoonEPCombineResult(TypedDict):
    hidden_states: torch.Tensor


class MoonEPPostCombineResult(TypedDict):
    hidden_states: torch.Tensor


class _MoonEPInvocation:
    """Own one fresh MoonEP plan and all of its device-side completion
    edges."""

    def __init__(self, runtime: MoonEPRuntime, *, layer_id: int, grad_slot: int) -> None:
        self._runtime = runtime
        self._layer_id = layer_id
        self._grad_slot = grad_slot
        self._plan: Any | None = None
        self._dispatch_done: Any | None = None
        self._combine_done: Any | None = None
        self._fallback_gradient_targets: ProjectionPair | None = None
        self._home_parameters: tuple[nn.Parameter, nn.Parameter] | None = None
        self._local_gradient_parameters: list[nn.Parameter | None] = [None, None]

    def dispatch(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        topk_weights: torch.Tensor,
        async_op: bool,
    ) -> MoonEPDispatchResult:
        hidden_nvsh, weights_nvs, cu_seqlens = _DispatchAutograd.apply(
            hidden_states,
            topk_ids,
            tokens_per_expert,
            topk_weights,
            self,
            async_op,
        )
        return MoonEPDispatchResult(
            hidden_states=hidden_nvsh,
            topk_weights=weights_nvs,
            cu_seqlens=cu_seqlens,
            _moonep_invocation=self,
        )

    def _current_home_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Return current FSDP leaves, staging their values only by request."""
        layer_fqn, projections = self._runtime._layers[self._layer_id]
        if not self._runtime._staging_reference:
            return fsdp_current_unsharded_expert_parameters(projections)

        workspace = self._runtime._workspace
        assert workspace is not None
        landings = workspace.landing(self._layer_id % 2)
        parameters: list[nn.Parameter] = []
        for linear, landing in zip(
            projections,
            landings,
            strict=True,
        ):
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

    def prepare_experts(
        self,
        dispatched: MoonEPDispatchResult,
        *,
        async_op: bool,
    ) -> MoonEPPostDispatchResult:
        del async_op
        runtime = self._runtime
        workspace = runtime._workspace
        buffer = runtime._buffer
        assert workspace is not None and buffer is not None and self._plan is not None

        # FSDP has exposed its BF16 unsharded Parameters at this point.
        # Staging changes only how their values reach the MoonEP workspace;
        # completed home gradients are handed back before FSDP post-backward.
        with torch.profiler.record_function("MoonEP::prepare_experts"):
            home_parameters = self._current_home_parameters()
            local_weights, gradient_targets, weights_ready = workspace.materialize(
                buffer=buffer,
                plan=self._plan,
                generation=self._layer_id % 2,
                grad_slot=self._grad_slot,
            )
            # async_op=True leaves dispatch on MoonEP's comm stream. Its event
            # is a device dependency (never a host wait) that makes cu_seqlens
            # safe before deriving local grouped-GEMM counts.
            assert self._dispatch_done is not None
            self._dispatch_done.wait()
            local_counts = workspace.local_token_counts(dispatched["cu_seqlens"])
            # MoonEP returns a fixed-capacity buffer while standard grouped
            # GEMM requires counts to cover every row. Assign the zeroed tail
            # to the final physical group so every backend sees one contract.
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
            weights_ready.wait()
        trainable_wgrad_outs: ProjectionPair | None = None
        if torch.is_grad_enabled():
            # A leaf Parameter lets AccumulateGrad steal the preallocated VMM
            # WGrad target. The post-accumulate hooks then complete both fused
            # projections as one MoonEP transaction before FSDP post-backward.
            differentiable_weights = (
                nn.Parameter(local_weights[0]),
                nn.Parameter(local_weights[1]),
            )
            self._home_parameters = home_parameters
            for projection, parameter in enumerate(differentiable_weights):
                parameter.register_post_accumulate_grad_hook(
                    lambda completed_parameter, projection=projection: self._record_parameter_gradient(
                        projection, completed_parameter
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

    def combine(self, expert_output: torch.Tensor, route_weights: torch.Tensor, *, async_op: bool) -> torch.Tensor:
        return _CombineAutograd.apply(expert_output, route_weights, self, async_op)

    def wait_combined(self) -> None:
        assert self._combine_done is not None
        self._combine_done.wait()

    def finish_forward_only(self) -> None:
        # Event waits above are device dependencies. Dropping Python plan
        # ownership must never query/synchronize route-dependent work.
        self._plan = None
        self._dispatch_done = None
        self._combine_done = None
        self._fallback_gradient_targets = None
        self._home_parameters = None
        self._local_gradient_parameters = [None, None]

    def _dispatch_backward(
        self,
        grad_hidden_nvsh: torch.Tensor,
        grad_route_weights_nvs: torch.Tensor,
    ) -> ProjectionPair:
        """Use forward's plan to combine source hidden/router gradients."""
        assert self._plan is not None and self._runtime._buffer is not None
        with torch.profiler.record_function("MoonEP::dispatch_backward"):
            grad_hidden, grad_route_weights, done = self._runtime._buffer.combine(
                plan=self._plan,
                hidden_nvsh=grad_hidden_nvsh.contiguous(),
                route_weights_nvs=grad_route_weights_nvs.contiguous(),
                async_finish=True,
                zero_copy=False,
            )
            assert grad_route_weights is not None
            done.wait()
        return grad_hidden, grad_route_weights

    def _combine_backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Dispatch output gradients and overlap duplicated-weight replay."""
        runtime = self._runtime
        workspace = runtime._workspace
        assert self._plan is not None and runtime._buffer is not None and workspace is not None
        with torch.profiler.record_function("MoonEP::combine_backward"):
            grad_weighted, no_weights, no_cu, reused_plan, dispatch_done = runtime._buffer.dispatch(
                grad_output.contiguous(),
                plan=self._plan,
                async_finish=True,
                zero_copy=False,
            )
            assert no_weights is None and no_cu is None and reused_plan is self._plan

            # The pre-backward AllGather has restored this generation before
            # duplicated weights are replayed.
            replay_home_parameters = self._current_home_parameters()
            if self._home_parameters is not None and any(
                replay is not forward
                for replay, forward in zip(replay_home_parameters, self._home_parameters, strict=True)
            ):
                raise RuntimeError("MoonEP backward observed a different FSDP unsharded Parameter")
            _, fallback_gradient_targets, replay_done = workspace.materialize(
                buffer=runtime._buffer,
                plan=self._plan,
                generation=self._layer_id % 2,
                grad_slot=self._grad_slot,
            )
            # These are fresh Tensor objects aliasing the same VMM slot as the
            # forward targets. Keeping them does not increase the reference
            # count that controls AccumulateGrad's storage-stealing fast path.
            self._fallback_gradient_targets = fallback_gradient_targets
            dispatch_done.wait()
        return grad_weighted, replay_done

    def _complete_weight_gradients(self, local_grads: ProjectionPair) -> ProjectionPair:
        """Return duplicated BF16 partials before FSDP ReduceScatter."""
        runtime = self._runtime
        assert self._plan is not None and runtime._buffer is not None and runtime._workspace is not None
        assert self._fallback_gradient_targets is not None
        with torch.profiler.record_function("MoonEP::gradient_handoff"):
            reduction_grads: list[torch.Tensor] = []
            for source, target in zip(local_grads, self._fallback_gradient_targets, strict=True):
                if source.data_ptr() == target.data_ptr():
                    reduction_grads.append(source)
                else:
                    # Extra references can force AccumulateGrad to copy the
                    # direct-output target into Parameter.grad. The original
                    # target is already complete, so reduce it without a
                    # second full-gradient copy.
                    reduction_grads.append(target)
            home_grads, done = runtime._workspace.complete_gradients(
                buffer=runtime._buffer,
                plan=self._plan,
                local_grads=tuple(reduction_grads),
                grad_slot=self._grad_slot,
            )
            done.wait()
        self._fallback_gradient_targets = None
        return home_grads

    def _record_parameter_gradient(self, projection: int, parameter: nn.Parameter) -> None:
        if parameter.grad is None:
            raise RuntimeError("MoonEP local expert Parameter completed without a gradient")
        self._local_gradient_parameters[projection] = parameter
        if any(item is None for item in self._local_gradient_parameters):
            return

        local_parameters = cast(list[nn.Parameter], self._local_gradient_parameters)
        local_grads = cast(ProjectionPair, tuple(parameter.grad for parameter in local_parameters))
        assert self._home_parameters is not None and all(gradient is not None for gradient in local_grads)
        home_grads = self._complete_weight_gradients(local_grads)
        accumulate_fsdp_unsharded_expert_gradients(self._home_parameters, home_grads)
        for local_parameter in local_parameters:
            local_parameter.grad = None
        self._home_parameters = None
        self._local_gradient_parameters = [None, None]


class _DispatchAutograd(torch.autograd.Function):
    """Concrete MoonEP dispatch forward paired with combine backward."""

    @staticmethod
    def forward(
        ctx: Any,
        source_hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        source_route_weights: torch.Tensor,
        invocation: _MoonEPInvocation,
        async_op: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.invocation = invocation
        buffer = invocation._runtime._buffer_for(source_hidden.shape[0])
        with torch.profiler.record_function("MoonEP::dispatch_forward"):
            hidden_nvsh, route_weights_nvs, cu_seqlens, plan, done = buffer.dispatch(
                source_hidden,
                route_weights_sk=source_route_weights,
                topk_experts_sk=topk_ids,
                tokens_per_expert=tokens_per_expert,
                async_finish=True,
                zero_copy=False,
            )
        assert route_weights_nvs is not None and cu_seqlens is not None
        invocation._plan = plan
        invocation._dispatch_done = done
        if not async_op:
            done.wait()
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
        grad_hidden, grad_route_weights = ctx.invocation._dispatch_backward(
            grad_hidden_nvsh,
            grad_route_weights_nvs,
        )
        return grad_hidden, None, None, grad_route_weights, None, None


class _CombineAutograd(torch.autograd.Function):
    """Fused route-scaled combine forward paired with plan-reuse dispatch."""

    @staticmethod
    def forward(
        ctx: Any,
        expert_output: torch.Tensor,
        route_weights: torch.Tensor,
        invocation: _MoonEPInvocation,
        async_op: bool,
    ) -> torch.Tensor:
        ctx.invocation = invocation
        ctx.save_for_backward(expert_output, route_weights)
        assert invocation._plan is not None and invocation._runtime._buffer is not None
        with torch.profiler.record_function("MoonEP::combine_forward"):
            output, gathered_weights, done = invocation._runtime._buffer.combine(
                plan=invocation._plan,
                hidden_nvsh=expert_output,
                hidden_scales_nvs=route_weights,
                route_weights_nvs=None,
                async_finish=True,
                zero_copy=False,
            )
        assert gathered_weights is None
        invocation._combine_done = done
        if not async_op:
            done.wait()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_weighted, replay_done = ctx.invocation._combine_backward(grad_output)
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
        return _MoonEPInvocation(self._runtime, layer_id=self._layer_id, grad_slot=grad_slot).dispatch(
            hidden_states=pre_dispatched["hidden_states"],
            topk_ids=pre_dispatched["topk_ids"],
            tokens_per_expert=pre_dispatched["tokens_per_expert"],
            topk_weights=topk_weights.to(dtype=torch.float32).contiguous(),
            async_op=async_op,
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
        del pre_dispatched, decoding
        return dispatched["_moonep_invocation"].prepare_experts(dispatched, async_op=async_op)

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
        invocation = pre_combined["_moonep_invocation"]
        return MoonEPCombineResult(
            hidden_states=invocation.combine(
                pre_combined["hidden_states"],
                dispatched["topk_weights"],
                async_op=async_op,
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
        invocation = pre_combined["_moonep_invocation"]
        if async_op:
            invocation.wait_combined()
        result = MoonEPPostCombineResult(hidden_states=combined["hidden_states"])
        if not torch.is_grad_enabled():
            invocation.finish_forward_only()
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
