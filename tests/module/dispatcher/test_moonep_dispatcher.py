import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.dispatcher.moonep import (
    MoonEPDispatcher,
    MoonEPModelRuntime,
    _MoonEPExpertGradBridge,
    _MoonEPLayerGradJoin,
)


class _Event:
    def wait(self) -> None:
        return None


class _Stream:
    def record_event(self):
        return _Event()

    def wait_event(self, event) -> None:
        del event

    def synchronize(self) -> None:
        return None


class _Buffer:
    def __init__(
        self,
        *,
        S,
        H,
        K,
        E,
        num_ep_ranks,
        group,
        explicitly_destroy,
        num_sms,
    ):
        self.S = S
        self.K = K
        self.E = E
        self.B = E // num_ep_ranks
        self.num_sms = num_sms
        self.destroyed = False
        self.prefetch_calls = 0

    def dispatch(
        self,
        hidden_states,
        route_weights_sk=None,
        topk_experts_sk=None,
        tokens_per_expert=None,
        plan=None,
        async_finish=False,
        zero_copy=False,
    ):
        if plan is None:
            plan = object()
            cu_seqlens = torch.full((self.E + self.B,), hidden_states.shape[0], dtype=torch.int32)
        else:
            cu_seqlens = None
        result = (hidden_states.clone(), route_weights_sk[:, 0].contiguous(), cu_seqlens, plan)
        return (*result, _Event()) if async_finish else result

    def prefetch_weight(self, **kwargs):
        assert kwargs["async_finish"] is False
        self.prefetch_calls += 1
        return None

    def combine(
        self,
        *,
        plan,
        hidden_nvsh,
        route_weights_nvs=None,
        hidden_scales_nvs=None,
        async_finish=False,
        zero_copy=False,
    ):
        output = hidden_nvsh
        if hidden_scales_nvs is not None:
            output = output * hidden_scales_nvs[:, None].to(output.dtype)
        result = (output, None, _Event() if async_finish else None)
        return result

    def destroy(self) -> None:
        self.destroyed = True


class _Workspace:
    allocated = []

    @classmethod
    def allocate(
        cls,
        *,
        projection_shapes,
        num_experts,
        ep_group,
        gradient_slots,
        **kwargs,
    ):
        instance = cls()
        b = num_experts // ep_group.size()
        instance._landings = tuple(
            tuple(torch.zeros(b, *shape, dtype=torch.bfloat16) for shape in projection_shapes) for _ in range(2)
        )
        instance._slots = tuple(
            tuple(torch.zeros(2 * b, *shape, dtype=torch.bfloat16) for shape in projection_shapes)
            for _ in range(gradient_slots)
        )
        instance.destroyed = False
        cls.allocated.append(instance)
        return instance

    def generation_for(self, ordinal):
        return ordinal % 2

    def landing(self, generation):
        return self._landings[generation]

    def prefetch_weights(self, *, buffer, plan, generation):
        landings = self.landing(generation)
        local_weights = tuple(torch.cat((weight, torch.zeros_like(weight))) for weight in landings)
        buffer.prefetch_weight(plan=plan, projections=landings, async_finish=False)
        return local_weights

    def local_compute_view(self, *, hidden_nvsh, cu_seqlens):
        b = self._landings[0][0].shape[0]
        counts = torch.tensor([hidden_nvsh.shape[0]] + [0] * (2 * b - 1), dtype=torch.int32)
        return hidden_nvsh, counts

    def return_expert_gradients(self, *, buffer, plan, gradients, grad_slot, initialize):
        del buffer, plan
        b = self._landings[0][0].shape[0]
        targets = self._slots[grad_slot]
        for target, gradient in zip(targets, gradients, strict=True):
            if initialize:
                target[:b].zero_()
            target[:b].add_(gradient[:b])
            target[b:].copy_(gradient[b:])
        return targets[0][:b], targets[1][:b]

    def destroy(self) -> None:
        self.destroyed = True


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused_w1w3 = nn.Linear(128, 2 * 2 * 128, bias=False, dtype=torch.bfloat16)
        self.fused_w2 = nn.Linear(128, 2 * 128, bias=False, dtype=torch.bfloat16)


@pytest.fixture
def backend(monkeypatch):
    from xtuner.v1.module.dispatcher import moonep as moonep_integration
    from xtuner.v1.module.grouped_linear import moe_group_linear
    from xtuner.v1.ops.moe.cuda.group_gemm import triton_group_gemm

    _Workspace.allocated.clear()
    module = SimpleNamespace(
        __file__="/tmp/MoonEP-mod/moonep/__init__.py",
        XTUNER_INTEGRATION_API_VERSION=3,
        Buffer=_Buffer,
    )
    monkeypatch.setattr(moonep_integration, "_moonep_backend", module)
    monkeypatch.setattr(moonep_integration, "_MOONEP_IMPORT_ERROR", None)
    stream = _Stream()
    monkeypatch.setattr(moonep_integration.torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(moonep_integration.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(moonep_integration.torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(
        moonep_integration._MoonEPResources,
        "enqueue",
        lambda self, operation, inputs=(): (operation(), _Event()),
    )
    monkeypatch.setattr(
        "xtuner.v1.module.dispatcher.moonep._ExpertVMMWorkspace",
        _Workspace,
    )
    monkeypatch.setattr(moe_group_linear, "group_gemm", triton_group_gemm)
    return module


def test_staging_dispatcher_runs_the_public_forward_path(backend) -> None:
    ep_group = SimpleNamespace(size=lambda: 2)
    runtime = MoonEPModelRuntime(
        ep_group=ep_group,
        hidden_size=128,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        intra_layer_micro_batch=1,
        staging_reference=True,
    )
    experts = _Experts()
    dispatcher = build_dispatcher(
        dispatcher="moonep",
        n_routed_experts=4,
        ep_group=ep_group,
        ep_runtime=runtime,
        layer_fqn="layers.0.experts",
        projections=(experts.fused_w1w3, experts.fused_w2),
    )
    runtime.validate_before_fsdp(
        SimpleNamespace(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            requires_grad=True,
            cpu_offload=False,
            reshard_after_forward=True,
        )
    )
    runtime.install_after_fsdp(fsdp_root=experts, execution_order=["layers.0.experts"])

    hidden_states = torch.randn(3, 128, dtype=torch.bfloat16, requires_grad=True)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    source_counts = torch.tensor([1, 2, 2, 1], dtype=torch.int64)
    route_weights = torch.full((3, 2), 0.5, dtype=torch.float32)

    with torch.no_grad():
        layer_inputs, layer_states = dispatcher.prepare_layer_inputs([hidden_states])
        layer_input, layer_state = layer_inputs[0], layer_states[0]
        assert layer_input.grad_fn is None
        pre = dispatcher.dispatch_preprocess(
            hidden_states=layer_input,
            topk_ids=topk_ids,
            topk_weights=route_weights,
            tokens_per_expert=source_counts,
            layer_state=layer_state,
        )
        dispatched = dispatcher.dispatch(pre_dispatched=pre, topk_weights=route_weights)
        assert runtime.resources.buffer.prefetch_calls == 1
        post = dispatcher.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)
        pre_combined = dispatcher.combine_preprocess(
            hidden_states=post["hidden_states"],
            pre_dispatched=pre,
            dispatched=dispatched,
            post_dispatched=post,
        )
        combined = dispatcher.combine(
            pre_dispatched=pre,
            dispatched=dispatched,
            post_dispatched=post,
            pre_combined=pre_combined,
        )
        result = dispatcher.combine_postprocess(
            pre_dispatched=pre,
            dispatched=dispatched,
            post_dispatched=post,
            pre_combined=pre_combined,
            combined=combined,
        )

    assert isinstance(dispatcher, MoonEPDispatcher)
    assert pre["topk_ids"].dtype == torch.int32
    assert pre["tokens_per_expert"].dtype == torch.int32
    assert torch.equal(pre["tokens_per_expert"], source_counts.to(torch.int32))
    assert post["tokens_per_expert"].shape == (4,)
    assert post["expert_weight_layout"].trainable_weights is not None
    assert all(isinstance(weight, torch.Tensor) for weight in post["expert_weight_layout"].trainable_weights)
    assert post["expert_weight_layout"].trainable_weights[0].shape == (4, 256, 128)
    assert torch.equal(result["hidden_states"], hidden_states * 0.5)
    assert not result["hidden_states"].requires_grad
    assert runtime.resources.buffer.num_sms == 64

    call_state_ref = weakref.ref(layer_state)
    del layer_state, layer_states, pre, dispatched, post, pre_combined, combined
    assert call_state_ref() is None

    with pytest.raises(RuntimeError, match="requires layer_state from prepare_layer_input"):
        dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=route_weights,
            tokens_per_expert=source_counts,
        )

    with pytest.raises(RuntimeError, match="fixed S changed"):
        invalid_inputs, invalid_states = dispatcher.prepare_layer_inputs([torch.randn(4, 128, dtype=torch.bfloat16)])
        dispatcher.dispatch_preprocess(
            hidden_states=invalid_inputs[0],
            topk_ids=torch.zeros(4, 2, dtype=torch.int64),
            topk_weights=torch.full((4, 2), 0.5),
            tokens_per_expert=torch.tensor([8, 0, 0, 0]),
            layer_state=invalid_states[0],
        )


def test_direct_install_failure_is_explicit_and_never_falls_back_to_staging(backend) -> None:
    ep_group = SimpleNamespace(size=lambda: 2)
    runtime = MoonEPModelRuntime(
        ep_group=ep_group,
        hidden_size=128,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        intra_layer_micro_batch=1,
        staging_reference=False,
    )
    experts = _Experts()
    build_dispatcher(
        dispatcher="moonep",
        n_routed_experts=4,
        ep_group=ep_group,
        ep_runtime=runtime,
        layer_fqn="layers.0.experts",
        projections=(experts.fused_w1w3, experts.fused_w2),
    )

    runtime.validate_before_fsdp(
        SimpleNamespace(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            requires_grad=True,
            cpu_offload=False,
            reshard_after_forward=True,
        )
    )
    with pytest.raises(RuntimeError, match="could not find FSDPParam"):
        runtime.install_after_fsdp(fsdp_root=experts, execution_order=["layers.0.experts"])

    assert _Workspace.allocated[-1].destroyed


def test_gradient_reduce_start_hands_dw_to_the_workspace_once() -> None:
    from xtuner.v1.module.dispatcher import moonep as moonep_integration

    class _CompletionEvent:
        def __init__(self) -> None:
            self.waits = 0

        def wait(self) -> None:
            self.waits += 1

    gradients = tuple(torch.arange(24, dtype=torch.bfloat16).view(4, 2, 3) + projection for projection in range(2))
    calls: list[tuple] = []
    event = _CompletionEvent()

    class _Workspace:
        def return_expert_gradients(self, *, buffer, plan, gradients, grad_slot, initialize):
            del buffer, plan
            calls.append((gradients, grad_slot, initialize))
            return gradients[0][:2], gradients[1][:2]

    def enqueue(operation, inputs=()):
        del inputs
        return operation(), event

    resources = SimpleNamespace(workspace=_Workspace(), buffer=object(), enqueue=enqueue)
    layer = moonep_integration._MoonEPLayer(
        fqn="layers.0.experts", projections=(nn.Linear(3, 3), nn.Linear(3, 3)), ordinal=0
    )
    call_state = moonep_integration._MoonEPLayerCallState(
        resources=resources,
        layer=layer,
        generation=0,
        grad_slot=0,
        layer_gradients=moonep_integration._MoonEPLayerGradients(),
    )
    call_state.plan = object()
    home_parameters = (
        nn.Parameter(torch.zeros_like(gradients[0][:2])),
        nn.Parameter(torch.zeros_like(gradients[1][:2])),
    )
    call_state.home_parameters = home_parameters

    moonep_integration.start_gradient_completion(call_state, gradients)

    assert len(calls) == 1
    assert calls[0][1] == 0 and calls[0][2] is True
    assert call_state.layer_gradients.initialized is True
    assert event.waits == 0

    parameters, home_grads = moonep_integration.finish_gradient_completion(call_state)
    assert event.waits == 1
    assert parameters is home_parameters
    for parameter, actual, expected in zip(parameters, home_grads, gradients, strict=True):
        assert parameter.grad is None  # Only the layer Join publishes H.
        torch.testing.assert_close(actual, expected[:2])
    assert call_state.gradient_completion is None


def test_gradient_reduce_start_and_join_preserve_device_order(monkeypatch) -> None:
    from xtuner.v1.module.dispatcher import moonep as moonep_integration

    events: list[str] = []
    targets = (torch.zeros(4), torch.zeros(4))

    def fake_start(call_state, gradients) -> None:
        del call_state
        for target, gradient in zip(targets, gradients, strict=True):
            target.copy_(gradient)
        events.append("start")

    def fake_finish(call_state):
        del call_state
        events.append("finish")
        return (nn.Parameter(torch.zeros(4)), nn.Parameter(torch.zeros(4))), targets

    monkeypatch.setattr(moonep_integration, "start_gradient_completion", fake_start)
    monkeypatch.setattr(moonep_integration, "finish_gradient_completion", fake_finish)

    class _WriteWGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, weight, projection):
            ctx.projection = projection
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(f"projection-{ctx.projection}")
            return grad, torch.full_like(grad, ctx.projection + 1), None

    call_state = object()
    source = torch.ones(4, requires_grad=True)

    (joined,) = _MoonEPLayerGradJoin.apply((call_state,), source)
    started, w0, w1 = _MoonEPExpertGradBridge.apply(
        joined, nn.Parameter(torch.ones(4)), nn.Parameter(torch.ones(4)), call_state
    )
    projection_0 = _WriteWGrad.apply(started, w0, 0)
    projection_1 = _WriteWGrad.apply(projection_0, w1, 1)
    projection_1.sum().backward()

    assert joined.data_ptr() == source.data_ptr()
    assert started.data_ptr() == source.data_ptr()
    assert events == ["projection-1", "projection-0", "start", "finish"]
