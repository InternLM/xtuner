from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xtuner.v1.module.dispatcher import build_dispatcher
from xtuner.v1.module.dispatcher.moonep import MoonEPDispatcher, MoonEPRuntime


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

    def landing(self, generation):
        return self._landings[generation]

    def prefetch_weights(self, *, buffer, plan, generation, grad_slot):
        landings = self.landing(generation)
        local_weights = tuple(torch.cat((weight, torch.zeros_like(weight))) for weight in landings)
        buffer.prefetch_weight(plan=plan, projections=landings, async_finish=False)
        return local_weights, self._slots[grad_slot]

    def local_token_counts(self, cu_seqlens):
        return torch.tensor([cu_seqlens[-1], 0, 0, 0], dtype=torch.int32)

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
        MoonEPRuntime,
        "_enqueue",
        lambda self, operation, inputs=(): (operation(), _Event()),
    )
    monkeypatch.setattr(
        "xtuner.v1.module.dispatcher.moonep._ExpertVMMWorkspace",
        _Workspace,
    )
    monkeypatch.setattr(moe_group_linear, "group_gemm", triton_group_gemm)
    return module


def test_staging_dispatcher_runs_the_public_six_stage_forward_seam(backend) -> None:
    ep_group = SimpleNamespace(size=lambda: 2)
    runtime = MoonEPRuntime(
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
        moonep_runtime=runtime,
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
    runtime.install_after_fsdp(fsdp_root=experts)

    hidden_states = torch.randn(3, 128, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    source_counts = torch.tensor([1, 2, 2, 1], dtype=torch.int64)
    route_weights = torch.full((3, 2), 0.5, dtype=torch.float32)

    with torch.no_grad():
        pre = dispatcher.dispatch_preprocess(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=route_weights,
            tokens_per_expert=source_counts,
        )
        dispatched = dispatcher.dispatch(pre_dispatched=pre, topk_weights=route_weights)
        assert runtime._buffer.prefetch_calls == 1
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
    assert post["expert_weight_layout"].trainable_weights[0].shape == (4, 256, 128)
    assert torch.equal(result["hidden_states"], hidden_states * 0.5)
    assert runtime._buffer.num_sms == 64

    with pytest.raises(RuntimeError, match="fixed S changed"):
        dispatcher.dispatch_preprocess(
            hidden_states=torch.randn(4, 128, dtype=torch.bfloat16),
            topk_ids=torch.zeros(4, 2, dtype=torch.int64),
            topk_weights=torch.full((4, 2), 0.5),
            tokens_per_expert=torch.tensor([8, 0, 0, 0]),
        )


def test_direct_install_failure_is_explicit_and_never_falls_back_to_staging(backend) -> None:
    ep_group = SimpleNamespace(size=lambda: 2)
    runtime = MoonEPRuntime(
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
        moonep_runtime=runtime,
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
        runtime.install_after_fsdp(fsdp_root=experts)

    assert _Workspace.allocated[-1].destroyed
