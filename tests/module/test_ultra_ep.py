import pytest
import torch
import torch.nn as nn

from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayer,
    _UltraEPGradReduceJoin,
    _UltraEPGradReduceStart,
    _UltraEPWeightRestoreJoin,
    _UltraEPWeightRestoreStart,
)
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.ultraep import UltraEPConfig
from xtuner.v1.module.ultraep import runtime as ultraep_runtime
from xtuner.v1.module.ultraep.runtime import UltraEPLayerRuntime, UltraEPManagerProvider
from xtuner.v1.ops.moe.cuda import group_gemm as group_gemm_module


class FakeGroup:
    def __init__(self, size: int = 8):
        self._size = size

    def size(self):
        return self._size


class FakeGroupedLinear:
    def __init__(self, shape):
        self.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        self.configure_calls = []
        self.select_calls = []

    def configure_ultra_ep_buffers(self, replica_weight, replica_grad):
        self.configure_calls.append((replica_weight, replica_grad))

    def select_ultra_ep_slot(self, slot):
        self.select_calls.append(slot)


class FakeEvent:
    def __init__(self, calls, virtual_layer_id):
        self.calls = calls
        self.virtual_layer_id = virtual_layer_id

    def current_stream_wait(self):
        self.calls.append(("wait", self.virtual_layer_id))


class FakeLayerManager:
    def __init__(self, *, num_master_experts=2, redundant=1, hidden_size=4, intermediate_size=3):
        self.num_local_redundant_experts = redundant
        self.max_microbatches = 2
        self.real_num_alloc_layers = 4
        self.local_replica_fc1_weight_buffer = torch.empty(
            self.max_microbatches, redundant * 2 * intermediate_size * hidden_size
        )
        self.local_replica_fc2_weight_buffer = torch.empty(
            self.max_microbatches, redundant * hidden_size * intermediate_size
        )
        self.local_replica_fc1_grad_buffer = torch.empty(
            self.max_microbatches,
            redundant * 2 * intermediate_size * hidden_size,
            dtype=torch.float32,
        )
        self.local_replica_fc2_grad_buffer = torch.empty(
            self.max_microbatches,
            redundant * hidden_size * intermediate_size,
            dtype=torch.float32,
        )
        self.master_fc1_grad_staging = torch.empty(
            num_master_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        self.master_fc2_grad_staging = torch.empty(
            num_master_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
        )
        self.register_calls = []
        self.refresh_calls = []
        self.weight_sync_calls = []
        self.stage_calls = []
        self.grad_reduce_calls = []
        self.restore_calls = []
        self.event_calls = []

    def replica_slot(self, virtual_layer_id):
        return virtual_layer_id // self.real_num_alloc_layers

    def register_master_pointers(self, **kwargs):
        self.register_calls.append(kwargs)

    def refresh_master_weight_pointers(self, **kwargs):
        self.refresh_calls.append(kwargs)

    def weight_sync(self, layer_id, *, async_finish):
        self.weight_sync_calls.append((layer_id, async_finish))
        return FakeEvent(self.event_calls, layer_id)

    def stage_master_gradients(self, *, virtual_layer_id, fc1_grad, fc2_grad):
        self.stage_calls.append(virtual_layer_id)
        self.master_fc1_grad_staging.copy_(fc1_grad)
        self.master_fc2_grad_staging.copy_(fc2_grad)

    def grad_reduce(self, layer_id, *, async_finish):
        self.grad_reduce_calls.append((layer_id, async_finish))
        self.master_fc1_grad_staging.add_(1.0)
        self.master_fc2_grad_staging.add_(2.0)
        return FakeEvent(self.event_calls, layer_id)

    def restore_master_gradients(self, *, virtual_layer_id, fc1_grad, fc2_grad):
        del fc1_grad, fc2_grad
        self.restore_calls.append(virtual_layer_id)
        return self.master_fc1_grad_staging, self.master_fc2_grad_staging


class FakeManagerProvider:
    num_model_layers = 4
    num_logical_experts = 16
    hidden_size = 4
    expert_intermediate_size = 3
    num_redundant_experts_per_rank = 1
    max_microbatches = 1

    def __init__(self, manager):
        self.manager = manager
        self.get_manager_calls = 0

    @property
    def num_dispatch_experts(self):
        return self.num_logical_experts + 8 * self.num_redundant_experts_per_rank

    def get_manager(self):
        self.get_manager_calls += 1
        return self.manager

    def configure_max_microbatches(self, requested):
        self.max_microbatches = max(self.max_microbatches, int(requested))


def make_fake_layer_runtime():
    manager = FakeLayerManager()
    provider = FakeManagerProvider(manager)
    fused_w1w3 = FakeGroupedLinear((2, 6, 4))
    fused_w2 = FakeGroupedLinear((2, 4, 3))
    runtime = UltraEPLayerRuntime(
        layer_id=2,
        manager_provider=provider,  # type: ignore[arg-type]
        fused_w1w3=fused_w1w3,
        fused_w2=fused_w2,
    )
    return runtime, manager, fused_w1w3, fused_w2


def test_ultra_ep_is_opt_in():
    config = Qwen3MoE235BA22Config()

    assert config.ultraep_cfg is None


def test_ultra_ep_requires_redundant_experts():
    with pytest.raises(ValueError, match="num_redundant_experts_per_rank"):
        UltraEPConfig(num_redundant_experts_per_rank=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_routed_experts": 33}, "n_routed_experts"),
        ({"ep_size": 1}, "ep_size"),
        ({"dispatcher": "all2all"}, "dispatcher='deepep'"),
        (
            {"float8_cfg": Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)},
            "BF16",
        ),
        ({"moe_bias": True}, "expert bias"),
        ({"expert_tp_size": 2}, "expert_tp_size == 1"),
        ({"mtp_config": MTPConfig(num_layers=1)}, "MTP expert layers"),
    ],
)
def test_ultra_ep_rejects_unsupported_model_config(overrides, match):
    kwargs = {
        "ep_size": 8,
        "n_routed_experts": 32,
        "dispatcher": "deepep",
        "ultraep_cfg": UltraEPConfig(num_redundant_experts_per_rank=1),
    }
    kwargs.update(overrides)
    config = Qwen3MoE235BA22Config(**kwargs)

    with pytest.raises(ValueError, match=match):
        config.build()


def test_ultra_ep_rejects_activation_recompute():
    model = object.__new__(MoE)
    model.config = Qwen3MoE235BA22Config(
        ep_size=8,
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )

    with pytest.raises(ValueError, match="activation recompute"):
        MoE.fully_shard(model, FSDPConfig(ep_size=8, recompute_ratio=0.5))


def test_ultra_ep_manager_provider_derives_shape_from_xtuner_config():
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(
            num_redundant_experts_per_rank=2,
        )
    )

    provider = UltraEPManagerProvider.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )

    assert provider.num_model_layers == config.num_hidden_layers
    assert provider.num_logical_experts == config.n_routed_experts
    assert provider.hidden_size == config.hidden_size
    assert provider.expert_intermediate_size == config.moe_intermediate_size
    assert provider.num_redundant_experts_per_rank == config.ultraep_cfg.num_redundant_experts_per_rank
    assert provider.max_microbatches == 1
    assert (
        provider.num_dispatch_experts
        == config.n_routed_experts + 8 * config.ultraep_cfg.num_redundant_experts_per_rank
    )
    assert provider._manager is None

    with pytest.raises(RuntimeError, match="installed after FSDP setup"):
        provider.ensure_materialized()
    with pytest.raises(RuntimeError, match="installed after FSDP setup"):
        provider.get_manager()

    runtime = UltraEPLayerRuntime(
        layer_id=config.num_hidden_layers - 1,
        manager_provider=provider,
        fused_w1w3=object(),  # type: ignore[arg-type]
        fused_w2=object(),  # type: ignore[arg-type]
    )
    assert runtime.manager_provider is provider
    assert provider._manager is None

    with pytest.raises(ValueError, match="layer_id"):
        UltraEPLayerRuntime(
            layer_id=config.num_hidden_layers,
            manager_provider=provider,
            fused_w1w3=object(),  # type: ignore[arg-type]
            fused_w2=object(),  # type: ignore[arg-type]
        )


def test_ultra_ep_provider_binding_install_is_transactional(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    provider = UltraEPManagerProvider.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    monkeypatch.setattr(
        binding,
        "install_ultraep_fsdp_binding",
        lambda **kwargs: ("binding",),
    )
    root = object()
    provider.install_after_fsdp(fsdp_root=root, targets=[])
    assert provider._state == "INSTALLED"
    assert provider._fsdp_root is root
    assert provider.fsdp_binding == ("binding",)


def test_ultra_ep_provider_failed_binding_does_not_publish_installed_state(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    provider = UltraEPManagerProvider.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    def fail(**kwargs):
        raise RuntimeError("binding failed")

    monkeypatch.setattr(binding, "install_ultraep_fsdp_binding", fail)
    with pytest.raises(RuntimeError, match="binding failed"):
        provider.install_after_fsdp(fsdp_root=object(), targets=[])
    assert provider._state == "CREATED"
    assert provider._fsdp_root is None
    assert not hasattr(provider, "fsdp_binding")


def test_ultra_ep_provider_requires_reshard_and_grad(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    provider = UltraEPManagerProvider.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    monkeypatch.setattr(ultraep_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(ultraep_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ultraep_runtime.dist, "get_world_size", lambda: 8)

    # EP8×DP1: historical subexperiment-8 path keeps params unsharded.
    provider.validate_before_fsdp(
        FSDPConfig(recompute_ratio=0, ep_size=8, reshard_after_forward=False)
    )
    # EP4×DP2: closed unshard window is required.
    with pytest.raises(ValueError, match="reshard_after_forward"):
        provider.validate_before_fsdp(
            FSDPConfig(recompute_ratio=0, ep_size=4, reshard_after_forward=False)
        )
    with pytest.raises(ValueError, match="reshard_after_forward"):
        provider.validate_before_fsdp(
            FSDPConfig(recompute_ratio=0, hsdp_sharding_size=2, reshard_after_forward=False)
        )
    with pytest.raises(ValueError, match="requires_grad"):
        provider.validate_before_fsdp(FSDPConfig(recompute_ratio=0, ep_size=8, requires_grad=False))


def test_ultra_ep_provider_configures_virtual_layer_capacity_before_materialization():
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    provider = UltraEPManagerProvider.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    provider.configure_max_microbatches(2)

    assert provider.max_microbatches == 2

    # Native Manager capacity is immutable once materialized; decreasing the
    # requested count is harmless, while an increase must fail loudly.
    provider._manager = object()  # type: ignore[assignment]
    provider.configure_max_microbatches(1)
    with pytest.raises(RuntimeError, match="already materialized"):
        provider.configure_max_microbatches(3)


def test_ultra_ep_microbatch_pipeline_scales_beyond_two():
    """Exercise the generic MB-N forward schedule without a CUDA dispatcher.

    All preprocesses complete before any dispatch; all expert pre_combines
    complete before any combine.  This also protects MB>2 from list-length
    or accidental hard-coded-two changes.
    """

    class FakeRuntime:
        def __init__(self):
            self.next_virtual_layer_id = 0
            self.configured = []
            self.calls = []

        def configure_max_microbatches(self, requested):
            self.configured.append(requested)

        def allocate_virtual_layer_id(self):
            virtual_layer_id = self.next_virtual_layer_id
            self.next_virtual_layer_id += 1
            self.calls.append(("allocate", virtual_layer_id))
            return virtual_layer_id

        def update_placement(self, topk_ids, virtual_layer_id):
            self.calls.append(("placement", virtual_layer_id))

        def sync_weights(self, virtual_layer_id, *, async_finish):
            self.calls.append(("sync", virtual_layer_id, async_finish))
            return FakeEvent(self.calls, virtual_layer_id)

        def reroute(self, topk_ids, virtual_layer_id):
            self.calls.append(("reroute", virtual_layer_id))
            return topk_ids

        def bind_virtual_layer_slot(self, virtual_layer_id):
            self.calls.append(("bind", virtual_layer_id))

    class FakeDispatcher:
        def __init__(self, calls):
            self.calls = calls

        def dispatch_preprocess(self, *, hidden_states, topk_ids, topk_weights, async_op):
            self.calls.append("preprocess")
            return {"hidden_states": hidden_states, "topk_ids": topk_ids}

        def dispatch(self, *, pre_dispatched, topk_weights, async_op):
            self.calls.append("dispatch")
            return {"pre_dispatched": pre_dispatched}

        def dispatch_postprocess(self, *, pre_dispatched, dispatched, async_op):
            self.calls.append("postprocess")
            hidden_states = pre_dispatched["hidden_states"]
            return {
                "hidden_states": hidden_states,
                "tokens_per_expert": torch.tensor([hidden_states.shape[0]]),
            }

        def combine_preprocess(self, *, hidden_states, **kwargs):
            self.calls.append("combine_preprocess")
            return {"hidden_states": hidden_states}

        def combine(self, *, pre_combined, **kwargs):
            self.calls.append("combine")
            return pre_combined

        def combine_postprocess(self, *, combined, **kwargs):
            self.calls.append("combine_postprocess")
            return combined

    layer = object.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer._ultraep = FakeRuntime()
    layer.n_shared_experts = 0
    layer.ep_mesh = None
    calls = []
    layer.dispatcher = FakeDispatcher(calls)
    layer.experts = lambda hidden_states, *args, **kwargs: hidden_states
    layer._pre_moe_forward = lambda hidden_states, **kwargs: (
        hidden_states,
        hidden_states,
        {
            "logits": hidden_states[..., :1],
            "router_weights": hidden_states[..., :1],
            "topk_ids": torch.zeros((*hidden_states.shape[:-1], 1), dtype=torch.long),
            "topk_weights": torch.ones((*hidden_states.shape[:-1], 1)),
        },
        None,
    )
    layer._post_moe_forward = lambda *, combined_hidden_states, residual, shared_experts_out: combined_hidden_states

    hidden_states = [torch.zeros(1, 2, 4) for _ in range(3)]
    output = layer._micro_batch_forward(
        hidden_states,
        seq_ctx_list=[None, None, None],  # type: ignore[list-item]
        position_embeddings_list=[(None, None), (None, None), (None, None)],  # type: ignore[list-item]
    )

    assert layer._ultraep.configured == [3]
    assert [call[1] for call in layer._ultraep.calls if call[0] == "allocate"] == [0, 1, 2]
    assert len(output["hidden_states"]) == 3
    assert calls[:3] == ["preprocess", "preprocess", "preprocess"]
    assert calls[3:12] == [
        "dispatch",
        "postprocess",
        "combine_preprocess",
        "dispatch",
        "postprocess",
        "combine_preprocess",
        "dispatch",
        "postprocess",
        "combine_preprocess",
    ]
    assert calls[12:15] == ["combine", "combine", "combine"]


def test_ultra_ep_manager_registry_reuses_group_and_rejects_shape_mismatch(monkeypatch):
    created = []

    class FakeUltraEPManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(ultraep_runtime, "_MANAGERS", {})
    monkeypatch.setattr(ultraep_runtime, "UltraEPManager", FakeUltraEPManager)
    group = FakeGroup()

    manager = ultraep_runtime.get_or_create_ultra_ep_manager(
        group=group,  # type: ignore[arg-type]
        num_layers=20,
        num_local_master_experts=4,
        num_local_redundant_experts=1,
        expert_fc1_numel=24,
        expert_fc2_numel=12,
        max_microbatches=1,
    )
    same_manager = ultraep_runtime.get_or_create_ultra_ep_manager(
        group=group,  # type: ignore[arg-type]
        num_layers=20,
        num_local_master_experts=4,
        num_local_redundant_experts=1,
        expert_fc1_numel=24,
        expert_fc2_numel=12,
        max_microbatches=1,
    )

    assert same_manager is manager
    assert created == [manager]

    with pytest.raises(RuntimeError, match="same UltraEP shape/configuration"):
        ultraep_runtime.get_or_create_ultra_ep_manager(
            group=group,  # type: ignore[arg-type]
            num_layers=21,
            num_local_master_experts=4,
            num_local_redundant_experts=1,
            expert_fc1_numel=24,
            expert_fc2_numel=12,
            max_microbatches=1,
        )


def test_ultra_ep_configure_buffers_selects_independent_microbatch_slot(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "triton_dual")
    linear = GroupedLinear(4, 6, 2)
    replica_weight = torch.stack(
        (
            torch.full((1, 6, 4), 1.0, dtype=torch.bfloat16),
            torch.full((1, 6, 4), 2.0, dtype=torch.bfloat16),
        )
    )
    replica_grad = torch.zeros_like(replica_weight, dtype=torch.float32)

    linear.configure_ultra_ep_buffers(replica_weight, replica_grad)
    torch.testing.assert_close(linear._ultra_ep_replica_weight, replica_weight[0])

    linear.select_ultra_ep_slot(1)
    torch.testing.assert_close(linear._ultra_ep_replica_weight, replica_weight[1])
    assert linear._ultra_ep_replica_weight.data_ptr() == replica_weight[1].data_ptr()
    assert linear._ultra_ep_replica_grad.data_ptr() == replica_grad[1].data_ptr()

    with pytest.raises(IndexError, match="outside the configured range"):
        linear.select_ultra_ep_slot(2)



def test_ultra_ep_configure_buffers_rejects_cutlass_backend(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "cutlass")
    linear = GroupedLinear(4, 6, 2)

    with pytest.raises(RuntimeError, match="Triton dual-base or TE"):
        linear.configure_ultra_ep_buffers(
            torch.empty(1, 1, 6, 4, dtype=torch.bfloat16),
            torch.empty(1, 1, 6, 4, dtype=torch.float32),
        )


def test_ultra_ep_buffers_are_not_parameters_or_state_dict_entries():
    linear = GroupedLinear(4, 6, 2)
    parameter_names_before = tuple(name for name, _ in linear.named_parameters())
    state_dict_names_before = tuple(linear.state_dict())

    linear.configure_ultra_ep_buffers(
        torch.empty(1, 1, 6, 4, dtype=torch.bfloat16),
        torch.empty(1, 1, 6, 4, dtype=torch.float32),
    )

    assert tuple(name for name, _ in linear.named_parameters()) == parameter_names_before
    assert tuple(linear.state_dict()) == state_dict_names_before
    assert parameter_names_before == ("weight",)


@pytest.mark.parametrize(
    ("replica_weight_shape", "replica_grad_shape", "replica_grad_dtype", "match"),
    [
        ((1, 1, 5, 4), (1, 1, 5, 4), torch.float32, "Unexpected UltraEP replica weight shape"),
        ((1, 1, 6, 4), (1, 2, 6, 4), torch.float32, "FP32 or BF16 tensor matching replica weight shape"),
        ((1, 1, 6, 4), (1, 1, 6, 4), torch.float16, "FP32 or BF16 tensor matching replica weight shape"),
    ],
)
def test_ultra_ep_rejects_invalid_replica_buffers(
    replica_weight_shape,
    replica_grad_shape,
    replica_grad_dtype,
    match,
):
    linear = GroupedLinear(4, 6, 2)

    with pytest.raises(ValueError, match=match):
        linear.configure_ultra_ep_buffers(
            torch.empty(replica_weight_shape, dtype=torch.bfloat16),
            torch.empty(replica_grad_shape, dtype=replica_grad_dtype),
        )


def test_ultra_ep_rejects_replica_buffers_when_expert_bias_is_enabled():
    linear = GroupedLinear(4, 6, 2, moe_bias=True)

    with pytest.raises(NotImplementedError, match="expert bias"):
        linear.configure_ultra_ep_buffers(
            torch.empty(1, 1, 6, 4, dtype=torch.bfloat16),
            torch.empty(1, 1, 6, 4, dtype=torch.float32),
        )


def test_ultra_ep_layer_runtime_configures_buffers_and_refreshes_weight_pointers():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()

    runtime.sync_weights(7, async_finish=True)
    assert len(fused_w1w3.configure_calls) == 1
    assert len(fused_w2.configure_calls) == 1
    assert fused_w1w3.configure_calls[0][0].shape == (2, 1, 6, 4)
    assert fused_w1w3.configure_calls[0][1].dtype == torch.float32
    assert fused_w2.configure_calls[0][0].shape == (2, 1, 4, 3)
    assert fused_w2.configure_calls[0][1].dtype == torch.float32
    assert len(manager.register_calls) == 1
    assert manager.register_calls[0]["layer_id"] == 2
    assert manager.refresh_calls[0]["fc1_weight"] is fused_w1w3.weight
    assert manager.refresh_calls[0]["fc2_weight"] is fused_w2.weight

    runtime.sync_weights(8, async_finish=False)
    assert len(fused_w1w3.configure_calls) == 1
    assert len(fused_w2.configure_calls) == 1
    assert len(manager.register_calls) == 1
    assert [call["layer_id"] for call in manager.refresh_calls] == [2, 2]
    assert manager.weight_sync_calls == [(7, True), (8, False)]

    runtime.bind_virtual_layer_slot(7)
    assert fused_w1w3.select_calls[-1] == 1
    assert fused_w2.select_calls[-1] == 1


def test_ultra_ep_weight_restore_launch_and_join_are_async():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()

    runtime.start_weight_restore(7)
    assert manager.weight_sync_calls == [(7, True)]
    assert fused_w1w3.select_calls == []
    assert fused_w2.select_calls == []

    runtime.finish_weight_restore(7)
    assert manager.event_calls == [("wait", 7)]
    assert fused_w1w3.select_calls == [1]
    assert fused_w2.select_calls == [1]
    assert runtime._weight_restore_events == {}


def test_ultra_ep_weight_restore_autograd_nodes_launch_before_join():
    runtime = object.__new__(UltraEPLayerRuntime)
    calls = []
    runtime._weight_restore_events = {11: object()}

    def start_weight_restore(virtual_layer_id):
        calls.append(("start", virtual_layer_id))

    def finish_weight_restore(virtual_layer_id):
        calls.append(("finish", virtual_layer_id))
        runtime._weight_restore_events.pop(virtual_layer_id)

    runtime.start_weight_restore = start_weight_restore
    runtime.finish_weight_restore = finish_weight_restore

    x = torch.ones(2, requires_grad=True)
    joined = _UltraEPWeightRestoreJoin.apply(x, runtime, 11)
    started = _UltraEPWeightRestoreStart.apply(joined, runtime, 11)
    started.sum().backward()

    assert calls == [("start", 11), ("finish", 11)]
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ultra_ep_grad_reduce_lifecycle_stages_reduces_and_restores():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()

    with pytest.raises(RuntimeError, match="not started"):
        runtime.finish_grad_reduce(3)

    with pytest.raises(RuntimeError, match="master gradients are unavailable"):
        runtime.start_grad_reduce(3)

    fused_w1w3.weight.grad = torch.full_like(fused_w1w3.weight, 1.0)
    fused_w2.weight.grad = torch.full_like(fused_w2.weight, 3.0)

    runtime.start_grad_reduce(3)
    assert manager.stage_calls == [3]
    assert manager.grad_reduce_calls == [(3, True)]

    runtime.finish_grad_reduce(3)
    assert manager.event_calls == [("wait", 3)]
    assert manager.restore_calls == [3]
    assert fused_w1w3.weight.grad.data_ptr() == manager.master_fc1_grad_staging.data_ptr()
    assert fused_w2.weight.grad.data_ptr() == manager.master_fc2_grad_staging.data_ptr()
    torch.testing.assert_close(fused_w1w3.weight.grad, torch.full_like(fused_w1w3.weight, 2.0))
    torch.testing.assert_close(fused_w2.weight.grad, torch.full_like(fused_w2.weight, 5.0))
    assert runtime._grad_reduce_events == {}


def test_ultra_ep_multi_microbatch_grad_reduce_remains_async():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()
    runtime.configure_max_microbatches(2)
    fused_w1w3.weight.grad = torch.full_like(fused_w1w3.weight, 1.0)
    fused_w2.weight.grad = torch.full_like(fused_w2.weight, 3.0)

    runtime.start_grad_reduce(17)

    assert manager.grad_reduce_calls == [(17, True)]
    assert manager.restore_calls == []
    runtime.finish_grad_reduce(17)
    assert manager.event_calls == [("wait", 17)]
    assert manager.restore_calls == [17]
    assert runtime._grad_reduce_events == {}


def test_ultra_ep_grad_reduce_writes_back_through_fsdp_binding(monkeypatch):
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()
    fused_w1w3.weight.grad = torch.full_like(fused_w1w3.weight, 1.0)
    fused_w2.weight.grad = torch.full_like(fused_w2.weight, 3.0)
    fused_w1w3._xtuner_ultraep_fsdp_param = object()
    fused_w2._xtuner_ultraep_fsdp_param = object()
    current_w1 = torch.nn.Parameter(torch.zeros_like(fused_w1w3.weight))
    current_w2 = torch.nn.Parameter(torch.zeros_like(fused_w2.weight))
    current_w1.grad = torch.full_like(current_w1, 1.0)
    current_w2.grad = torch.full_like(current_w2, 3.0)

    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    monkeypatch.setattr(
        binding,
        "fsdp_current_unsharded_expert_parameters",
        lambda projections: (current_w1, current_w2),
    )

    runtime.start_grad_reduce(19)
    runtime.finish_grad_reduce(19)

    assert current_w1.grad.data_ptr() == manager.master_fc1_grad_staging.data_ptr()
    assert current_w2.grad.data_ptr() == manager.master_fc2_grad_staging.data_ptr()
    torch.testing.assert_close(current_w1.grad, torch.full_like(current_w1, 2.0))
    torch.testing.assert_close(current_w2.grad, torch.full_like(current_w2, 5.0))
    torch.testing.assert_close(fused_w1w3.weight.grad, torch.full_like(fused_w1w3.weight, 1.0))
    torch.testing.assert_close(fused_w2.weight.grad, torch.full_like(fused_w2.weight, 3.0))


def test_ultra_ep_grad_reduce_autograd_nodes_start_before_join():
    runtime = object.__new__(UltraEPLayerRuntime)
    calls = []

    def start_grad_reduce(virtual_layer_id):
        calls.append(("start", virtual_layer_id))

    def finish_grad_reduce(virtual_layer_id):
        calls.append(("finish", virtual_layer_id))

    runtime.start_grad_reduce = start_grad_reduce
    runtime.finish_grad_reduce = finish_grad_reduce

    x = torch.ones(2, requires_grad=True)
    joined = _UltraEPGradReduceJoin.apply(x, runtime, 11)
    output = _UltraEPGradReduceStart.apply(joined, runtime, 11)
    output.sum().backward()

    assert calls == [("start", 11), ("finish", 11)]
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ultra_ep_output_wrapper_restores_replica_weight_before_te_grouped_gemm_backward(monkeypatch):
    """Later layers overwrite the shared replica slot after this forward.

    ``_UltraEPWeightRestoreJoin`` must restore it before TE DGrad reads the
    same tensor. The torch TE backend keeps this test off CUDA.
    """

    monkeypatch.setenv("XTUNER_GROUP_GEMM", "te")
    monkeypatch.setenv("XTUNER_TE_GEMM_BACKEND", "torch")

    import xtuner.v1.ops.moe.cuda.group_gemm_te as adapter
    from xtuner.v1.ops.moe.cuda.group_gemm_te import te_grouped_gemm

    replica_seen_in_gemm = []
    real_physical_weights = adapter._physical_weights

    def spy_physical_weights(master_weight, replica_weight):
        if replica_weight is not None:
            replica_seen_in_gemm.append(replica_weight.detach().clone())
        return real_physical_weights(master_weight, replica_weight)

    monkeypatch.setattr(adapter, "_physical_weights", spy_physical_weights)

    x = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    master_weight = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    replica_weight = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    original_replica_weight = replica_weight.clone()
    replica_ptr = replica_weight.data_ptr()
    replica_grad = torch.empty_like(replica_weight, dtype=torch.float32)
    tokens_per_expert = torch.tensor([1, 1])

    runtime = object.__new__(UltraEPLayerRuntime)
    runtime._weight_restore_events = {}
    sync_calls = []

    def fake_sync_weights(virtual_layer_id, *, async_finish):
        sync_calls.append((virtual_layer_id, async_finish))
        replica_weight.copy_(original_replica_weight)

        class _RestoreEvent:
            def current_stream_wait(self):
                return None

        return _RestoreEvent()

    runtime.sync_weights = fake_sync_weights
    runtime.bind_virtual_layer_slot = lambda virtual_layer_id: None
    runtime.start_weight_restore = UltraEPLayerRuntime.start_weight_restore.__get__(
        runtime, UltraEPLayerRuntime
    )
    runtime.finish_weight_restore = UltraEPLayerRuntime.finish_weight_restore.__get__(
        runtime, UltraEPLayerRuntime
    )

    output = te_grouped_gemm(
        x,
        master_weight,
        tokens_per_expert,
        replica_weight=replica_weight,
        replica_grad=replica_grad,
    )
    # UltraEP reuses its persistent slots for the next layer before this
    # layer's backward. Start launches the restore; Join waits before DGrad.
    replica_weight.fill_(100.0)
    joined = _UltraEPWeightRestoreJoin.apply(output, runtime, 7)
    _UltraEPWeightRestoreStart.apply(joined, runtime, 7).sum().backward()

    assert sync_calls == [(7, True)]
    assert replica_weight.data_ptr() == replica_ptr
    assert len(replica_seen_in_gemm) == 2
    torch.testing.assert_close(replica_seen_in_gemm[0], original_replica_weight)
    torch.testing.assert_close(replica_seen_in_gemm[1], original_replica_weight)
    torch.testing.assert_close(x.grad, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    torch.testing.assert_close(master_weight.grad, torch.tensor([[[1.0, 1.0], [1.0, 1.0]]]))
    torch.testing.assert_close(replica_grad, torch.tensor([[[2.0, 1.0], [2.0, 1.0]]]))


def test_dual_base_group_gemm_restores_replica_weight_before_backward(monkeypatch):
    """The dual-base DGrad must read the replica slot restored by its hook."""

    dual_gemm_calls = []

    def fake_dual_gemm(x, master_weight, replica_weight, tokens_per_expert, *, trans_b):
        dual_gemm_calls.append((trans_b, replica_weight.detach().clone()))
        weight = torch.cat((master_weight, replica_weight), dim=0)
        chunks = []
        offset = 0
        for expert_idx, count in enumerate(tokens_per_expert.tolist()):
            x_chunk = x[offset : offset + count]
            rhs = weight[expert_idx].T if trans_b else weight[expert_idx]
            chunks.append(x_chunk @ rhs)
            offset += count
        return torch.cat(chunks)

    def fake_k_grouped_gemm(grad_output, x, tokens_per_expert):
        chunks = []
        offset = 0
        for count in tokens_per_expert.tolist():
            grad_chunk = grad_output[offset : offset + count]
            x_chunk = x[offset : offset + count]
            chunks.append(grad_chunk.T @ x_chunk)
            offset += count
        return torch.stack(chunks)

    monkeypatch.setattr(group_gemm_module, "m_grouped_gemm_dual_weight", fake_dual_gemm)
    monkeypatch.setattr(group_gemm_module, "k_grouped_gemm", fake_k_grouped_gemm)

    x = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    master_weight = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    replica_weight = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    original_replica_weight = replica_weight.clone()
    replica_grad = torch.empty_like(replica_weight, dtype=torch.float32)
    tokens_per_expert = torch.tensor([1, 1])

    runtime = object.__new__(UltraEPLayerRuntime)
    runtime._weight_restore_events = {}
    hook_calls = []

    def fake_sync_weights(virtual_layer_id, *, async_finish):
        hook_calls.append(("sync", virtual_layer_id, async_finish))
        replica_weight.copy_(original_replica_weight)

        class _RestoreEvent:
            def current_stream_wait(self):
                hook_calls.append(("wait", virtual_layer_id))

        return _RestoreEvent()

    runtime.bind_virtual_layer_slot = lambda virtual_layer_id: hook_calls.append(("bind", virtual_layer_id))
    runtime.sync_weights = fake_sync_weights
    runtime.start_weight_restore = UltraEPLayerRuntime.start_weight_restore.__get__(
        runtime, UltraEPLayerRuntime
    )
    runtime.finish_weight_restore = UltraEPLayerRuntime.finish_weight_restore.__get__(
        runtime, UltraEPLayerRuntime
    )

    output = group_gemm_module.triton_group_gemm(
        x,
        master_weight,
        tokens_per_expert,
        replica_weight=replica_weight,
        replica_grad=replica_grad,
    )
    replica_weight.fill_(100.0)
    joined = _UltraEPWeightRestoreJoin.apply(output, runtime, 7)
    _UltraEPWeightRestoreStart.apply(joined, runtime, 7).sum().backward()

    assert hook_calls == [("sync", 7, True), ("wait", 7), ("bind", 7)]
    assert [call[0] for call in dual_gemm_calls] == [True, False]
    torch.testing.assert_close(dual_gemm_calls[0][1], original_replica_weight)
    torch.testing.assert_close(dual_gemm_calls[1][1], original_replica_weight)
    torch.testing.assert_close(x.grad, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    torch.testing.assert_close(master_weight.grad, torch.tensor([[[1.0, 1.0], [1.0, 1.0]]]))
    torch.testing.assert_close(replica_grad, torch.tensor([[[2.0, 1.0], [2.0, 1.0]]]))


def test_dual_base_group_gemm_supports_empty_local_dispatch():
    x = torch.empty(0, 2, requires_grad=True)
    master_weight = torch.ones(1, 3, 2, requires_grad=True)
    replica_weight = torch.ones(1, 3, 2)
    replica_grad = torch.full_like(replica_weight, torch.nan, dtype=torch.float32)

    output = group_gemm_module.triton_group_gemm(
        x,
        master_weight,
        torch.tensor([0, 0]),
        replica_weight=replica_weight,
        replica_grad=replica_grad,
    )
    output.sum().backward()

    assert output.shape == (0, 3)
    torch.testing.assert_close(x.grad, torch.empty_like(x))
    torch.testing.assert_close(master_weight.grad, torch.zeros_like(master_weight))
    torch.testing.assert_close(replica_grad, torch.zeros_like(replica_grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("trans_b", [True, False])
def test_dual_base_kernel_matches_contiguous_reference(trans_b):
    torch.manual_seed(0)
    counts = torch.tensor([128, 128, 128], device="cuda", dtype=torch.int64)
    num_master, num_replica, n, k = 2, 1, 256, 256
    x = torch.randn(int(counts.sum()), k, device="cuda", dtype=torch.bfloat16)
    if trans_b:
        master = torch.randn(num_master, n, k, device="cuda", dtype=torch.bfloat16)
        replica = torch.randn(num_replica, n, k, device="cuda", dtype=torch.bfloat16)
    else:
        master = torch.randn(num_master, k, n, device="cuda", dtype=torch.bfloat16)
        replica = torch.randn(num_replica, k, n, device="cuda", dtype=torch.bfloat16)

    actual = group_gemm_module.m_grouped_gemm_dual_weight(
        x,
        master,
        replica,
        counts,
        trans_b=trans_b,
    )
    expected = group_gemm_module.m_grouped_gemm(
        x,
        torch.cat((master, replica), dim=0),
        counts,
        trans_b=trans_b,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("counts_values", [(64, 64, 64), (0, 17, 33)])
@pytest.mark.parametrize("shape", [(64, 64), (96, 64)])
def test_dual_weight_group_gemm_backward_matches_contiguous_reference(counts_values, shape):
    torch.manual_seed(0)
    device = torch.device("cuda")
    counts = torch.tensor(counts_values, device=device, dtype=torch.int64)
    out_features, in_features = shape
    master = torch.randn(2, out_features, in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
    replica = torch.randn(1, out_features, in_features, device=device, dtype=torch.bfloat16)
    replica_grad = torch.full_like(replica, torch.nan, dtype=torch.float32)
    x = torch.randn(int(counts.sum()), in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
    actual = group_gemm_module.triton_group_gemm(
        x, master, counts, replica_weight=replica, replica_grad=replica_grad
    )
    ref_x = x.detach().clone().requires_grad_(True)
    ref_master = master.detach().clone().requires_grad_(True)
    ref_replica = replica.detach().clone().requires_grad_(True)
    expected = group_gemm_module.triton_group_gemm(ref_x, torch.cat((ref_master, ref_replica)), counts)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(master.grad, ref_master.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(replica_grad, ref_replica.grad.float(), rtol=2e-2, atol=2e-2)
