import pytest
import torch

from xtuner.v1.config import FSDPConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    _UltraEPGradReduceJoin,
    _UltraEPGradReduceStart,
    _UltraEPWeightSyncForBackward,
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
        self.weight = torch.nn.Parameter(torch.zeros(shape))
        self.configure_calls = []

    def configure_ultra_ep_buffers(self, replica_weight, replica_grad):
        self.configure_calls.append((replica_weight, replica_grad))


class FakeEvent:
    def __init__(self, calls, virtual_layer_id):
        self.calls = calls
        self.virtual_layer_id = virtual_layer_id

    def current_stream_wait(self):
        self.calls.append(("wait", self.virtual_layer_id))


class FakeLayerManager:
    def __init__(self, *, num_master_experts=2, redundant=1, hidden_size=4, intermediate_size=3):
        self.num_local_redundant_experts = redundant
        self.local_replica_fc1_weight_buffer = torch.empty(redundant * 2 * intermediate_size * hidden_size)
        self.local_replica_fc2_weight_buffer = torch.empty(redundant * hidden_size * intermediate_size)
        self.local_replica_fc1_grad_buffer = torch.empty(
            redundant * 2 * intermediate_size * hidden_size,
            dtype=torch.float32,
        )
        self.local_replica_fc2_grad_buffer = torch.empty(
            redundant * hidden_size * intermediate_size,
            dtype=torch.float32,
        )
        self.master_fc1_grad_staging = torch.empty(num_master_experts, 2 * intermediate_size, hidden_size)
        self.master_fc2_grad_staging = torch.empty(num_master_experts, hidden_size, intermediate_size)
        self.register_calls = []
        self.refresh_calls = []
        self.weight_sync_calls = []
        self.stage_calls = []
        self.grad_reduce_calls = []
        self.restore_calls = []
        self.event_calls = []

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
        self.restore_calls.append(virtual_layer_id)
        fc1_grad.copy_(self.master_fc1_grad_staging)
        fc2_grad.copy_(self.master_fc2_grad_staging)


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
    assert provider.num_dispatch_experts == config.n_routed_experts + 8 * config.ultraep_cfg.num_redundant_experts_per_rank
    assert provider._manager is None

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


def test_ultra_ep_buffers_are_not_parameters_or_state_dict_entries():
    linear = GroupedLinear(4, 6, 2)
    parameter_names_before = tuple(name for name, _ in linear.named_parameters())
    state_dict_names_before = tuple(linear.state_dict())

    linear.configure_ultra_ep_buffers(
        torch.empty(1, 6, 4, dtype=torch.bfloat16),
        torch.empty(1, 6, 4, dtype=torch.float32),
    )

    assert tuple(name for name, _ in linear.named_parameters()) == parameter_names_before
    assert tuple(linear.state_dict()) == state_dict_names_before
    assert parameter_names_before == ("weight",)


@pytest.mark.parametrize(
    ("replica_weight_shape", "replica_grad_shape", "replica_grad_dtype", "match"),
    [
        ((1, 5, 4), (1, 5, 4), torch.float32, "Unexpected UltraEP replica weight shape"),
        ((1, 6, 4), (2, 6, 4), torch.float32, "FP32 tensor matching replica weight shape"),
        ((1, 6, 4), (1, 6, 4), torch.bfloat16, "FP32 tensor matching replica weight shape"),
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
            torch.empty(1, 6, 4, dtype=torch.bfloat16),
            torch.empty(1, 6, 4, dtype=torch.float32),
        )


def test_ultra_ep_layer_runtime_configures_buffers_and_refreshes_weight_pointers():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()

    runtime.sync_weights(7, async_finish=True)
    assert len(fused_w1w3.configure_calls) == 1
    assert len(fused_w2.configure_calls) == 1
    assert fused_w1w3.configure_calls[0][0].shape == (1, 6, 4)
    assert fused_w1w3.configure_calls[0][1].dtype == torch.float32
    assert fused_w2.configure_calls[0][0].shape == (1, 4, 3)
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
    torch.testing.assert_close(fused_w1w3.weight.grad, torch.full_like(fused_w1w3.weight, 2.0))
    torch.testing.assert_close(fused_w2.weight.grad, torch.full_like(fused_w2.weight, 5.0))
    assert runtime._grad_reduce_events == {}


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


def test_ultra_ep_output_wrapper_restores_replica_weight_before_group_gemm_backward(monkeypatch):
    dual_gemm_calls = []

    def fake_m_grouped_gemm_dual_weight(x, master_weight, replica_weight, tokens_per_expert, *, trans_b):
        dual_gemm_calls.append(
            (master_weight.shape[0], replica_weight.shape[0], trans_b, replica_weight.data_ptr())
        )
        weight = torch.cat((master_weight, replica_weight), dim=0)
        chunks = []
        offset = 0
        for expert_idx, count in enumerate(tokens_per_expert.tolist()):
            x_chunk = x[offset : offset + count]
            chunks.append(x_chunk @ (weight[expert_idx].T if trans_b else weight[expert_idx]))
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

    monkeypatch.setattr(group_gemm_module, "m_grouped_gemm_dual_weight", fake_m_grouped_gemm_dual_weight)
    monkeypatch.setattr(group_gemm_module, "k_grouped_gemm", fake_k_grouped_gemm)

    x = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    master_weight = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    replica_weight = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    original_replica_weight = replica_weight.clone()
    replica_grad = torch.empty_like(replica_weight, dtype=torch.float32)
    tokens_per_expert = torch.tensor([1, 1])

    # The production wrapper invokes Manager.weight_sync during backward.
    # A bare runtime instance is sufficient to validate autograd ordering.
    runtime = object.__new__(UltraEPLayerRuntime)
    sync_calls = []

    def fake_sync_weights(virtual_layer_id, *, async_finish):
        sync_calls.append((virtual_layer_id, async_finish))
        replica_weight.copy_(original_replica_weight)

    runtime.sync_weights = fake_sync_weights

    output = group_gemm_module.ultra_ep_group_gemm(
        x,
        master_weight,
        replica_weight,
        replica_grad,
        tokens_per_expert,
    )
    # UltraEP reuses its persistent slots for the next layer before this
    # layer's backward.  The output-side node restores them before DGrad.
    replica_weight.fill_(100.0)
    _UltraEPWeightSyncForBackward.apply(output, runtime, 7).sum().backward()

    assert sync_calls == [(7, False)]
    torch.testing.assert_close(x.grad, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    torch.testing.assert_close(master_weight.grad, torch.tensor([[[1.0, 1.0], [1.0, 1.0]]]))
    torch.testing.assert_close(replica_grad, torch.tensor([[[2.0, 1.0], [2.0, 1.0]]]))
    assert dual_gemm_calls == [
        (1, 1, True, replica_weight.data_ptr()),
        (1, 1, False, replica_weight.data_ptr()),
    ]


def test_ultra_ep_group_gemm_supports_empty_local_dispatch():
    x = torch.empty(0, 2, requires_grad=True)
    master_weight = torch.ones(1, 3, 2, requires_grad=True)
    replica_weight = torch.ones(1, 3, 2)
    replica_grad = torch.full_like(replica_weight, torch.nan, dtype=torch.float32)

    output = group_gemm_module.ultra_ep_group_gemm(
        x,
        master_weight,
        replica_weight,
        replica_grad,
        torch.tensor([0, 0]),
    )
    output.sum().backward()

    assert output.shape == (0, 3)
    torch.testing.assert_close(x.grad, torch.empty_like(x))
    torch.testing.assert_close(master_weight.grad, torch.zeros_like(master_weight))
    torch.testing.assert_close(replica_grad, torch.zeros_like(replica_grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("trans_b", [True, False])
def test_dual_weight_group_gemm_matches_contiguous_reference(trans_b):
    torch.manual_seed(0)
    device = torch.device("cuda")
    counts = torch.tensor([128, 128, 128], device=device, dtype=torch.int64)
    num_master, num_replica, n, k = 2, 1, 256, 256
    x = torch.randn(int(counts.sum()), k, device=device, dtype=torch.bfloat16)
    if trans_b:
        master = torch.randn(num_master, n, k, device=device, dtype=torch.bfloat16)
        replica = torch.randn(num_replica, n, k, device=device, dtype=torch.bfloat16)
    else:
        master = torch.randn(num_master, k, n, device=device, dtype=torch.bfloat16)
        replica = torch.randn(num_replica, k, n, device=device, dtype=torch.bfloat16)

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
    num_master, num_replica = 2, 1

    x = torch.randn(int(counts.sum()), in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
    master = torch.randn(
        num_master,
        out_features,
        in_features,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    replica = torch.randn(num_replica, out_features, in_features, device=device, dtype=torch.bfloat16)
    replica_grad = torch.full_like(replica, torch.nan, dtype=torch.float32)

    actual = group_gemm_module.ultra_ep_group_gemm(
        x,
        master,
        replica,
        replica_grad,
        counts,
    )

    ref_x = x.detach().clone().requires_grad_(True)
    ref_master = master.detach().clone().requires_grad_(True)
    ref_replica = replica.detach().clone().requires_grad_(True)
    expected = group_gemm_module.triton_group_gemm(
        ref_x,
        torch.cat((ref_master, ref_replica), dim=0),
        counts,
    )

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(master.grad, ref_master.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(replica_grad, ref_replica.grad.float(), rtol=2e-2, atol=2e-2)
