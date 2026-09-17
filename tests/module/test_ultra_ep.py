import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xtuner.v1.config import FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.moe.moe import MoE
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
from xtuner.v1.module.dispatcher import ExpertWeightLayout, NoEPExecutionRuntime, build_ep_execution_runtime
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.ultraep import UltraEPConfig
from xtuner.v1.module.ultraep import runtime as ultraep_runtime
from xtuner.v1.module.ultraep.dispatcher import (
    UltraEPDispatcher,
    _UltraEPGradReduceJoin,
    _UltraEPGradReduceStart,
    _UltraEPWeightRestoreJoin,
    _UltraEPWeightRestoreStart,
)
from xtuner.v1.module.ultraep.runtime import UltraEPLayerRuntime, UltraEPModelRuntime
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
        self._ultra_ep_replica_weight = None
        self._ultra_ep_replica_grad = None
        self._ultra_ep_replica_weight_slots = None
        self._ultra_ep_replica_grad_slots = None

    def configure_ultra_ep_buffers(self, replica_weight, replica_grad):
        self.configure_calls.append((replica_weight, replica_grad))
        self._ultra_ep_replica_weight_slots = replica_weight
        self._ultra_ep_replica_grad_slots = replica_grad
        self.select_ultra_ep_slot(0)

    def select_ultra_ep_slot(self, slot):
        self.select_calls.append(slot)
        if self._ultra_ep_replica_weight_slots is None:
            return
        self._ultra_ep_replica_weight = self._ultra_ep_replica_weight_slots[slot]
        self._ultra_ep_replica_grad = self._ultra_ep_replica_grad_slots[slot]


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
        self.allocate_calls = []
        self.placement_calls = []
        self.reroute_calls = []
        self.control_calls = []
        self._next_vid = 0

    def replica_slot(self, virtual_layer_id):
        return virtual_layer_id // self.real_num_alloc_layers

    def allocate_microbatch_slot(self, layer_id):
        vid = self._next_vid
        self._next_vid += 1
        self.allocate_calls.append((layer_id, vid))
        return vid

    def update_placement_sparse(self, layer_id, logical_topk_ids):
        self.control_calls.append("placement")
        self.placement_calls.append((layer_id, logical_topk_ids.clone()))

    def reroute_sparse(self, layer_id, physical_topk_ids):
        self.control_calls.append("reroute")
        self.reroute_calls.append(layer_id)
        physical_topk_ids.add_(1)

    def register_master_pointers(self, **kwargs):
        self.register_calls.append(kwargs)

    def refresh_master_weight_pointers(self, **kwargs):
        self.refresh_calls.append(kwargs)

    def weight_sync(self, layer_id, *, async_finish):
        self.control_calls.append("sync")
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


class FakeModelRuntime:
    num_model_layers = 4
    num_logical_experts = 16
    hidden_size = 4
    expert_intermediate_size = 3
    num_redundant_experts_per_rank = 1
    max_microbatches = 1
    inner_dispatcher = "deepep"
    training_dtype = "bf16"
    generate_dtype = "bf16"

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
    model_runtime = FakeModelRuntime(manager)
    fused_w1w3 = FakeGroupedLinear((2, 6, 4))
    fused_w2 = FakeGroupedLinear((2, 4, 3))
    runtime = UltraEPLayerRuntime(
        layer_id=2,
        model_runtime=model_runtime,  # type: ignore[arg-type]
        fused_w1w3=fused_w1w3,
        fused_w2=fused_w2,
    )
    return runtime, manager, fused_w1w3, fused_w2


def test_ultra_ep_is_opt_in():
    config = Qwen3MoE235BA22Config()

    assert config.ultraep_cfg is None


def test_disabled_ultra_ep_does_not_import_optional_backend():
    repo = Path(__file__).resolve().parents[2]
    source = """
import sys
sys.modules["ultra_ep"] = None
from xtuner.v1.module.dispatcher import (
    NaiveDispatcher,
    NoEPExecutionRuntime,
    build_dispatcher,
    build_ep_execution_runtime,
)
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config

dispatcher = build_dispatcher(None, n_routed_experts=4)
assert isinstance(dispatcher, NaiveDispatcher)
runtime = build_ep_execution_runtime(Qwen3MoE235BA22Config(), None)
assert type(runtime) is NoEPExecutionRuntime
assert sys.modules.get("ultra_ep") is None
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")])
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
    )
    assert completed.returncode == 0, completed.stderr


def test_enabled_ultra_ep_reports_missing_optional_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultra_ep" or name.startswith("ultra_ep."):
            raise ImportError("No module named 'ultra_ep'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    sys.modules.pop("ultra_ep", None)

    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=Qwen3MoE235BA22Config(
            ep_size=8,
            ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
        ),
    )
    runtime.bind_layer(
        layer_id=0,
        projections=(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)),
    )
    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    monkeypatch.setattr(binding, "install_ultraep_fsdp_binding", lambda **kwargs: ())
    runtime.install_after_fsdp(fsdp_root=nn.Linear(1, 1), execution_order=[])

    with pytest.raises(ImportError, match="Python package/CUDA extension is unavailable"):
        runtime.get_manager()


def test_ultra_ep_fsdp_private_api_is_isolated_to_binding_module():
    ultraep_dir = Path(__file__).resolve().parents[2] / "xtuner" / "v1" / "module" / "ultraep"
    users = [
        path.name
        for path in ultraep_dir.glob("*.py")
        if "torch.distributed.fsdp._fully_shard" in path.read_text()
    ]

    assert users == ["fsdp_expert_binding.py"]


def test_ultra_ep_requires_redundant_experts():
    with pytest.raises(ValueError, match="num_redundant_experts_per_rank"):
        UltraEPConfig(num_redundant_experts_per_rank=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_routed_experts": 33}, "n_routed_experts"),
        ({"ep_size": 1}, "ep_size"),
        (
            {"float8_cfg": Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE)},
            "BF16",
        ),
        ({"moe_bias": True}, "expert bias"),
        ({"expert_tp_size": 2}, "expert_tp_size == 1"),
        ({"mtp_config": MTPConfig(num_layers=1)}, "MTP expert layers"),
        ({"intra_layer_micro_batch": 0}, "intra_layer_micro_batch must be positive"),
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


def test_ultra_ep_rejects_unimplemented_inner_dispatcher():
    config = Qwen3MoE235BA22Config(
        ep_size=8,
        n_routed_experts=32,
        dispatcher="all2all",
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )

    with pytest.raises(NotImplementedError, match="inner dispatcher"):
        config.build()


def test_ultra_ep_rejects_activation_recompute():
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=Qwen3MoE235BA22Config(
            ep_size=8,
            ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
        ),
    )

    with pytest.raises(ValueError, match="activation recompute"):
        runtime.validate_before_fsdp(FSDPConfig(ep_size=8, recompute_ratio=0.5))


class _CaptureDecoderLayer(nn.Module):
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).last_kwargs = kwargs


def test_moe_init_creates_and_passes_ultraep_runtime(monkeypatch):
    class FakeMesh:
        def __getitem__(self, name):
            return self

        def get_group(self):
            return FakeGroup()

    _CaptureDecoderLayer.last_kwargs = None
    created = []
    real_from_config = UltraEPModelRuntime.from_xtuner_config

    @classmethod
    def _count(cls, **kwargs):
        runtime = real_from_config(**kwargs)
        created.append(runtime)
        return runtime

    monkeypatch.setattr(UltraEPModelRuntime, "from_xtuner_config", _count)
    monkeypatch.setattr("xtuner.v1.model.moe.moe.dist.get_world_size", lambda: 8)
    monkeypatch.setattr("xtuner.v1.model.moe.moe.init_device_mesh", lambda *args, **kwargs: FakeMesh())
    monkeypatch.setattr(MoE, "moe_decoder_layer_cls", _CaptureDecoderLayer)
    monkeypatch.setattr(MoE, "build_rotary_embedding", lambda self, config: nn.Identity())
    monkeypatch.setattr(MoE, "build_embeddings", lambda self, config: nn.Embedding(32, 8))
    monkeypatch.setattr(MoE, "_init_load_spec", lambda self: None)
    monkeypatch.setattr(MoE, "_maybe_enable_compile", lambda self, cfg: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: object())

    config = Qwen3MoE235BA22Config(
        ep_size=8,
        n_routed_experts=32,
        dispatcher="deepep",
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
        num_hidden_layers=1,
        hidden_size=8,
        vocab_size=32,
        intermediate_size=16,
        moe_intermediate_size=16,
    )
    model = MoE(config)
    assert len(created) == 1
    assert model._ep_runtime is created[0]

    assert isinstance(model._ep_runtime, UltraEPModelRuntime)
    assert model._ep_runtime.num_logical_experts == 32
    assert model._ep_runtime.num_redundant_experts_per_rank == 1
    assert _CaptureDecoderLayer.last_kwargs is not None
    assert _CaptureDecoderLayer.last_kwargs["ep_runtime"] is model._ep_runtime
    assert _CaptureDecoderLayer.last_kwargs["layer_idx"] == 0
    assert "layer_fqn" not in _CaptureDecoderLayer.last_kwargs
    assert _CaptureDecoderLayer.last_kwargs["ep_mesh"] is model.ep_mesh
    assert not hasattr(model, "ultraep_manager_provider")
    assert not hasattr(model._ep_runtime, "provider")


def test_build_ep_execution_runtime_creates_ultraep_once(monkeypatch):
    created = []
    real_from_config = UltraEPModelRuntime.from_xtuner_config

    @classmethod
    def _capture(cls, **kwargs):
        runtime = real_from_config(**kwargs)
        created.append(runtime)
        return runtime

    monkeypatch.setattr(UltraEPModelRuntime, "from_xtuner_config", _capture)

    class FakeMesh:
        def get_group(self):
            return FakeGroup()

    config = Qwen3MoE235BA22Config(
        ep_size=8,
        n_routed_experts=32,
        dispatcher="deepep",
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = build_ep_execution_runtime(config, FakeMesh())
    assert runtime is created[0]
    assert isinstance(runtime, UltraEPModelRuntime)
    assert build_ep_execution_runtime(Qwen3MoE235BA22Config(), None).__class__ is NoEPExecutionRuntime


def test_ultra_ep_model_runtime_bind_layer_returns_dispatcher():
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    w1 = nn.Linear(4, 4, bias=False)
    w2 = nn.Linear(4, 4, bias=False)

    dispatcher = runtime.bind_layer(layer_id=0, projections=(w1, w2))
    assert isinstance(dispatcher, UltraEPDispatcher)
    assert dispatcher._layer.layer_id == 0
    assert dispatcher._layer.model_runtime is runtime
    assert dispatcher._inner is None
    assert runtime._layers == [(0, (w1, w2))]
    with pytest.raises(ValueError, match="duplicate UltraEP expert projections"):
        runtime.bind_layer(layer_id=0, projections=(w1, w2))
    with pytest.raises(ValueError, match="needs a model layer id"):
        runtime.bind_layer(projections=(w1, w2))


def test_ultra_ep_install_without_fsdp_keeps_created_state():
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=Qwen3MoE235BA22Config(
            ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
        ),
    )
    experts = nn.Module()
    experts.fused_w1w3 = nn.Linear(4, 4, bias=False)
    experts.fused_w2 = nn.Linear(4, 4, bias=False)
    runtime.bind_layer(layer_id=0, projections=(experts.fused_w1w3, experts.fused_w2))

    with pytest.raises(RuntimeError, match="could not find FSDPParam"):
        runtime.install_after_fsdp(fsdp_root=experts, execution_order=[])

    assert runtime._state == "CREATED"
    assert runtime._fsdp_root is None
    assert not hasattr(runtime, "fsdp_binding")


def test_ultra_ep_get_inner_builds_named_transport(monkeypatch):
    config = Qwen3MoE235BA22Config(
        dispatcher="deepep",
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    dispatcher = runtime.bind_layer(
        layer_id=0,
        projections=(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)),
    )
    captured: dict[str, object] = {}

    def _fake_transport(name, n_routed_experts, **kwargs):
        captured["name"] = name
        captured["n_routed_experts"] = n_routed_experts
        captured["ep_group"] = kwargs.get("ep_group")
        captured["training_dtype"] = kwargs.get("training_dtype")
        return object()

    monkeypatch.setattr("xtuner.v1.module.dispatcher.build_dispatcher", _fake_transport)
    inner = dispatcher._get_inner()
    assert inner is dispatcher._inner
    assert captured["name"] == "deepep"
    assert captured["n_routed_experts"] == runtime.num_dispatch_experts
    assert captured["ep_group"] is runtime.group
    assert captured["training_dtype"] == "bf16"


def test_ultra_ep_model_runtime_install_uses_bind_layer_targets(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    w1 = nn.Linear(4, 4, bias=False)
    w2 = nn.Linear(4, 4, bias=False)
    runtime.bind_layer(layer_id=0, projections=(w1, w2))

    installed = {}

    def _install(*, fsdp_root, targets):
        installed["fsdp_root"] = fsdp_root
        installed["targets"] = targets
        return ("binding",)

    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    monkeypatch.setattr(binding, "install_ultraep_fsdp_binding", _install)
    root = object()
    runtime.install_after_fsdp(fsdp_root=root, execution_order=["layers.0.experts"])
    assert installed["fsdp_root"] is root
    assert installed["targets"] == [("layer_0", (w1, w2))]
    assert runtime._state == "INSTALLED"


def test_moe_close_ep_runtime_closes_runtime():
    class _Runtime:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    model = object.__new__(MoE)
    model._ep_runtime = _Runtime()  # type: ignore[assignment]
    model.close_ep_runtime()
    assert model._ep_runtime.closed == 1


def test_base_model_close_ep_runtime_is_noop():
    object.__new__(BaseModel).close_ep_runtime()


def test_train_engine_close_releases_ep_and_async_resources():
    calls = []

    class _Model:
        def close_ep_runtime(self):
            calls.append("ep")

        def destroy_async_hf_resources(self):
            calls.append("hf")

    engine = object.__new__(TrainEngine)
    engine.model = _Model()  # type: ignore[assignment]
    engine.destroy_async_checkpoint_pg = lambda: calls.append("ckpt")  # type: ignore[method-assign]
    engine.close()
    assert calls == ["ep", "hf", "ckpt"]


def test_ultra_ep_model_runtime_derives_shape_from_xtuner_config():
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(
            num_redundant_experts_per_rank=2,
        )
    )

    model_runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )

    assert model_runtime.num_model_layers == config.num_hidden_layers
    assert model_runtime.num_logical_experts == config.n_routed_experts
    assert model_runtime.hidden_size == config.hidden_size
    assert model_runtime.expert_intermediate_size == config.moe_intermediate_size
    assert model_runtime.num_redundant_experts_per_rank == config.ultraep_cfg.num_redundant_experts_per_rank
    assert model_runtime.inner_dispatcher == "deepep"
    assert model_runtime.training_dtype == "bf16"
    assert model_runtime.generate_dtype == "bf16"
    assert model_runtime.max_microbatches == config.intra_layer_micro_batch == 1
    assert (
        model_runtime.num_dispatch_experts
        == config.n_routed_experts + 8 * config.ultraep_cfg.num_redundant_experts_per_rank
    )
    assert model_runtime._manager is None

    with pytest.raises(RuntimeError, match="installed after FSDP setup"):
        model_runtime.get_manager()

    layer_runtime = UltraEPLayerRuntime(
        layer_id=config.num_hidden_layers - 1,
        model_runtime=model_runtime,
        fused_w1w3=object(),  # type: ignore[arg-type]
        fused_w2=object(),  # type: ignore[arg-type]
    )
    assert layer_runtime.model_runtime is model_runtime
    assert model_runtime._manager is None

    with pytest.raises(ValueError, match="layer_id"):
        UltraEPLayerRuntime(
            layer_id=config.num_hidden_layers,
            model_runtime=model_runtime,
            fused_w1w3=object(),  # type: ignore[arg-type]
            fused_w2=object(),  # type: ignore[arg-type]
        )


def test_ultra_ep_runtime_binding_install_is_transactional(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
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
    runtime.install_after_fsdp(fsdp_root=root, execution_order=[])
    assert runtime._state == "INSTALLED"
    assert runtime._fsdp_root is root
    assert runtime.fsdp_binding == ("binding",)


def test_ultra_ep_runtime_failed_binding_does_not_publish_installed_state(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    import xtuner.v1.module.ultraep.fsdp_expert_binding as binding

    def fail(**kwargs):
        raise RuntimeError("binding failed")

    monkeypatch.setattr(binding, "install_ultraep_fsdp_binding", fail)
    with pytest.raises(RuntimeError, match="binding failed"):
        runtime.install_after_fsdp(fsdp_root=object(), execution_order=[])
    assert runtime._state == "CREATED"
    assert runtime._fsdp_root is None
    assert not hasattr(runtime, "fsdp_binding")


def test_ultra_ep_runtime_requires_reshard_and_grad(monkeypatch):
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )
    monkeypatch.setattr(ultraep_runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(ultraep_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ultraep_runtime.dist, "get_world_size", lambda: 8)

    # EP8×DP1: historical subexperiment-8 path keeps params unsharded.
    runtime.validate_before_fsdp(
        FSDPConfig(recompute_ratio=0, ep_size=8, reshard_after_forward=False)
    )
    # EP4×DP2: closed unshard window is required.
    with pytest.raises(ValueError, match="reshard_after_forward"):
        runtime.validate_before_fsdp(
            FSDPConfig(recompute_ratio=0, ep_size=4, reshard_after_forward=False)
        )
    with pytest.raises(ValueError, match="reshard_after_forward"):
        runtime.validate_before_fsdp(
            FSDPConfig(recompute_ratio=0, hsdp_sharding_size=2, reshard_after_forward=False)
        )
    with pytest.raises(ValueError, match="requires_grad"):
        runtime.validate_before_fsdp(FSDPConfig(recompute_ratio=0, ep_size=8, requires_grad=False))


def test_ultra_ep_runtime_capacity_is_fixed_from_config():
    config = Qwen3MoE235BA22Config(
        ultraep_cfg=UltraEPConfig(num_redundant_experts_per_rank=1),
        intra_layer_micro_batch=2,
    )
    runtime = UltraEPModelRuntime.from_xtuner_config(
        group=FakeGroup(),  # type: ignore[arg-type]
        config=config,
    )

    assert runtime.max_microbatches == 2
    assert not hasattr(runtime, "configure_max_microbatches")

    layer_runtime = UltraEPLayerRuntime(
        layer_id=0,
        model_runtime=runtime,
        fused_w1w3=object(),  # type: ignore[arg-type]
        fused_w2=object(),  # type: ignore[arg-type]
    )
    layer_runtime.validate_microbatch_capacity(2)
    with pytest.raises(ValueError, match="exceeds the configured microbatch slots"):
        layer_runtime.validate_microbatch_capacity(3)


def test_prepare_layer_inputs_rejects_width_above_configured_slots():
    layer_runtime, _manager, _w1, _w2 = make_fake_layer_runtime()
    layer_runtime.model_runtime.group = FakeGroup()
    assert layer_runtime.model_runtime.max_microbatches == 1
    dispatcher = UltraEPDispatcher(
        model_runtime=layer_runtime.model_runtime,  # type: ignore[arg-type]
        layer_runtime=layer_runtime,
        inner=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="exceeds the configured microbatch slots"):
        dispatcher.prepare_layer_inputs([torch.ones(2, 4), torch.ones(2, 4)])


def test_moe_list_forward_uses_incoming_width_when_ultraep_is_off():
    model = object.__new__(MoE)
    model.config = Qwen3MoE235BA22Config(intra_layer_micro_batch=1)
    captured: dict[str, object] = {}

    def _fake_micro_batch_forward(*, seq_ctx_list, loss_ctx_list, return_router_logits=False):
        captured["seq_ctx_list"] = seq_ctx_list
        captured["loss_ctx_list"] = loss_ctx_list
        return "ok"

    model._micro_batch_forward = _fake_micro_batch_forward  # type: ignore[method-assign]
    contexts = [object(), object()]
    losses = [{}, {}]

    assert model.forward(seq_ctx=contexts, loss_ctx=losses) == "ok"
    assert captured["seq_ctx_list"] is contexts
    assert captured["loss_ctx_list"] is losses


def test_ultra_ep_microbatch_pipeline_scales_beyond_two():
    """Exercise the generic MB-N forward schedule without a CUDA dispatcher.

    All preprocesses complete before any dispatch; all expert pre_combines
    complete before any combine.  This also protects MB>2 from list-length
    or accidental hard-coded-two changes.
    """

    class FakeDispatcher:
        def __init__(self, calls):
            self.calls = calls
            self.prepared = []

        def prepare_layer_inputs(self, layer_inputs):
            self.prepared.append(len(layer_inputs))
            return layer_inputs, [f"state-{i}" for i in range(len(layer_inputs))]

        def prepare_microbatch_input(self, hidden_states, layer_state):
            self.calls.append(("prepare_input", layer_state))
            return hidden_states

        def dispatch_preprocess(self, *, hidden_states, topk_ids, topk_weights, layer_state=None, async_op=False):
            self.calls.append(("preprocess", layer_state))
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

    assert layer.dispatcher.prepared == [3]
    assert [call[1] for call in calls if isinstance(call, tuple) and call[0] == "preprocess"] == [
        "state-0",
        "state-1",
        "state-2",
    ]
    assert len(output["hidden_states"]) == 3
    assert calls[:6] == [
        item
        for i in range(3)
        for item in (("prepare_input", f"state-{i}"), ("preprocess", f"state-{i}"))
    ]
    stage_names = [
        call if isinstance(call, str) else call[0]
        for call in calls
        if isinstance(call, str) or call[0] != "prepare_input"
    ]
    assert stage_names[:3] == ["preprocess", "preprocess", "preprocess"]
    assert stage_names[3:12] == [
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
    assert stage_names[12:15] == ["combine", "combine", "combine"]


@pytest.mark.parametrize("microbatches", [1, 2])
def test_ultra_ep_decoder_joins_each_reduce_before_next_start(monkeypatch, microbatches):
    """Run the real decoder/dispatcher graph with a strict staging owner.

    Hoisting all Join nodes before attention reproduces start(1), start(0)
    and fails the owner check even without CUDA or torch.compile.
    """
    runtime, manager, _, _ = make_fake_layer_runtime()
    runtime.model_runtime.group = FakeGroup()
    runtime.model_runtime.max_microbatches = microbatches
    events = []
    owner = None

    def start(vid):
        nonlocal owner
        assert owner is None, f"staging still owned by {owner}; attempted to start {vid}"
        owner = vid
        events.append(("start", vid))

    def finish(vid):
        nonlocal owner
        assert owner == vid
        owner = None
        events.append(("join", vid))

    monkeypatch.setattr(runtime, "start_grad_reduce", start)
    monkeypatch.setattr(runtime, "finish_grad_reduce", finish)

    class IdentityTransport:
        def dispatch_preprocess(self, *, hidden_states, **kwargs):
            return {"hidden_states": hidden_states}

        def dispatch(self, *, pre_dispatched, **kwargs):
            return pre_dispatched

        def dispatch_postprocess(self, *, dispatched, **kwargs):
            return {"hidden_states": dispatched["hidden_states"], "tokens_per_expert": torch.tensor([2])}

        def combine_preprocess(self, *, hidden_states, **kwargs):
            return {"hidden_states": hidden_states}

        def combine(self, *, pre_combined, **kwargs):
            return pre_combined

        def combine_postprocess(self, *, combined, **kwargs):
            return combined

    layer = object.__new__(MoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.n_shared_experts = 0
    layer.ep_mesh = None
    layer.dispatcher = UltraEPDispatcher(
        model_runtime=runtime.model_runtime,
        layer_runtime=runtime,
        inner=IdentityTransport(),
    )
    attention_weight = nn.Parameter(torch.tensor(0.5))
    expert_weight = nn.Parameter(torch.tensor(0.25))

    def pre_moe(hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = hidden_states * attention_weight
        router = {
            "logits": hidden_states[..., :1],
            "router_weights": hidden_states[..., :1],
            "topk_ids": torch.zeros((2, 1), dtype=torch.long),
            "topk_weights": torch.ones(2, 1),
        }
        return residual, hidden_states, router, None

    layer._pre_moe_forward = pre_moe
    layer.experts = lambda hidden_states, *args, **kwargs: hidden_states * expert_weight
    layer._post_moe_forward = lambda *, combined_hidden_states, residual, shared_experts_out: (
        combined_hidden_states + residual
    )
    inputs = [torch.ones(1, 2, 4, requires_grad=True) for _ in range(microbatches)]
    if microbatches == 1:
        outputs = [layer._forward(inputs[0], None, (None, None))["hidden_states"]]
    else:
        outputs = layer._micro_batch_forward(
            inputs, [None] * microbatches, [(None, None)] * microbatches
        )["hidden_states"]
    sum(x.sum() for x in outputs).backward()

    vids = [vid for _, vid in manager.allocate_calls]
    assert events == [event for vid in reversed(vids) for event in (("start", vid), ("join", vid))]
    assert owner is None
    torch.testing.assert_close(expert_weight.grad, torch.tensor(4.0 * microbatches))
    torch.testing.assert_close(attention_weight.grad, torch.tensor(2.0 * microbatches))


def test_ultra_ep_dispatcher_maps_control_plane_onto_six_stages():
    layer_runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()
    model_runtime = layer_runtime.model_runtime
    model_runtime.group = FakeGroup()

    inner_calls: list[object] = []

    class FakeInner:
        def dispatch_preprocess(self, *, hidden_states, topk_ids, topk_weights, async_op=False, layer_state=None):
            inner_calls.append(("preprocess", int(topk_ids[0, 0]), hidden_states.shape))
            return {"hidden_states": hidden_states, "topk_ids": topk_ids}

        def dispatch(self, *, pre_dispatched, topk_weights, async_op=False, decoding=False):
            inner_calls.append("dispatch")
            return {"hidden_states": pre_dispatched["hidden_states"]}

        def dispatch_postprocess(self, *, pre_dispatched, dispatched, async_op=False):
            inner_calls.append("postprocess")
            return {
                "hidden_states": dispatched["hidden_states"],
                "tokens_per_expert": torch.tensor([1]),
            }

        def combine_preprocess(self, *, hidden_states, **kwargs):
            inner_calls.append(("combine_preprocess", hidden_states.shape))
            return {"hidden_states": hidden_states}

        def combine(self, *, pre_combined, **kwargs):
            inner_calls.append("combine")
            return pre_combined

        def combine_postprocess(self, *, combined, **kwargs):
            inner_calls.append("combine_postprocess")
            return {"hidden_states": combined["hidden_states"]}

    dispatcher = UltraEPDispatcher(
        model_runtime=model_runtime,  # type: ignore[arg-type]
        layer_runtime=layer_runtime,
        inner=FakeInner(),  # type: ignore[arg-type]
    )
    hidden = torch.ones(2, 4, requires_grad=True)
    prepared, states = dispatcher.prepare_layer_inputs([hidden])
    assert prepared[0] is hidden
    prepared[0] = dispatcher.prepare_microbatch_input(prepared[0], states[0])
    assert layer_runtime.model_runtime.max_microbatches == 1
    assert manager.allocate_calls == [(2, 0)]
    assert len(fused_w1w3.configure_calls) == 1
    assert len(fused_w2.configure_calls) == 1
    assert states[0] is not None
    assert states[0].virtual_layer_id == 0

    topk_ids = torch.zeros(2, 1, dtype=torch.long)
    pre = dispatcher.dispatch_preprocess(
        hidden_states=prepared[0],
        topk_ids=topk_ids,
        topk_weights=torch.ones(2, 1),
        layer_state=states[0],
    )
    assert manager.control_calls == ["placement", "sync", "reroute"]
    assert manager.weight_sync_calls == [(0, True)]
    assert inner_calls[0][0] == "preprocess"
    assert inner_calls[0][1] == 1

    dispatched = dispatcher.dispatch(pre_dispatched=pre, topk_weights=torch.ones(2, 1))
    post = dispatcher.dispatch_postprocess(pre_dispatched=pre, dispatched=dispatched)
    assert manager.event_calls == [("wait", 0)]
    layout = post["expert_weight_layout"]
    assert isinstance(layout, ExpertWeightLayout)
    assert layout.trainable_weights is None
    assert fused_w1w3.select_calls[-1] == 0
    assert fused_w2.select_calls[-1] == 0

    experts_out = post["hidden_states"]
    pre_combined = dispatcher.combine_preprocess(
        hidden_states=experts_out,
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
    post_combined = dispatcher.combine_postprocess(
        pre_dispatched=pre,
        dispatched=dispatched,
        post_dispatched=post,
        pre_combined=pre_combined,
        combined=combined,
    )
    assert inner_calls[-3:] == [("combine_preprocess", experts_out.shape), "combine", "combine_postprocess"]
    assert post_combined["hidden_states"].shape == hidden.shape

    with pytest.raises(RuntimeError, match="requires layer_state from prepare_layer_inputs"):
        dispatcher.dispatch_preprocess(
            hidden_states=hidden,
            topk_ids=topk_ids,
            topk_weights=torch.ones(2, 1),
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


def test_grouped_linear_uses_selected_module_slot(monkeypatch):
    monkeypatch.setenv("XTUNER_GROUP_GEMM", "triton_dual")
    seen: list[object] = []

    def fake_gemm(
        x,
        weight,
        tokens,
        tokens_per_expert_cpu=None,
        replica_weight=None,
        replica_grad=None,
    ):
        seen.append(replica_weight)
        return x @ weight[0].T

    linear = GroupedLinear(4, 6, 2, group_gemm=fake_gemm)
    replica_weight = torch.stack(
        (
            torch.full((1, 6, 4), 1.0, dtype=torch.bfloat16),
            torch.full((1, 6, 4), 2.0, dtype=torch.bfloat16),
        )
    )
    replica_grad = torch.zeros_like(replica_weight, dtype=torch.float32)
    linear.configure_ultra_ep_buffers(replica_weight, replica_grad)
    linear.select_ultra_ep_slot(1)

    linear(torch.ones(3, 4), torch.tensor([2, 1]))
    assert seen[0] is linear._ultra_ep_replica_weight
    torch.testing.assert_close(seen[0], replica_weight[1])



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
    assert fused_w1w3._ultra_ep_replica_weight.data_ptr() == fused_w1w3._ultra_ep_replica_weight_slots[1].data_ptr()


def test_ultra_ep_weight_restore_launch_and_join_are_async():
    runtime, manager, fused_w1w3, fused_w2 = make_fake_layer_runtime()

    runtime.start_weight_restore(7)
    assert manager.weight_sync_calls == [(7, True)]
    assert fused_w1w3.select_calls == [0]
    assert fused_w2.select_calls == [0]

    runtime.finish_weight_restore(7)
    assert manager.event_calls == [("wait", 7)]
    assert fused_w1w3.select_calls == [0, 1]
    assert fused_w2.select_calls == [0, 1]
    assert runtime._weight_restore_events == {}


def test_ultra_ep_weight_restore_autograd_nodes_launch_before_join():
    calls = []

    class _RestoreOrderRuntime:
        def start_weight_restore(self, virtual_layer_id):
            calls.append(("start", virtual_layer_id))

        def finish_weight_restore(self, virtual_layer_id):
            calls.append(("finish", virtual_layer_id))

    runtime = _RestoreOrderRuntime()
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
    calls = []

    class _GradReduceOrderRuntime:
        def start_grad_reduce(self, virtual_layer_id):
            calls.append(("start", virtual_layer_id))

        def finish_grad_reduce(self, virtual_layer_id):
            calls.append(("finish", virtual_layer_id))

    runtime = _GradReduceOrderRuntime()
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

    if not adapter.TE_GROUPED_GEMM_INSTALLED:
        pytest.skip("te_grouped_gemm is not installed (UltraEP TE extra, not in .[all])")

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

    runtime, _, _, _ = make_fake_layer_runtime()
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

    runtime, _, _, _ = make_fake_layer_runtime()
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
