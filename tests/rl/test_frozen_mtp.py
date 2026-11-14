"""External-source identity, actual native iteration, export and resume gates."""

import hashlib
import json
import shutil
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import save_file

from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.role_worker import RoleWorker
from xtuner.v1.rl.weight_update import frozen_mtp
from xtuner.v1.rl.weight_update.frozen_mtp import FrozenMTPConfig, FrozenMTPSource
from xtuner.v1.rl.weight_update.weight_iterator import WeightIterator


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 4,
            "vocab_size": 8,
            "num_experts": 2,
            "moe_intermediate_size": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    tensors = {
        "mtp.fc.weight": torch.arange(32, dtype=torch.float16).reshape(4, 8),
        "mtp.norm.weight": torch.arange(4, dtype=torch.float32),
    }
    save_file(tensors, directory / "model.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in tensors}})
    )
    specs = {
        key: dict(
            shape=list(t.shape),
            dtype=str(t.dtype),
            nbytes=t.numel() * t.element_size(),
            sha256=hashlib.sha256(memoryview(t.view(torch.uint8).numpy()).cast("B")).hexdigest(),
        )
        for key, t in tensors.items()
    }
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"tensors": specs, "config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()}
        )
    )
    return FrozenMTPConfig(source_dir=directory, manifest_path=manifest, content_sha256=digest(specs))


def rollout(**changes):
    result = dict(
        rollout_backend="lmdeploy",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        weight_transport_type="ipc",
        extra_rollout_config={
            "lmdeploy_speculative_algorithm": "qwen3_5_mtp",
            "lmdeploy_speculative_num_draft_tokens": 3,
        },
    )
    result.update(changes)
    return SimpleNamespace(**result)


def actor_config(**changes):
    cls = type("Qwen3_5_TestConfig", (SimpleNamespace,), {})
    config = dict(
        hidden_size=4,
        vocab_size=8,
        n_routed_experts=2,
        moe_intermediate_size=2,
        mtp_config=None,
        attention=SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, head_dim=2),
    )
    config.update(changes)
    return cls(**config)


def actor_export(directory):
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.language_model.embed_tokens.weight": torch.full((8, 4), 7.0),
        "lm_head.weight": torch.full((8, 4), 9.0),
    }
    save_file(tensors, directory / "model.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 256}, "weight_map": {key: "model.safetensors" for key in tensors}})
    )
    return tensors


def test_explicit_source_identity_model_and_backend_contract(source, monkeypatch):
    monkeypatch.setenv("XT_FROZEN_MTP_SOURCE", "/ignored/obsolete/env")
    monkeypatch.delenv("XTUNER_USE_SGLANG", raising=False)
    monkeypatch.delenv("XTUNER_USE_VLLM", raising=False)
    source.validate_model(actor_config())
    source.validate_model(SimpleNamespace(text_config=actor_config()))
    source.validate_rollout(rollout())
    with pytest.raises(ValueError, match="MTP training is unsupported"):
        source.validate_model(actor_config(mtp_config=object()))
    with pytest.raises(ValueError, match="hidden_size"):
        source.validate_model(actor_config(hidden_size=16))
    monkeypatch.setenv("XTUNER_USE_VLLM", "1")
    with pytest.raises(ValueError, match="LMDeploy pytorch"):
        source.validate_rollout(rollout())


@pytest.mark.parametrize(
    "changes",
    [
        {"tensor_parallel_size": 2},
        {"expert_parallel_size": 2},
        {"weight_transport_type": "checkpoint_engine"},
        {"extra_rollout_config": {"lmdeploy_backend": "turbomind"}},
        {"extra_rollout_config": {}},
        *[
            {
                "extra_rollout_config": {
                    "lmdeploy_speculative_algorithm": "qwen3_5_mtp",
                    "lmdeploy_speculative_num_draft_tokens": n,
                }
            }
            for n in [0, -1, True, 1.5]
        ],
    ],
)
def test_unsupported_rollout_rejected_without_launching_engine(source, changes):
    with pytest.raises(ValueError):
        source.validate_rollout(rollout(**changes))


@pytest.mark.parametrize("corruption", ["tensor", "manifest", "missing_key", "shape", "config"])
def test_source_corruption_is_rejected_before_transfer(source, corruption):
    directory = source.source_dir
    if corruption in {"tensor", "shape"}:
        tensors = {
            "mtp.fc.weight": torch.zeros((4, 8) if corruption == "tensor" else (4, 9), dtype=torch.float16),
            "mtp.norm.weight": torch.arange(4, dtype=torch.float32),
        }
        save_file(tensors, directory / "model.safetensors")
    elif corruption == "manifest":
        data = json.loads(source.manifest_path.read_text())
        data["tensors"]["mtp.norm.weight"]["sha256"] = "0" * 64
        source.manifest_path.write_text(json.dumps(data))
    elif corruption == "missing_key":
        path = directory / "model.safetensors.index.json"
        data = json.loads(path.read_text())
        del data["weight_map"]["mtp.norm.weight"]
        path.write_text(json.dumps(data))
    else:
        (directory / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        FrozenMTPSource(source)


def test_native_iterator_sends_target_then_frozen_before_final_on_every_update(source, monkeypatch):
    from xtuner.v1.rl.weight_update import weight_iterator

    monkeypatch.setattr(weight_iterator, "DEVICE", "cpu")
    parameter = torch.full((2,), 11.0)

    class Model:
        fsdp_config = None

        def _load_spec_params(self):
            return [(parameter, SimpleNamespace(is_fused=False))]

        def _get_hf_param(self, params, **kwargs):
            if params:
                yield ["lm_head.weight"], [parameter.clone()]

    frozen = FrozenMTPSource(source)
    iterator = WeightIterator(
        config=SimpleNamespace(model_cfg=None, update_weight_bucket_size_in_gb=0.01),
        engine=SimpleNamespace(model=Model()),
        global_hf_keys_mapping_cache={},
        rollout_info=SimpleNamespace(backend="pytorch", tp=1, ep=1, transport_type="ipc"),
        frozen_mtp_source=frozen,
    )
    expected = {key: value.clone() for batch in frozen.iter_batches("cpu") for key, value in batch.state_dict.items()}
    for step in range(2):
        parameter.add_(1)
        received = {}  # Model a sleep2 receiver which discarded ALL its weights.
        batches = list(iterator.iter_hf_batches(final_update=True))
        assert [batch.finished for batch in batches] == [False, False, False, True]
        for batch in batches[:-1]:
            received.update(batch.state_dict)
        assert torch.equal(received["lm_head.weight"], parameter)
        for key, value in expected.items():
            assert torch.equal(received[key], value)
        # A receiver mutating its payload cannot alter the pinned source snapshot.
        received["mtp.fc.weight"].zero_()
    native = WeightIterator(
        config=iterator.config,
        engine=iterator._engine,
        global_hf_keys_mapping_cache={},
        rollout_info=iterator.rollout_info,
    )
    assert len(list(native.iter_hf_batches(final_update=True))) == 2


def test_actor_hf_export_completes_correct_role_then_marks_ready(source, tmp_path):
    actor = ActorWorker.__new__(ActorWorker)
    actor.rank = 0
    actor.frozen_mtp_source = FrozenMTPSource(source)
    actor.engine = SimpleNamespace(save_hf=lambda hf_dir, **kw: actor_export(hf_dir))
    directory = tmp_path / "hf" / "actor"
    actor.save_hf(directory, torch.bfloat16)
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    assert (directory / "READY").is_file()
    assert not (directory.parent / "READY").exists()
    with safe_open(directory / mapping["mtp.fc.weight"], framework="pt") as stream:
        assert torch.equal(stream.get_tensor("mtp.fc.weight"), torch.arange(32, dtype=torch.float16).reshape(4, 8))
    with safe_open(directory / mapping["lm_head.weight"], framework="pt") as stream:
        assert torch.equal(stream.get_tensor("lm_head.weight"), torch.full((8, 4), 9.0))
    old_index = (directory / "model.safetensors.index.json").read_bytes()
    with pytest.raises(RuntimeError, match="empty actor directory"):
        actor.save_hf(directory, torch.bfloat16)
    assert (directory / "model.safetensors.index.json").read_bytes() == old_index


@pytest.mark.parametrize("failure", [None, "native", "completion"])
def test_hf_warm_start_does_not_republish_copied_old_ready(source, tmp_path, monkeypatch, failure):
    frozen = FrozenMTPSource(source)
    old = tmp_path / "old" / "actor"
    actor_export(old)
    frozen.complete_hf(old)
    old_ready = (old / "READY").read_bytes()
    new = tmp_path / "new" / "actor"

    def native(hf_dir, **kwargs):
        actor_export(hf_dir)
        # The real BaseModel HF saver copies non-safetensors files from
        # _hf_path, including the two adapter records on an HF warm-start.
        for name in ["READY", frozen_mtp.SOURCE_RECORD]:
            shutil.copyfile(old / name, hf_dir / name)
        if failure == "native":
            raise OSError("native export failed after copying metadata")

    if failure == "completion":
        monkeypatch.setattr(frozen_mtp, "save_file", lambda *a, **kw: (_ for _ in ()).throw(OSError("shard failure")))
    actor = ActorWorker.__new__(ActorWorker)
    actor.rank, actor.frozen_mtp_source = 0, frozen
    actor.engine = SimpleNamespace(save_hf=native)
    if failure:
        with pytest.raises((OSError, RuntimeError)):
            actor.save_hf(new, torch.bfloat16)
        assert not (new / "READY").exists()
    else:
        actor.save_hf(new, torch.bfloat16)
        assert json.loads((new / "READY").read_text()) == frozen.identity
    assert (old / "READY").read_bytes() == old_ready


def _rank_zero_failure_worker(rank, init_file, result_dir, source_config=None):
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:

        def fail():
            raise OSError("injected rank-zero filesystem failure")

        try:
            if source_config is None:
                FrozenMTPSource._rank_zero(rank, fail)
            else:
                from pathlib import Path
                from unittest.mock import patch

                frozen = FrozenMTPSource(source_config)
                directory = Path(result_dir) / "export"

                def native():
                    if rank == 0:
                        directory.mkdir()
                    dist.barrier()

                with patch.object(Path, "unlink", side_effect=PermissionError("injected cleanup failure")):
                    frozen.export_hf(directory, native, rank=rank)
            result = "unexpected success"
        except RuntimeError as exc:
            result = str(exc)
        from pathlib import Path

        (Path(result_dir) / str(rank)).write_text(result)
    finally:
        dist.destroy_process_group()


def test_rank_zero_filesystem_failure_reaches_both_real_cpu_ranks(tmp_path):
    mp.spawn(_rank_zero_failure_worker, args=(str(tmp_path / "gloo-init"), str(tmp_path)), nprocs=2, join=True)
    errors = [(tmp_path / str(rank)).read_text() for rank in range(2)]
    assert errors[0] == errors[1]
    assert "injected rank-zero filesystem failure" in errors[0]


def test_export_cleanup_failure_reaches_both_real_cpu_ranks(source, tmp_path):
    mp.spawn(_rank_zero_failure_worker, args=(str(tmp_path / "gloo-init"), str(tmp_path), source), nprocs=2, join=True)
    errors = [(tmp_path / str(rank)).read_text() for rank in range(2)]
    assert errors[0] == errors[1]
    assert "injected cleanup failure" in errors[0]


@pytest.mark.parametrize("failure", ["shard", "index", "binding"])
def test_partial_hf_export_never_publishes_ready(source, tmp_path, monkeypatch, failure):
    directory = tmp_path / "hf"
    actor_export(directory)
    frozen = FrozenMTPSource(source)
    if failure == "shard":

        def fail(*args, **kwargs):
            raise OSError("injected shard failure")

        monkeypatch.setattr(frozen_mtp, "save_file", fail)
    else:
        original = frozen_mtp._atomic_json

        def write(path, value):
            if path.name == ("model.safetensors.index.json" if failure == "index" else frozen_mtp.SOURCE_RECORD):
                raise OSError("injected metadata failure")
            original(path, value)

        monkeypatch.setattr(frozen_mtp, "_atomic_json", write)
    with pytest.raises(OSError, match="injected"):
        frozen.complete_hf(directory)
    assert not (directory / "READY").exists()
    if failure != "shard":
        with pytest.raises(FileExistsError):
            frozen.complete_hf(directory)


def test_dcp_binding_rejects_missing_changed_or_disabled_source_before_role_load(source, tmp_path, monkeypatch):
    frozen = FrozenMTPSource(source)
    directory = tmp_path / "dcp" / "actor"
    directory.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(RoleWorker, "resume", lambda *args: calls.append("load"))
    actor = ActorWorker.__new__(ActorWorker)
    actor.frozen_mtp_source = frozen
    with pytest.raises(ValueError, match="binding is missing"):
        actor.resume(directory, object())
    assert calls == []
    frozen.save_binding(directory)
    actor.resume(directory, object())
    assert calls == ["load"]

    actor.frozen_mtp_source = None
    with pytest.raises(ValueError, match="requires its external"):
        actor.resume(directory, object())
    actor.frozen_mtp_source = frozen
    record = json.loads((directory / frozen_mtp.SOURCE_RECORD).read_text())
    record["content_sha256"] = "0" * 64
    (directory / frozen_mtp.SOURCE_RECORD).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs"):
        actor.resume(directory, object())
    assert calls == ["load"]


def test_missing_actor_shard_never_produces_ready(source, tmp_path):
    directory = tmp_path / "hf"
    actor_export(directory)
    (directory / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Actor HF shard is missing"):
        FrozenMTPSource(source).complete_hf(directory)
    assert not (directory / "READY").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_lmdeploy_config_consumes_prefixed_speculation_fields(monkeypatch, enabled):
    from xtuner.v1.rl.rollout import lmdeploy
    from xtuner.v1.rl.rollout.worker import RolloutConfig

    monkeypatch.setenv("XTUNER_USE_LMDEPLOY", "1")
    monkeypatch.delenv("XTUNER_USE_SGLANG", raising=False)
    monkeypatch.delenv("XTUNER_USE_VLLM", raising=False)

    extra = {"lmdeploy_language_model_only": True}
    if enabled:
        extra.update(lmdeploy_speculative_algorithm="qwen3_5_mtp", lmdeploy_speculative_num_draft_tokens=3)
    worker = lmdeploy.LMDeployWorker.__new__(lmdeploy.LMDeployWorker)
    worker.config = RolloutConfig(
        model_path="/tmp/unused-model", model_name="fixture", context_length=16384, extra_rollout_config=extra
    )
    worker.accelerator = "GPU"
    worker.model_name = "fixture"
    worker.host = "127.0.0.1"
    worker.server_port = 12345
    worker.api_keys = []
    worker.server_launch_spec = SimpleNamespace(engine_rank=0, placement_group_bundle_idxs=[0])
    monkeypatch.setattr(lmdeploy.ray, "get_runtime_context", lambda: SimpleNamespace(namespace="fixture"))
    monkeypatch.setattr(lmdeploy.ray.util, "get_current_placement_group", lambda: object())
    monkeypatch.setattr(lmdeploy, "placement_group_table", lambda pg: {"name": "fixture"})
    resolved = worker._transform_rollout_config_to_server_configs()
    if enabled:
        assert resolved.speculative_config.method == "qwen3_5_mtp"
        assert resolved.speculative_config.num_speculative_tokens == 3
    else:
        assert resolved.speculative_config is None


@pytest.mark.parametrize("metadata", [None, {}, {"total_size": 256}])
def test_hf_export_does_not_invent_partial_total_size(source, tmp_path, metadata):
    directory = tmp_path / "target"
    actor_export(directory)
    index = directory / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    if metadata is None:
        data.pop("metadata")
    else:
        data["metadata"] = metadata
    index.write_text(json.dumps(data))
    frozen = FrozenMTPSource(source)
    frozen.complete_hf(directory)
    exported = json.loads(index.read_text())
    if metadata is not None and "total_size" in metadata:
        assert exported["metadata"]["total_size"] == 256 + sum(spec["nbytes"] for spec in frozen.specs.values())
    else:
        assert "total_size" not in exported["metadata"]
    assert (directory / frozen_mtp.READY).is_file()
    assert set(exported["weight_map"]) == {
        "model.language_model.embed_tokens.weight", "lm_head.weight", *frozen.specs
    }
