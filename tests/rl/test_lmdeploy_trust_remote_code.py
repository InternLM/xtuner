# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for LMDeploy rollout server configuration."""

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest

from xtuner.v1.rl.rollout.lmdeploy import LMDeployWorker, _model_dir_declares_custom_code
from xtuner.v1.rl.rollout.worker import RolloutConfig


pytest.importorskip("lmdeploy")


@pytest.fixture(autouse=True)
def _select_lmdeploy_backend(monkeypatch):
    monkeypatch.delenv("XTUNER_USE_SGLANG", raising=False)
    monkeypatch.delenv("XTUNER_USE_VLLM", raising=False)
    monkeypatch.setenv("XTUNER_USE_LMDEPLOY", "1")


class _StubLaunchSpec:
    placement_group_bundle_idxs = [0, 1]


@dataclass(kw_only=True)
class _LegacyPytorchEngineConfig:
    tp: int
    ep: int
    dp: int
    dp_rank: int
    max_batch_size: int
    empty_init: bool
    distributed_executor_backend: str
    mp_engine_backend: str
    device_type: str
    logprobs_mode: str
    session_len: int | None
    model_format: str | None
    cache_max_entry_count: float
    enable_prefix_caching: bool = False


@dataclass(kw_only=True)
class _PartialPytorchEngineConfig(_LegacyPytorchEngineConfig):
    prefix_cache_state_budget: int = 0
    prefill_save_alignment: int = 256


@dataclass(kw_only=True)
class _SupportedPytorchEngineConfig(_PartialPytorchEngineConfig):
    prefill_save_alignment: int = 1024
    prefix_cache_decode_save_first: bool = False
    prefix_cache_decode_state_interval: int = 0
    prefill_save_last_only: bool = False


def _write_tokenizer_config(directory: Path, with_auto_map: bool) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    config = {"tokenizer_class": "Glm5Tokenizer"} if with_auto_map else {"tokenizer_class": "PreTrainedTokenizerFast"}
    if with_auto_map:
        config["auto_map"] = {"AutoTokenizer": ["tokenization_glm5.Glm5Tokenizer"]}
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    return str(directory)


def _server_config(tmp_path: Path, with_auto_map: bool, extra: dict | None = None, **rollout_options) -> Namespace:
    model_path = _write_tokenizer_config(tmp_path / ("custom" if with_auto_map else "stock"), with_auto_map)
    worker = LMDeployWorker.__new__(LMDeployWorker)
    worker.config = RolloutConfig(
        model_path=model_path,
        context_length=128,
        rollout_max_batch_size_per_instance=1,
        extra_rollout_config={"lmdeploy_backend": "turbomind", **(extra or {})},
        **rollout_options,
    )
    worker.model_name = "test-model"
    worker.host = "127.0.0.1"
    worker.server_port = 12345
    worker.api_keys = None
    worker.accelerator = "GPU"
    worker.server_launch_spec = _StubLaunchSpec()
    return worker._transform_rollout_config_to_server_configs()


def test_detection_reads_auto_map(tmp_path):
    custom = _write_tokenizer_config(tmp_path / "custom", with_auto_map=True)
    stock = _write_tokenizer_config(tmp_path / "stock", with_auto_map=False)
    assert _model_dir_declares_custom_code(custom) is True
    assert _model_dir_declares_custom_code(stock) is False
    assert _model_dir_declares_custom_code(tmp_path / "missing") is False


def test_trust_remote_code_follows_auto_map_declaration(tmp_path):
    assert _server_config(tmp_path, with_auto_map=True).trust_remote_code is True
    assert _server_config(tmp_path, with_auto_map=False).trust_remote_code is False


def test_trust_remote_code_override_wins_both_ways(tmp_path):
    opt_out = _server_config(tmp_path, with_auto_map=True, extra={"lmdeploy_trust_remote_code": False})
    opt_in = _server_config(tmp_path, with_auto_map=False, extra={"lmdeploy_trust_remote_code": True})
    assert opt_out.trust_remote_code is False
    assert opt_in.trust_remote_code is True


class TestLMDeployPrefixCaching:
    @pytest.fixture(autouse=True)
    def _stub_ray_placement(self, monkeypatch):
        monkeypatch.setattr("ray.get_runtime_context", lambda: Namespace(namespace="test"))
        monkeypatch.setattr("ray.util.get_current_placement_group", lambda: None)
        monkeypatch.setattr("xtuner.v1.rl.rollout.lmdeploy.placement_group_table", lambda _: {"name": "test"})

    @pytest.mark.parametrize("enabled", [False, True])
    def test_legacy_engine_accepts_default_ssm_settings(self, tmp_path, monkeypatch, enabled):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _LegacyPytorchEngineConfig)
        server = _server_config(
            tmp_path, with_auto_map=False, extra={"lmdeploy_backend": "pytorch"}, enable_prefix_caching=enabled
        )
        assert server.backend_config.enable_prefix_caching is enabled

    def test_legacy_engine_accepts_explicit_previous_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _LegacyPytorchEngineConfig)
        server = _server_config(
            tmp_path,
            with_auto_map=False,
            extra={"lmdeploy_backend": "pytorch"},
            enable_prefix_caching=True,
            prefill_save_alignment=1024,
            prefill_save_last_only=False,
            prefix_cache_decode_save_first=False,
            prefix_cache_decode_state_interval=0,
        )
        assert server.backend_config.enable_prefix_caching is True

    def test_partial_engine_accepts_supported_ssm_setting(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _PartialPytorchEngineConfig)
        server = _server_config(
            tmp_path, with_auto_map=False, extra={"lmdeploy_backend": "pytorch"}, prefix_cache_state_budget=3
        )
        assert server.backend_config.prefix_cache_state_budget == 3
        assert server.backend_config.prefill_save_alignment == 64

    def test_partial_engine_rejects_requested_unsupported_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _PartialPytorchEngineConfig)
        with pytest.raises(
            RuntimeError, match="requested SSM prefix-checkpoint controls prefix_cache_decode_save_first"
        ):
            _server_config(
                tmp_path,
                with_auto_map=False,
                extra={"lmdeploy_backend": "pytorch"},
                prefix_cache_decode_save_first=True,
            )

    @pytest.mark.parametrize("enabled", [False, True])
    def test_supported_engine_receives_rollout_defaults(self, tmp_path, monkeypatch, enabled):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _SupportedPytorchEngineConfig)
        server = _server_config(
            tmp_path, with_auto_map=False, extra={"lmdeploy_backend": "pytorch"}, enable_prefix_caching=enabled
        )
        cfg = server.backend_config
        assert cfg.prefill_save_alignment == 64
        assert cfg.prefill_save_last_only is True
        assert cfg.prefix_cache_decode_save_first is True
        assert cfg.prefix_cache_decode_state_interval == 1024
        assert cfg.prefix_cache_state_budget == 0
        assert cfg.enable_prefix_caching is enabled

    def test_supported_engine_receives_all_five_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lmdeploy.messages.PytorchEngineConfig", _SupportedPytorchEngineConfig)
        settings = {
            "prefill_save_alignment": 512,
            "prefix_cache_state_budget": 3,
            "prefix_cache_decode_save_first": False,
            "prefix_cache_decode_state_interval": 0,
            "prefill_save_last_only": False,
        }
        server = _server_config(
            tmp_path,
            with_auto_map=False,
            extra={"lmdeploy_backend": "pytorch"},
            enable_prefix_caching=True,
            **settings,
        )
        assert {name: getattr(server.backend_config, name) for name in settings} == settings
