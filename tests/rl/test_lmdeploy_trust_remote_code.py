# Copyright (c) OpenMMLab. All rights reserved.
"""The LMDeploy server must trust custom code only when the model dir declares
it."""

import json
from argparse import Namespace
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


def _write_tokenizer_config(directory: Path, with_auto_map: bool) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    config = {"tokenizer_class": "Glm5Tokenizer"} if with_auto_map else {"tokenizer_class": "PreTrainedTokenizerFast"}
    if with_auto_map:
        config["auto_map"] = {"AutoTokenizer": ["tokenization_glm5.Glm5Tokenizer"]}
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    return str(directory)


def _server_config(tmp_path: Path, with_auto_map: bool, extra: dict | None = None) -> Namespace:
    model_path = _write_tokenizer_config(tmp_path / ("custom" if with_auto_map else "stock"), with_auto_map)
    worker = LMDeployWorker.__new__(LMDeployWorker)
    worker.config = RolloutConfig(
        model_path=model_path,
        context_length=128,
        extra_rollout_config={"lmdeploy_backend": "turbomind", **(extra or {})},
    )
    worker.model_name = "test-model"
    worker.host = "127.0.0.1"
    worker.server_port = 12345
    worker.api_keys = None
    worker.accelerator = "GPU"
    worker.server_launch_spec = _StubLaunchSpec()
    # turbomind backend avoids the ray runtime-context branch of the transform.
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
