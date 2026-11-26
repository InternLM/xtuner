from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from lmdeploy.messages import KVTransferConfig

from xtuner.v1.rl.rollout.lmdeploy import LMDeployWorker
from xtuner.v1.rl.rollout.worker import RolloutConfig


def resolve(extra: dict[str, Any], r3: bool = False, backend: str = "pytorch") -> Namespace:
    cfg = RolloutConfig(
        env="store-test",
        model_path="/model",
        expert_parallel_size=4,
        tensor_parallel_size=1,
        rollout_max_batch_size_per_instance=64,
        enable_return_routed_experts=r3,
        extra_rollout_config={"lmdeploy_backend": backend, **extra},
    )
    topology = LMDeployWorker.build_rollout_topology(
        cfg, [(r, r) for r in range(8)], {r: f"127.0.0.1:{19000 + r}" for r in range(8)}
    )
    shell = SimpleNamespace(
        config=cfg,
        accelerator="GPU",
        server_launch_spec=topology.server_launch_specs()[0],
        model_name=cfg.model_name,
        host="127.0.0.1",
        server_port=19000,
        api_keys=None,
    )
    with (
        patch("xtuner.v1.rl.rollout.lmdeploy.ray.get_runtime_context", return_value=SimpleNamespace(namespace="test")),
        patch("ray.util.get_current_placement_group", return_value=object()),
        patch("xtuner.v1.rl.rollout.lmdeploy.placement_group_table", return_value={"name": "test"}),
    ):
        return LMDeployWorker._transform_rollout_config_to_server_configs(shell)


def transfer() -> dict[str, Any]:
    return {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"mooncake_config_path": "/private/store.json", "cache_prefix": "test"},
    }


class TestLMDeployStoreAdapter:
    def test_disabled_unchanged_and_allows_r3(self) -> None:
        assert resolve({}, r3=True).backend_config.kv_transfer_config is None

    def test_store_passes_to_real_engine_without_mutating_user_config(self) -> None:
        data = transfer()
        before = deepcopy(data)
        a = resolve({"lmdeploy_kv_transfer_config": data}).backend_config.kv_transfer_config
        b = resolve({"lmdeploy_kv_transfer_config": data}).backend_config.kv_transfer_config
        assert data == before and a.kv_connector == "MooncakeStoreConnector"
        assert a.kv_connector_extra_config["rl_weight_versioning"] is True
        assert a.kv_connector_extra_config["weight_version"] != b.kv_connector_extra_config["weight_version"]

    def test_dataclass_input(self) -> None:
        value = KVTransferConfig(**transfer())
        got = resolve({"lmdeploy_kv_transfer_config": value}).backend_config.kv_transfer_config
        assert got is not value and "weight_version" not in value.kv_connector_extra_config

    def test_r3_requires_real_store_sidecars(self) -> None:
        value = resolve({"lmdeploy_kv_transfer_config": transfer()}, r3=True).backend_config.kv_transfer_config
        assert value.kv_connector_extra_config["store_routed_experts"] is True
        assert value.kv_connector_extra_config["rl_weight_versioning"] is True

    def test_rejects_unknown_connector(self) -> None:
        bad = transfer()
        bad["kv_connector"] = "Unknown"
        with pytest.raises(ValueError, match="MooncakeStoreConnector only"):
            resolve({"lmdeploy_kv_transfer_config": bad})

    def test_small_cache_and_decode_save_options_reach_engine_without_store(self) -> None:
        got = resolve(
            {
                "lmdeploy_num_gpu_blocks": 4100,
                "lmdeploy_prefix_cache_state_budget": 2,
                "lmdeploy_prefix_cache_decode_state_interval": 8192,
                "lmdeploy_prefix_cache_decode_save_first": True,
                "lmdeploy_prefill_save_last_only": True,
            }
        ).backend_config
        assert got.kv_transfer_config is None
        assert got.num_gpu_blocks == 4100 and got.prefix_cache_state_budget == 2
        assert got.prefix_cache_decode_state_interval == 8192 and got.prefix_cache_decode_save_first

    def test_last_prefill_checkpoint_option_is_not_a_generate_api_kwarg(self) -> None:
        cfg = resolve({"lmdeploy_prefill_save_last_only": True})
        assert cfg.backend_config.prefill_save_last_only is True
