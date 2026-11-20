from xtuner._testing.patch_rollout_config import get_rollout_config_dist_port_base_default
from xtuner.v1.rl.rollout.cross_weight_update_rollout import CrossWeightUpdateRolloutConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig


def _build_rollout_config(**kwargs) -> RolloutConfig:
    return RolloutConfig(
        model_path="/tmp/nonexistent-model",
        context_length=1,
        **kwargs,
    )


class TestCrossWeightUpdateRolloutConfig:
    def test_rollout_config_json_round_trip_preserves_cross_update_settings(self):
        config = _build_rollout_config(
            cross_weight_update_rollout={"wait_timeout": 7, "freeze_timeout": 2, "drain_timeout": 3}
        )
        restored = RolloutConfig.model_validate_json(config.model_dump_json())
        assert restored.cross_weight_update_rollout == CrossWeightUpdateRolloutConfig(
            wait_timeout=7, freeze_timeout=2, drain_timeout=3
        )


def test_rollout_config_default_dist_port_base_is_staggered(monkeypatch):
    monkeypatch.delenv("XTUNER_USE_SGLANG", raising=False)
    monkeypatch.delenv("XTUNER_USE_VLLM", raising=False)
    monkeypatch.setenv("XTUNER_USE_LMDEPLOY", "1")

    base = get_rollout_config_dist_port_base_default()
    first = _build_rollout_config()
    second = _build_rollout_config()
    explicit = _build_rollout_config(dist_port_base=45678)
    third = _build_rollout_config()

    assert first.dist_port_base == base
    assert second.dist_port_base == base + 24
    assert explicit.dist_port_base == 45678
    assert third.dist_port_base == base + 48


def test_rollout_config_warns_for_deprecated_allow_over_concurrency_ratio(monkeypatch):
    monkeypatch.delenv("XTUNER_USE_SGLANG", raising=False)
    monkeypatch.delenv("XTUNER_USE_VLLM", raising=False)
    monkeypatch.setenv("XTUNER_USE_LMDEPLOY", "1")

    warnings = []

    class _FakeLogger:
        def warning(self, message):
            warnings.append(message)

    monkeypatch.setattr("xtuner.v1.rl.rollout.worker.get_logger", lambda *args, **kwargs: _FakeLogger())

    _build_rollout_config(allow_over_concurrency_ratio=2.0)

    assert len(warnings) == 1
    assert "allow_over_concurrency_ratio is deprecated" in warnings[0]
    assert "will be ignored" in warnings[0]
