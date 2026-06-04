from xtuner._testing.patch_rollout_config import get_rollout_config_dist_port_base_default
from xtuner.v1.rl.rollout.worker import RolloutConfig


def _build_rollout_config(**kwargs) -> RolloutConfig:
    return RolloutConfig(
        model_path="/tmp/nonexistent-model",
        context_length=1,
        **kwargs,
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
