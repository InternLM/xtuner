from __future__ import annotations

import threading


_PATCHED_ATTR = "_xtuner_testing_dist_port_base_patched"
_NEXT_BASE_ATTR = "_xtuner_testing_next_dist_port_base"
_STEP_ATTR = "_xtuner_testing_dist_port_base_step"
_LOCK_ATTR = "_xtuner_testing_dist_port_base_lock"
_ORIGINAL_DEFAULT_ATTR = "_xtuner_testing_dist_port_base_original_default"


def reset_rollout_config_dist_port_base_sequence(initial_base: int | None = None) -> int:
    """Reset the next automatic RolloutConfig dist_port_base for this process."""
    from xtuner.v1.rl.rollout.worker import RolloutConfig

    if initial_base is None:
        initial_base = int(
            getattr(RolloutConfig, _ORIGINAL_DEFAULT_ATTR, RolloutConfig.model_fields["dist_port_base"].default)
        )
    setattr(RolloutConfig, _NEXT_BASE_ATTR, initial_base)
    return initial_base


def get_rollout_config_dist_port_base_default() -> int:
    """Return the dist_port_base that the next implicit RolloutConfig will use."""
    from xtuner.v1.rl.rollout.worker import RolloutConfig

    return int(getattr(RolloutConfig, _NEXT_BASE_ATTR, RolloutConfig.model_fields["dist_port_base"].default))


def patch_rollout_config_dist_port_base(step: int = 24, initial_base: int | None = None) -> None:
    """Assign a per-process dist_port_base sequence to implicit RolloutConfig defaults.

    When ``dist_port_base`` is not passed to ``RolloutConfig(...)``, each new
    object receives the next base port in a process-local sequence. Explicit
    ``dist_port_base`` values are left untouched and do not consume the sequence.
    """
    from xtuner.v1.rl.rollout.worker import RolloutConfig

    field = RolloutConfig.model_fields["dist_port_base"]
    dist_port_base_default = int(getattr(RolloutConfig, _ORIGINAL_DEFAULT_ATTR, field.default))
    if not hasattr(RolloutConfig, _ORIGINAL_DEFAULT_ATTR):
        setattr(RolloutConfig, _ORIGINAL_DEFAULT_ATTR, dist_port_base_default)

    setattr(RolloutConfig, _STEP_ATTR, step)
    if not hasattr(RolloutConfig, _LOCK_ATTR):
        setattr(RolloutConfig, _LOCK_ATTR, threading.Lock())
    if initial_base is not None or not hasattr(RolloutConfig, _NEXT_BASE_ATTR):
        reset_rollout_config_dist_port_base_sequence(initial_base)

    if getattr(RolloutConfig, _PATCHED_ATTR, False):
        return

    def dist_port_base_factory() -> int:
        with getattr(RolloutConfig, _LOCK_ATTR):
            next_base = int(getattr(RolloutConfig, _NEXT_BASE_ATTR))
            updated_base = next_base + int(getattr(RolloutConfig, _STEP_ATTR))
            updated_base = (updated_base - dist_port_base_default) % (65536 - dist_port_base_default) + dist_port_base_default
            setattr(RolloutConfig, _NEXT_BASE_ATTR, updated_base)
            return next_base

    field.default_factory = dist_port_base_factory
    RolloutConfig.model_rebuild(force=True)
    setattr(RolloutConfig, _PATCHED_ATTR, True)


__all__ = [
    "get_rollout_config_dist_port_base_default",
    "patch_rollout_config_dist_port_base",
    "reset_rollout_config_dist_port_base_sequence",
]
