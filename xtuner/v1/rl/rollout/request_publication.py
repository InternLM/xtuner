"""Model-version publication owned by the RolloutController.

One turn-resume window is: freeze the fixed topology, seal and drain every SessionServer, pause every backend,
run training/weight transfer, install the new version, resume the backends, then reopen admission. Those steps
must happen in that order under one owner, so they live here instead of being spread over the controller and
the agent-loop managers.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

import ray

from .health_manager import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthManager
from .worker_registry import RolloutWorkerRegistry, WorkerLifecycleState


@dataclass(frozen=True)
class CrossWeightUpdateRolloutConfig:
    """Keep live Agent rollouts running across model weight updates.

    Admitted LLM requests settle before switching weights; later requests may use
    the new version. Live Agent/tool state is not serialized in checkpoints.

    Args:
        wait_timeout (float): Maximum time an LLM request may wait for admission to reopen.
        freeze_timeout (float): Budget for freezing topology and for the install/resume/reopen control batch.
        drain_timeout (float): Maximum time to wait for already admitted requests to settle.
        quiesce_timeout (float): Maximum time to finish live Agent tasks at a checkpoint or the final step.
    """

    wait_timeout: float = 7200
    freeze_timeout: float = 30
    drain_timeout: float = 3900
    quiesce_timeout: float = 28800

    def __post_init__(self) -> None:
        for name in ("wait_timeout", "freeze_timeout", "drain_timeout", "quiesce_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"CrossWeightUpdateRolloutConfig.{name} must be positive")


@dataclass(frozen=True)
class RequestBarrier:
    """Fixed entrypoint snapshot that a pause sealed.

    Args:
        epoch (int): Monotonic admission epoch.
        endpoints (tuple[str, ...]): SessionServer URLs captured under the freeze.
        servers (tuple[Any, ...]): SessionServer actor handles for the same endpoints.
    """

    epoch: int
    endpoints: tuple[str, ...]
    servers: tuple[Any, ...]


class RequestPublication:
    """Publish rollout model versions behind a complete-turn barrier.

    Args:
        registry (RolloutWorkerRegistry): Worker registry used to snapshot fixed entrypoints.
        health_manager (RolloutHealthManager): Health lifecycle owner paused while the barrier is held.
        settings (TurnResumeConfig): Timeouts for freeze, drain, publication and reopen.
    """

    def __init__(
        self,
        registry: RolloutWorkerRegistry,
        health_manager: RolloutHealthManager,
        settings: TurnResumeConfig,
    ):
        self._registry = registry
        self._health_manager = health_manager
        self._settings = settings
        self._epoch = 0
        self._barrier: RequestBarrier | None = None
        self._published_model_step: int | None = None

    def pause(self) -> None:
        """Freeze the topology, seal and drain every entrypoint, then pause
        every backend."""
        if self._barrier is not None:
            raise RuntimeError("A request barrier already exists")
        settings = self._settings
        self._health_manager.pause()
        if not self._health_manager.wait_recovery_done(timeout=settings.freeze_timeout):
            raise TimeoutError("Could not freeze rollout lifecycle")
        workers = self._registry.active_entrypoints()
        servers = ray.get(
            [worker.actor.get_session_server.remote() for worker in workers],  # type: ignore[attr-defined]
            timeout=settings.freeze_timeout,
        )
        if not servers or any(server is None for server in servers):
            raise RuntimeError("Every fixed entrypoint must own a SessionServer")
        self._epoch += 1
        barrier = RequestBarrier(self._epoch, tuple(str(worker.session_url) for worker in workers), tuple(servers))
        self._barrier = barrier
        # Send every seal before waiting for any drain; only after all seals may backends be paused.
        ray.get(
            [server.seal_requests.remote(barrier.epoch) for server in barrier.servers],
            timeout=settings.freeze_timeout,
        )
        ray.get(
            [server.drain_requests.remote(settings.drain_timeout) for server in barrier.servers],
            timeout=settings.drain_timeout + settings.freeze_timeout,
        )
        results = ray.get(
            [
                worker.actor.pause_generation.remote()  # type: ignore[attr-defined]
                for worker in self._registry.active_workers()
            ],
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )
        if any(result is False for result in results):
            raise RuntimeError("Backend pause failed under request barrier")

    def publish(self, model_step: int) -> None:
        """Install ``model_step`` on every sealed entrypoint, resume backends,
        then reopen admission."""
        barrier = self._barrier
        if barrier is None:
            # The first produce of an already open window re-enters here; re-publishing the same version is a no-op.
            if model_step != self._published_model_step:
                raise RuntimeError("A different model version requires a pause barrier")
            return
        endpoints = tuple(worker.session_url for worker in self._registry.active_entrypoints())
        if endpoints != barrier.endpoints:
            raise RuntimeError("Fixed rollout topology changed under barrier")
        deadline = monotonic() + self._settings.freeze_timeout

        def wait_all(submit: Callable[[], list[Any]]) -> list[Any]:
            # The control batch shares one budget; do not dispatch the next phase after the budget is gone.
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Request publication deadline exceeded")
            return ray.get(submit(), timeout=remaining)

        wait_all(
            lambda: [server.install_request_version.remote(barrier.epoch, model_step) for server in barrier.servers]
        )
        results = wait_all(
            lambda: [
                worker.actor.continue_generation.remote()  # type: ignore[attr-defined]
                for worker in self._registry.get_target_state_workers(WorkerLifecycleState.ACTIVE)
            ]
        )
        if any(result is False for result in results):
            raise RuntimeError("Backend resume failed under request barrier")
        # Reopen only after every version install and backend resume succeeded. Reopening multiple entrypoints
        # is not atomic; a partial RPC failure must stop the run rather than claim a global rollback.
        wait_all(lambda: [server.reopen_requests.remote(barrier.epoch) for server in barrier.servers])
        self._published_model_step = model_step
        self._barrier = None
        self._health_manager.resume()

    def assert_healthy(self) -> None:
        """Fail if a fixed entrypoint is missing or carries an unsettled
        request."""
        workers = self._registry.active_entrypoints()
        servers = ray.get(
            [worker.actor.get_session_server.remote() for worker in workers],  # type: ignore[attr-defined]
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )
        if not servers or any(server is None for server in servers):
            raise RuntimeError("A fixed request entrypoint is missing")
        states = ray.get([server.request_status.remote() for server in servers], timeout=ROLLOUT_RAY_GET_TIMEOUT)
        failures = [state for state in states if state["failures"]]
        if failures:
            raise RuntimeError(f"Unsettled requests stop this run: {failures}")
