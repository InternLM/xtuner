from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import ray

from xtuner.v1.rl.rollout.worker_registry import WorkerLifecycleState
from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.controller import RolloutControllerProxy
    from xtuner.v1.rl.rollout.worker import RolloutConfig
    from xtuner.v1.rl.trainer.controller import TrainingController


RL_HEALTH_MANAGER_RAY_GET_TIMEOUT = 3600
RL_HEALTH_MANAGER_STOP_JOIN_TIMEOUT = 30.0
PENDING_ROLLOUT_WORKER_CHECK_INTERVAL = 1.0
ROLLOUT_WEIGHT_UPDATE_DRAIN_TIMEOUT = 600.0


class RLHealthManager:
    """Coordinate driver-side recovery of colocated rollout workers.

    RolloutHealthManager owns worker restart and moves successfully restarted workers to PENDING_WEIGHTS. This manager
    polls pending workers on the driver and updates them from the latest registered checkpoint before promoting them to
    ACTIVE.
    """

    def __init__(
        self,
        *,
        train_controller: TrainingController,
        rollout_controller: RolloutControllerProxy,
        rollout_config: RolloutConfig,
    ) -> None:
        self.enable_pending_weight_recovery = (
            rollout_config is not None and rollout_config.weight_transport_type == "checkpoint_engine"
        )
        self.train_controller = train_controller
        self.rollout_controller = rollout_controller
        self._rollout_config = rollout_config
        self._rollout_resources_available = threading.Event()
        self._rollout_weight_update_lock = threading.Lock()
        self._pending_rollout_weight_update_stop_event = threading.Event()
        self._pending_rollout_weight_update_thread: threading.Thread | None = None
        self.logger = get_logger(tag="RLHealthManager")

    def _check_enabled_dependencies(self) -> None:
        if not self.enable_pending_weight_recovery:
            return
        if self.train_controller is None or self.rollout_controller is None or self._rollout_config is None:
            raise RuntimeError("RLHealthManager is missing Checkpoint Engine recovery dependencies.")

    def start(self) -> None:
        if not self.enable_pending_weight_recovery:
            return
        self._check_enabled_dependencies()
        if self._pending_rollout_weight_update_thread is not None:
            return

        self._pending_rollout_weight_update_stop_event.clear()
        self._pending_rollout_weight_update_thread = threading.Thread(
            target=self._pending_rollout_worker_weight_update_loop,
            name="pending-rollout-weight-update",
            daemon=True,
        )
        self._pending_rollout_weight_update_thread.start()
        self.logger.info("Started pending rollout checkpoint-engine update thread.")

    def stop(self) -> None:
        if not self.enable_pending_weight_recovery:
            return
        self._check_enabled_dependencies()
        self._pending_rollout_weight_update_stop_event.set()

        thread = self._pending_rollout_weight_update_thread
        if thread is not None:
            thread.join(timeout=RL_HEALTH_MANAGER_STOP_JOIN_TIMEOUT)
            if thread.is_alive():
                self.logger.warning(
                    "Pending rollout weight update thread did not stop before "
                    f"timeout={RL_HEALTH_MANAGER_STOP_JOIN_TIMEOUT}s."
                )
                return

        self._pending_rollout_weight_update_thread = None
        self.logger.info("Stopped pending rollout checkpoint-engine update thread.")

    def set_rollout_resources_available(self, available: bool) -> None:
        # If rollout resources are unavailable, shutdown any inactive rollout workers to free up resources for training.
        if not available:
            ray.get(
                self.rollout_controller.shutdown_inactive_workers.remote(),
                timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
            )

        if not self.enable_pending_weight_recovery:
            return
        if available:
            self._rollout_resources_available.set()
            return

        # Stop admitting new pending-worker updates, then wait for an update
        # that already owns the lock to finish before training reuses the
        # colocated rollout resources.
        self._rollout_resources_available.clear()
        acquired = self._rollout_weight_update_lock.acquire(timeout=ROLLOUT_WEIGHT_UPDATE_DRAIN_TIMEOUT)
        if not acquired:
            raise TimeoutError(
                "Timed out waiting for pending rollout weight update before switching to training: "
                f"timeout={ROLLOUT_WEIGHT_UPDATE_DRAIN_TIMEOUT}s."
            )
        self._rollout_weight_update_lock.release()

    @contextmanager
    def weight_update_guard(self):
        """Serialize normal and recovery Checkpoint Engine weight updates."""
        if not self.enable_pending_weight_recovery:
            yield
            return
        self._check_enabled_dependencies()
        with self._rollout_weight_update_lock:
            yield

    def _update_pending_rollout_weights_from_checkpoint_engine(self) -> tuple[tuple[int, ...], ...]:
        """Update every currently pending rollout group from Checkpoint
        Engine."""
        self._check_enabled_dependencies()
        assert self.rollout_controller is not None
        assert self.train_controller is not None
        assert self._rollout_config is not None
        if not self.train_controller.has_registered_weight_checkpoint():
            self.logger.info(
                "Defer pending rollout checkpoint-engine update because no train checkpoint has been registered yet."
            )
            return ()

        pending_targets, pending_group_ranks = ray.get(
            self.rollout_controller.get_weight_update_targets.remote(
                target_state=WorkerLifecycleState.PENDING_WEIGHTS,
                return_group_ranks=True,
            ),
            timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
        )
        if not pending_targets:
            return ()

        try:
            self.logger.info(
                "Updating pending rollout workers from Checkpoint Engine: "
                f"group_ranks={pending_group_ranks}, targets={pending_targets}."
            )
            self.train_controller.bind_rollout_weight_update(
                targets=pending_targets,
                rollout_config=self._rollout_config,
            )
            ray.get(
                self.rollout_controller.onload_weights.remote(target_state=WorkerLifecycleState.PENDING_WEIGHTS),
                timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
            )
            self.train_controller.weight_update(need_register=False, need_update=True)
            ray.get(
                self.rollout_controller.onload_kvcache.remote(target_state=WorkerLifecycleState.PENDING_WEIGHTS),
                timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
            )
            ray.get(
                self.rollout_controller.mark_worker_groups_lifecycle_state.remote(
                    group_ranks=list(pending_group_ranks),
                    source_state=WorkerLifecycleState.PENDING_WEIGHTS,
                    target_state=WorkerLifecycleState.ACTIVE,
                ),
                timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
            )
            self.logger.info(
                f"Recovered rollout workers weight updated from Checkpoint Engine: {pending_group_ranks}."
            )
            return tuple(pending_group_ranks)
        except Exception:
            self.logger.exception(
                f"Failed to update recovered rollout workers weight from Checkpoint Engine: {pending_group_ranks}."
            )
            ray.get(
                self.rollout_controller.mark_worker_groups_lifecycle_state.remote(
                    group_ranks=list(pending_group_ranks),
                    source_state=WorkerLifecycleState.PENDING_WEIGHTS,
                    target_state=WorkerLifecycleState.INACTIVE,
                ),
                timeout=RL_HEALTH_MANAGER_RAY_GET_TIMEOUT,
            )
            return ()

    def _pending_rollout_worker_weight_update_loop(self) -> None:
        while not self._pending_rollout_weight_update_stop_event.wait(PENDING_ROLLOUT_WORKER_CHECK_INTERVAL):
            if not self._rollout_resources_available.is_set():
                continue
            if not self._rollout_weight_update_lock.acquire(blocking=False):
                continue
            try:
                # The rollout phase may have ended after the check above but
                # before this thread acquired the lock.
                if not self._rollout_resources_available.is_set():
                    continue
                updated_groups = self._update_pending_rollout_weights_from_checkpoint_engine()
                if updated_groups:
                    self.logger.info(
                        f"Background pending rollout checkpoint-engine update completed: {updated_groups}."
                    )
            except Exception:
                self.logger.exception("Background pending rollout weight update failed.")
            finally:
                self._rollout_weight_update_lock.release()


__all__ = ["RLHealthManager"]
