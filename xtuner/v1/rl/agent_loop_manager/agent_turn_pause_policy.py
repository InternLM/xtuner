"""Disaggregated pause policy; PendingTasks remains the sole live-task owner."""

from .produce_utils import pause_pending_tasks, retain_pending_tasks


class DrainPendingPausePolicy:
    pending_cleanup = staticmethod(pause_pending_tasks)

    async def check_requests(self, controller):
        return

    async def before_sync(self, manager, checkpoint_due):
        return

    async def pause(self, controller):
        await controller.pause_generation.remote()

    async def resume(self, controller, model_step):
        await controller.continue_generation.remote()


class AgentTurnPausePolicy:
    pending_cleanup = staticmethod(retain_pending_tasks)

    def __init__(self, *, freeze_timeout: float, drain_timeout: float, quiesce_timeout: float = 1800):
        self.quiesce_timeout = quiesce_timeout
        self.freeze_timeout = freeze_timeout
        self.drain_timeout = drain_timeout
        self.barrier = None

    async def check_requests(self, controller):
        await controller.assert_request_gates_healthy.remote()

    async def before_sync(self, manager, checkpoint_due):
        if checkpoint_due:
            await manager.quiesce(self.quiesce_timeout)

    async def pause(self, controller):
        self.barrier = await controller.prepare_pause.remote(
            freeze_timeout=self.freeze_timeout, drain_timeout=self.drain_timeout
        )
        await controller.pause_generation.remote()

    async def resume(self, controller, model_step):
        # Initial construction has no admitted requests and no barrier yet.
        if self.barrier is None:
            await controller.continue_generation.remote()
        else:
            await controller.resume_requests.remote(self.barrier, model_step, timeout=self.freeze_timeout)
            self.barrier = None
