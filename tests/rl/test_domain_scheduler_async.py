"""Adaptive admissions use the native async owner, replay and request barrier."""

import asyncio
import itertools
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, AgentLoopManagerConfig, TaskSpecConfig
from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.agent_loop_manager.colocate_keepalive import ColocateKeepaliveManager
from xtuner.v1.rl.agent_loop_manager.domain_scheduler import (
    AdaptiveDomainSchedulerConfig,
    DomainAdmissionWindow,
    DomainSchedulingState,
)
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks, _TaskRunner, take_train_batch
from xtuner.v1.rl.agent_loop_manager.producer import AsyncProduceStrategyConfig, ProduceProgress
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt


def config(**overrides):
    return AdaptiveDomainSchedulerConfig(
        **{
            "target_groups": {"slow": 1, "fast": 1},
            "initial_acceptance_rates": {"slow": 1.0, "fast": 1.0},
            "max_concurrent_samples": 2,
            "seed": 1,
            **overrides,
        }
    )


def group(uid, *, status=Status.COMPLETED, score=1.0, versions=None):
    return [
        RolloutState(
            rollout_id=uid,
            group_id=uid,
            message=[{"role": "user", "content": str(uid)}],
            prompt_ids=[10],
            tokens=[10],
            response="ok",
            response_ids=[],
            reward=None if score is None else {"score": score},
            response_model_steps=versions or [],
            status=status,
            extra_fields={},
        )
    ]


class Sampler:
    prompt_repeat_k = 1

    def __init__(self, start):
        self.ids = itertools.count(start)
        self.consumed = 0

    def __len__(self):
        return 100

    async def sample(self, **kwargs):
        self.consumed += 1
        return group(next(self.ids), status=Status.INIT)

    def save(self, path):
        (Path(path) / "sampler.json").write_text(json.dumps(self.consumed))

    def resume(self, path):
        self.consumed = json.loads((Path(path) / "sampler.json").read_text())


class GateController:
    def __init__(self, gate):
        self.gate = gate
        self.continue_generation = SimpleNamespace(remote=AsyncMock())
        self.pause_generation = SimpleNamespace(remote=AsyncMock())
        self.assert_request_gates_healthy = SimpleNamespace(remote=AsyncMock())
        self.prepare_pause = SimpleNamespace(remote=self.pause)
        self.resume_requests = SimpleNamespace(remote=self.resume)
        self.set_enable_partial_rollout = SimpleNamespace(remote=MagicMock())

    async def pause(self, **kwargs):
        epoch = self.gate.snapshot()["epoch"] + 1
        await self.gate.seal(epoch)
        await self.gate.drain(1)
        return epoch

    async def resume(self, epoch, model_step, **kwargs):
        await self.gate.install_version(epoch, model_step)
        await self.gate.reopen(epoch)


def manager(loops, *, cfg=None, gate=None, filters=None, strategy_kwargs=None):
    gate = gate or RequestGate(wait_timeout=5)
    runners = []
    for i, (name, loop) in enumerate(loops.items()):
        strategy = AsyncProduceStrategyConfig(
            **{
                "over_sample_threshold": 1.5,
                "max_staleness": 2,
                **(strategy_kwargs or {}),
            }
        ).build()
        strategy.retain_live_tasks = True
        runners.append(
            _TaskRunner(
                task_name=name,
                agent_loop=loop,
                produce_strategy=strategy,
                sampler=Sampler(i * 1000),
                order=i,
            )
        )
    result = AgentLoopManager(
        runners,
        AsyncReplayBufferConfig().build(),
        GateController(gate),
        logger=MagicMock(),
        domain_scheduler=cfg or config(),
        filter_enabled=filters,
    )
    result._keepalive = AgentTurnPausePolicy(freeze_timeout=1, drain_timeout=1)
    return result


async def close(manager):
    await asyncio.gather(*(runner.produce_strategy._pending_tasks.cancel_all() for runner in manager.task_runners))
    await asyncio.sleep(0)  # completion callbacks release admission reservations


class AsyncSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_build_retains_the_preview_keepalive_wrapper_and_strategy(self):
        tasks = [
            TaskSpecConfig.model_construct(
                task_name=name,
                agent_loop_config=SimpleNamespace(require_turn_provenance=True, build=lambda **kw: object()),
                produce_strategy_config=AsyncProduceStrategyConfig(),
                sampler_config=SimpleNamespace(build=lambda **kw: Sampler(0)),
            )
            for name in ("slow", "fast")
        ]
        cfg = AgentLoopManagerConfig(tasks=tasks, domain_scheduler=config().model_dump(), turn_keepalive=True)
        with patch("ray.get", return_value=None):
            built = cfg.build(
                GateController(RequestGate(wait_timeout=5)), MagicMock(), AsyncReplayBufferConfig().build()
            )
        try:
            self.assertIsInstance(built, ColocateKeepaliveManager)
            self.assertEqual(built._domain_scheduler.config.target_groups, {"slow": 1, "fast": 1})
            self.assertTrue(all(task.produce_strategy.retain_live_tasks for task in built.task_runners))
            self.assertEqual(cfg.domain_scheduler.kind, "adaptive")
        finally:
            await built.close()

    async def test_slow_session_survives_update_and_keeps_native_turn_versions(self):
        gate = RequestGate(wait_timeout=5)
        release_tool = asyncio.Event()
        slow_started = asyncio.Event()
        sessions = []
        request_ids = []
        cancelled = []

        class Loop:
            def __init__(self, slow):
                self.slow = slow

            async def generate_group(self, states, **kwargs):
                session = f"session-{states[0].group_id}"
                versions = []
                try:
                    async with gate.request(session) as lease:
                        versions.append(lease.model_step)
                        request_ids.append(lease.request_id)
                        lease.settlement = SettlementReceipt(True, True, "completed")
                    if self.slow:
                        sessions.append(session)
                        slow_started.set()
                        await release_tool.wait()
                        async with gate.request(session) as lease:
                            versions.append(lease.model_step)
                            request_ids.append(lease.request_id)
                            sessions.append(lease.session_id)
                            lease.settlement = SettlementReceipt(True, True, "completed")
                    else:
                        await slow_started.wait()
                        await asyncio.sleep(0.005)
                    state = states[0]
                    state.status = Status.COMPLETED
                    state.response_ids = [20] * len(versions)
                    state.tokens = [10] + state.response_ids
                    state.response_model_steps = versions
                    return states
                except asyncio.CancelledError:
                    cancelled.append(session)
                    raise

        m = manager({"slow": Loop(True), "fast": Loop(False)}, gate=gate)
        slow_owner = m.task_runners[0].produce_strategy._pending_tasks
        try:
            first = await asyncio.wait_for(m.produce_batch(2, 0, model_step=0), 5)
            self.assertEqual([g[0].task_name for g in first.rollout_states], ["fast", "fast"])
            self.assertEqual(slow_owner.count(), 1)
            self.assertEqual(cancelled, [])
            self.assertFalse(gate.snapshot()["open"])
            release_tool.set()
            await asyncio.sleep(0.02)
            # The next model call is blocked at the original closed request gate.
            self.assertEqual(len(sessions), 1)
            second = await asyncio.wait_for(m.produce_batch(2, 1, model_step=1, quiesce=True), 5)
            slow = [g[0] for g in second.rollout_states if g[0].group_id == 0]
            self.assertEqual(len(slow), 1)
            self.assertEqual(slow[0].response_model_steps, [0, 1])
            self.assertEqual(sessions[:2], [sessions[0], sessions[0]])
            self.assertTrue(all(uid.startswith("chatcmpl-") for uid in request_ids))
            self.assertIs(m.task_runners[0].produce_strategy._pending_tasks, slow_owner)
            self.assertEqual(cancelled, [])
            self.assertEqual(m._domain_scheduler.in_flight, {"slow": 0, "fast": 0})
        finally:
            await close(m)

    async def test_native_filter_rejections_are_replaced_without_changing_loss_masks(self):
        class Loop:
            def __init__(self, status):
                self.status = status

            async def generate_group(self, states, **kwargs):
                await asyncio.sleep(0.001)
                states[0].status = self.status
                return states

        m = manager(
            {"slow": Loop(Status.FILTERED), "fast": Loop(Status.COMPLETED)},
            filters={"slow": True, "fast": False},
        )
        try:
            result = await asyncio.wait_for(m.produce_batch(2, 0, model_step=0, quiesce=True), 5)
            self.assertEqual([g[0].task_name for g in result.rollout_states], ["fast", "fast"])
            state = m._domain_scheduler
            self.assertGreater(state.acceptance_observations["slow"], 0)
            self.assertLess(state.acceptance["slow"], 1)
            self.assertEqual(state.acceptance["fast"], 1)
            self.assertEqual(state.acceptance_observations["fast"], 0)
        finally:
            await close(m)

    async def test_cancelled_and_failed_coroutines_release_sample_slots(self):
        state = DomainSchedulingState(config())
        window = DomainAdmissionWindow(
            state,
            AsyncReplayBufferConfig().build(),
            siblings={"slow": 1, "fast": 1},
            filter_enabled={"slow": True, "fast": True},
        )
        owners = {name: _PendingTasks() for name in ("slow", "fast")}
        release = asyncio.Event()

        async def fail():
            await release.wait()
            raise RuntimeError("native task failed")

        async def spawn():
            return asyncio.create_task(fail())

        await asyncio.gather(
            *(
                window.register(name, pending=owners[name], spawn=spawn, max_pending=1, oversample_budget=2)
                for name in owners
            )
        )
        self.assertTrue(await window.schedule_one())
        self.assertTrue(await window.schedule_one())
        self.assertFalse(await window.schedule_one())
        self.assertEqual(sum(state.in_flight.values()), 2)
        await owners["slow"].cancel_all()
        release.set()
        await asyncio.sleep(0.01)
        for owner in owners.values():
            for task in await owner.claim_ready():
                self.assertIsInstance(task.exception(), RuntimeError)
        self.assertEqual(sum(state.in_flight.values()), 0)
        self.assertEqual(sum(state.duration_observations.values()), 0)
        self.assertEqual(sum(state.acceptance_observations.values()), 0)

    async def test_fifo_retains_surplus_and_uses_original_token_masking(self):
        m = manager({"slow": object(), "fast": object()}, strategy_kwargs={"max_token_staleness": 2})
        oldest = group(1)[0]
        oldest.response_ids = [30, -1, 40]
        oldest.response_model_steps = [0, 4, 4]
        oldest.set_training_fields(
            input_ids=[10, 30, -1, 40],
            labels=[-100, 30, -100, 40],
            logprobs=None,
        )
        oldest.extra_fields["native_tool_error"] = "recoverable argument error"
        await m.replay_buffer.put([oldest], "fast")
        await m.replay_buffer.put(group(2), "slow")
        await m.replay_buffer.put(group(3), "fast")
        await m.replay_buffer.put(group(4, status=Status.ABORTED), "slow")
        counts = await m.replay_buffer.fifo_counts(["slow", "fast"], limit=2)
        self.assertEqual(counts, {"slow": 1, "fast": 1})
        result = await take_train_batch(
            task_runners=m.task_runners,
            replay_buffer=m.replay_buffer,
            logger=MagicMock(),
            manager_name="multi_task",
            task_batch_sizes={"slow": 1, "fast": 1},
            progress=ProduceProgress.build(task_names=m.task_names, target_samples=counts),
            current_train_step=5,
            shared_fifo=True,
        )
        self.assertEqual([g[0].group_id for g in result.rollout_states], [1, 2])
        self.assertEqual(oldest.labels, [-100, -100, -100, 40])
        self.assertEqual(oldest.extra_fields["native_tool_error"], "recoverable argument error")
        self.assertEqual(await m.replay_buffer.count("fast", Status.COMPLETED), 1)
        self.assertEqual(await m.replay_buffer.count("slow", Status.ABORTED), 1)
        self.assertEqual(result.task_batch_sizes, {"fast": 1, "slow": 1})

    async def test_shrinking_domain_cap_does_not_cancel_existing_groups(self):
        state = DomainSchedulingState(
            config(
                target_groups={"slow": 2, "fast": 2},
                max_concurrent_samples=12,
                concurrency_pbar=0.5,
                concurrency_pmax=2,
                initial_duration_seconds={"slow": 100, "fast": 1},
                duration_ewma_alpha=1,
            )
        )
        window = DomainAdmissionWindow(
            state,
            AsyncReplayBufferConfig().build(),
            siblings={"slow": 2, "fast": 1},
            filter_enabled={"slow": False, "fast": False},
        )
        owners = {name: _PendingTasks() for name in ("slow", "fast")}
        cancelled = []

        async def wait():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def spawn():
            return asyncio.create_task(wait())

        await asyncio.gather(
            *(
                window.register(name, pending=owners[name], spawn=spawn, max_pending=4, oversample_budget=4)
                for name in owners
            )
        )
        try:
            # The sample ceiling remains independent of per-domain group caps.
            for _ in range(10):
                await window.schedule_one()
                self.assertLessEqual(state.in_flight["slow"] * 2 + state.in_flight["fast"], 12)
            await asyncio.sleep(0)
            pending = sum(owner.count() for owner in owners.values())
            self.assertEqual(state.in_flight["slow"], 4)
            state.observe("slow", duration_seconds=0.1, filter_verdict=None)
            self.assertEqual(state.solution.group_caps["slow"], 2)
            self.assertEqual(sum(owner.count() for owner in owners.values()), pending)
            self.assertEqual(cancelled, [])
            while await window.schedule_one():
                self.assertEqual(state.in_flight["slow"], 4)
                self.assertLessEqual(state.in_flight["slow"] * 2 + state.in_flight["fast"], 12)
        finally:
            await asyncio.gather(*(owner.cancel_all() for owner in owners.values()))


def test_checkpoint_restores_fifo_and_draw_cursor_but_resets_online_estimates(tmp_path):
    async def run():
        m = manager({"slow": object(), "fast": object()})
        await m.replay_buffer.put(group(1), "fast")
        await m.replay_buffer.put(group(2), "slow")
        m._domain_scheduler.draw({}, m.task_names)
        m._domain_scheduler.observe("slow", duration_seconds=2, filter_verdict=False)
        await m.save(tmp_path, model_step=4)
        expected = [m._domain_scheduler.draw({}, m.task_names) for _ in range(20)]
        restored = manager({"slow": object(), "fast": object()})
        assert await restored.resume(tmp_path) == 4
        assert [restored._domain_scheduler.draw({}, m.task_names) for _ in range(20)] == expected
        assert restored._domain_scheduler.acceptance == {"slow": 1, "fast": 1}
        _, _, groups = await restored.replay_buffer.take_fifo_batch(2, m.task_names)
        assert [g[0].group_id for g in groups] == [1, 2]

    asyncio.run(run())


def test_global_pacing_has_no_catchup_credit(monkeypatch):
    async def run():
        now = [100.0]
        monkeypatch.setattr("xtuner.v1.rl.agent_loop_manager.domain_scheduler.time.perf_counter", lambda: now[0])
        state = DomainSchedulingState(config(max_concurrent_samples=10, group_start_interval=1))
        window = DomainAdmissionWindow(
            state,
            AsyncReplayBufferConfig().build(),
            siblings={"slow": 1, "fast": 1},
            filter_enabled={"slow": False, "fast": False},
        )
        owners = {name: _PendingTasks() for name in ("slow", "fast")}

        async def spawn():
            return asyncio.create_task(asyncio.Event().wait())

        await asyncio.gather(
            *(
                window.register(name, pending=owners[name], spawn=spawn, max_pending=10, oversample_budget=10)
                for name in owners
            )
        )
        try:
            assert await window.schedule_one()
            assert not await window.schedule_one()
            now[0] = 120
            assert await window.schedule_one()
            assert not await window.schedule_one()
            assert window.next_start == 121
        finally:
            await asyncio.gather(*(owner.cancel_all() for owner in owners.values()))

    asyncio.run(run())


@pytest.mark.parametrize(
    "status,score,enabled,expected",
    [
        (Status.COMPLETED, 0.0, True, 1),
        (Status.FILTERED, 0.0, True, 1),
        (Status.FAILED, 0.0, True, 0),
        (Status.ABORTED, 0.0, True, 0),
        (Status.COMPLETED, None, True, 0),
        (Status.COMPLETED, 1.0, False, 0),
    ],
)
def test_acceptance_excludes_infrastructure_unscored_and_disabled_filter(status, score, enabled, expected):
    async def run():
        state = DomainSchedulingState(config())
        window = DomainAdmissionWindow(
            state,
            AsyncReplayBufferConfig().build(),
            siblings={"slow": 1, "fast": 1},
            filter_enabled={"slow": enabled, "fast": False},
        )
        state.in_flight["slow"] = 1
        result = asyncio.create_task(asyncio.sleep(0, result=group(1, status=status, score=score)))
        await result
        window._finished("slow", time.perf_counter() - 1, result)
        assert state.in_flight["slow"] == 0
        assert state.acceptance_observations["slow"] == expected
        assert state.duration_observations["slow"] == 1

    asyncio.run(run())
