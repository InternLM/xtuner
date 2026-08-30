"""SingleTurnAgentLoop 的 batch judge / pause 控制流测试。

本文件不加载 tokenizer、processor、真实 rollout controller 或 judger 服务。
当前测试点：
- batch judge 只在整组样本全部 COMPLETED 时触发。
- batch judge 返回结果必须保持输入顺序。
- rollout group validity check 在 batch judge 之后执行。
- 组内存在 ABORTED / FAILED 等非 COMPLETED 样本时跳过 judge。
- pause 发生在 slow judger 期间时，整组样本被标记为 ABORTED。
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import AgentLoopActor
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop, SingleTurnAgentLoopConfig


class _RemoteGenerate:
    def __init__(self, statuses_by_uid: dict[int, Status]):
        self.statuses_by_uid = statuses_by_uid
        self.calls: list[RolloutState] = []

    async def remote(self, rollout_state: RolloutState):
        self.calls.append(rollout_state)
        rollout_state.status = self.statuses_by_uid[rollout_state.rollout_id]
        if rollout_state.status == Status.COMPLETED:
            rollout_state.response = f"response {rollout_state.rollout_id}"
            rollout_state.response_ids = [rollout_state.rollout_id or 0]
            rollout_state.finish_reason = "stop"
        elif rollout_state.status == Status.ABORTED:
            rollout_state.finish_reason = "abort"
        else:
            rollout_state.finish_reason = "error"
        return rollout_state


class _BatchJudger:
    def __init__(self):
        self.calls: list[list[RolloutState]] = []

    async def batch_judge(self, rollout_states):
        self.calls.append(rollout_states)
        for state in rollout_states:
            state.reward = {"score": float(state.rollout_id)}
        return rollout_states


class _SlowJudger:
    async def batch_judge(self, rollout_states):
        await asyncio.sleep(60)
        return rollout_states


class TestSingleTurnAgentLoop(unittest.IsolatedAsyncioTestCase):
    def _state(self, uid: int) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=uid,
            message=[{"role": "user", "content": f"prompt {uid}"}],
            prompt_ids=[uid],
            tokens=None,
            response=None,
            response_ids=None,
            status=Status.INIT,
            extra_fields={},
        )

    def _build_loop(self, statuses_by_uid: dict[int, Status], judger=None, is_valid_sample_fn=None):
        loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
        rollout_ctl = MagicMock()
        rollout_ctl.generate = _RemoteGenerate(statuses_by_uid)
        rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        loop.rollout_ctl = rollout_ctl
        loop.sample_params = SampleParams(max_tokens=8, temperature=0.7)
        loop.judger = judger
        loop.enable_batch_judge = True
        loop.is_valid_sample_fn = is_valid_sample_fn
        loop._judger_pause_event = asyncio.Event()
        loop.logger = MagicMock()
        return loop

    def test_config_binds_validity_check_to_local_agent_loop_once(self):
        is_valid_sample_fn = MagicMock()
        local_loop = MagicMock()
        config = SingleTurnAgentLoopConfig.model_construct(hf_checkpoint="unused", cpu_resources=None)

        with patch.object(SingleTurnAgentLoopConfig, "build_local", return_value=local_loop) as build_local:
            result = config.build(MagicMock(), is_valid_sample_fn=is_valid_sample_fn)

        self.assertIs(result, local_loop)
        self.assertIs(local_loop.is_valid_sample_fn, is_valid_sample_fn)
        build_local.assert_called_once()

    def test_config_forwards_validity_check_when_building_ray_actor(self):
        is_valid_sample_fn = MagicMock()
        ray_agent_loop = MagicMock()
        cpu_resources = MagicMock()
        cpu_resources.num_workers = 1
        config = SingleTurnAgentLoopConfig.model_construct(
            hf_checkpoint="unused",
            cpu_resources=cpu_resources,
        )

        with (
            patch("xtuner.v1.rl.agent_loop.agent_loop.register_cpu_resources"),
            patch.object(SingleTurnAgentLoopConfig, "_build_ray_actor", return_value=ray_agent_loop) as build_actor,
        ):
            result = config.build(MagicMock(), is_valid_sample_fn=is_valid_sample_fn)

        self.assertIs(result, ray_agent_loop)
        self.assertIs(build_actor.call_args.kwargs["is_valid_sample_fn"], is_valid_sample_fn)

    def test_actor_binds_validity_check_during_construction(self):
        is_valid_sample_fn = MagicMock()
        local_loop = MagicMock()
        config = MagicMock()
        config.build_local.return_value = local_loop

        actor = AgentLoopActor(config, MagicMock(), is_valid_sample_fn=is_valid_sample_fn)

        self.assertIs(actor.agent_loop.is_valid_sample_fn, is_valid_sample_fn)

    async def test_generate_filters_completed_group_after_batch_judge_and_logs(self):
        filtered_rewards = []

        def is_valid_sample_fn(samples):
            filtered_rewards.extend(state.reward for state in samples)
            return False

        loop = self._build_loop(
            {1: Status.COMPLETED, 2: Status.COMPLETED},
            judger=_BatchJudger(),
            is_valid_sample_fn=is_valid_sample_fn,
        )

        result = await loop.generate_group([self._state(1), self._state(2)])

        self.assertEqual(filtered_rewards, [{"score": 1.0}, {"score": 2.0}])
        self.assertTrue(all(state.status == Status.FILTERED for state in result))
        loop.logger.info.assert_called_once()

    async def test_batch_judge_runs_once_when_all_samples_completed_and_preserves_order(self):
        # 整组样本全部 COMPLETED 时才触发 batch judger；返回顺序必须和输入顺序一致。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.COMPLETED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertEqual(len(judger.calls), 1)
        self.assertEqual([state.rollout_id for state in judger.calls[0]], [1, 2])
        self.assertEqual([state.reward for state in result], [{"score": 1.0}, {"score": 2.0}])
        self.assertTrue(all(state.sample_params == loop.sample_params for state in result))

    async def test_batch_judge_is_skipped_when_any_sample_is_aborted(self):
        # 组内只要出现 ABORTED，batch judger 就不应被调用，避免给不可训练样本写 reward。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.ABORTED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertEqual([state.status for state in result], [Status.COMPLETED, Status.ABORTED])
        self.assertEqual(judger.calls, [])
        self.assertTrue(all(state.reward is None for state in result))

    async def test_batch_judge_is_skipped_when_any_sample_is_not_completed(self):
        # batch judge 的 contract 是全组 COMPLETED 才打分；FAILED/FILTERED 也必须跳过。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.FAILED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertEqual([state.status for state in result], [Status.COMPLETED, Status.FAILED])
        self.assertEqual(judger.calls, [])
        self.assertTrue(all(state.reward is None for state in result))

    async def test_pause_during_batch_judge_marks_group_aborted(self):
        # pause 到来后 slow judger 超过取消等待时间，run_judger 应取消任务并把整组样本标记为 ABORTED。
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.COMPLETED}, judger=_SlowJudger())
        samples = [self._state(1), self._state(2)]

        with patch("xtuner.v1.rl.agent_loop.agent_loop.JUDGER_PAUSE_JUDGE_TASK_TIMEOUT_S", 0.01):
            task = asyncio.create_task(loop.run_judger(samples))
            await asyncio.sleep(0)
            loop._judger_pause_event.set()
            result = await task

        self.assertEqual([state.rollout_id for state in result], [1, 2])
        self.assertTrue(all(state.status == Status.ABORTED for state in result))
        self.assertTrue(all(state.finish_reason == "abort" for state in result))
        self.assertTrue(all(state.reward is None for state in result))


if __name__ == "__main__":
    unittest.main()
