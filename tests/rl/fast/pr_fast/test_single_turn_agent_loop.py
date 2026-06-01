"""SingleTurnAgentLoop 的 batch judge / pause 控制流测试。

本文件不加载 tokenizer、processor、真实 rollout controller 或 judger 服务。
当前测试点：
- batch judge 只在整组样本全部 COMPLETED 时触发。
- batch judge 返回结果必须保持输入顺序。
- 组内存在 ABORTED / FAILED 等非 COMPLETED 样本时跳过 judge。
- pause 发生在 slow judger 期间时，整组样本被标记为 ABORTED。
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class _RemoteGenerate:
    def __init__(self, statuses_by_uid: dict[int, Status]):
        self.statuses_by_uid = statuses_by_uid
        self.calls: list[RolloutState] = []

    async def remote(self, rollout_state: RolloutState):
        self.calls.append(rollout_state)
        rollout_state.status = self.statuses_by_uid[rollout_state.uid]
        if rollout_state.status == Status.COMPLETED:
            rollout_state.response = f"response {rollout_state.uid}"
            rollout_state.response_ids = [rollout_state.uid or 0]
            rollout_state.finish_reason = "stop"
        elif rollout_state.status == Status.ABORTED:
            rollout_state.finish_reason = "abort"
        else:
            rollout_state.finish_reason = "error"
        return rollout_state


class _BatchJudger:
    def __init__(self):
        self.calls: list[list[RolloutState]] = []

    async def judge(self, rollout_states):
        self.calls.append(rollout_states)
        for state in rollout_states:
            state.reward = {"score": float(state.uid)}
        return rollout_states


class _SlowJudger:
    async def judge(self, rollout_states):
        await asyncio.sleep(60)
        return rollout_states


class TestSingleTurnAgentLoop(unittest.IsolatedAsyncioTestCase):
    def _state(self, uid: int) -> RolloutState:
        return RolloutState(
            uid=uid,
            message_uid=uid,
            message=[{"role": "user", "content": f"prompt {uid}"}],
            prompt_ids=[uid],
            tokens=None,
            response=None,
            response_ids=None,
            status=Status.INIT,
            extra_fields={},
        )

    def _build_loop(self, statuses_by_uid: dict[int, Status], judger=None):
        loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
        rollout_ctl = MagicMock()
        rollout_ctl.generate = _RemoteGenerate(statuses_by_uid)
        rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        loop.rollout_ctl = rollout_ctl
        loop.sample_params = SampleParams(max_tokens=8, temperature=0.7)
        loop.judger = judger
        loop.enable_batch_judge = True
        loop._pause_event = asyncio.Event()
        loop.logger = MagicMock()
        return loop

    async def test_batch_judge_runs_once_when_all_samples_completed_and_preserves_order(self):
        # 整组样本全部 COMPLETED 时才触发 batch judger；返回顺序必须和输入顺序一致。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.COMPLETED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.uid for state in result], [1, 2])
        self.assertEqual(len(judger.calls), 1)
        self.assertEqual([state.uid for state in judger.calls[0]], [1, 2])
        self.assertEqual([state.reward for state in result], [{"score": 1.0}, {"score": 2.0}])
        self.assertTrue(all(state.sample_params == loop.sample_params for state in result))

    async def test_batch_judge_is_skipped_when_any_sample_is_aborted(self):
        # 组内只要出现 ABORTED，batch judger 就不应被调用，避免给不可训练样本写 reward。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.ABORTED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.uid for state in result], [1, 2])
        self.assertEqual([state.status for state in result], [Status.COMPLETED, Status.ABORTED])
        self.assertEqual(judger.calls, [])
        self.assertTrue(all(state.reward is None for state in result))

    async def test_batch_judge_is_skipped_when_any_sample_is_not_completed(self):
        # batch judge 的 contract 是全组 COMPLETED 才打分；FAILED/FILTERED 也必须跳过。
        judger = _BatchJudger()
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.FAILED}, judger=judger)
        samples = [self._state(1), self._state(2)]

        result = await loop.generate_group(samples)

        self.assertEqual([state.uid for state in result], [1, 2])
        self.assertEqual([state.status for state in result], [Status.COMPLETED, Status.FAILED])
        self.assertEqual(judger.calls, [])
        self.assertTrue(all(state.reward is None for state in result))

    async def test_pause_during_batch_judge_marks_group_aborted(self):
        # pause 到来后 slow judger 超过取消等待时间，run_judger 应取消任务并把整组样本标记为 ABORTED。
        loop = self._build_loop({1: Status.COMPLETED, 2: Status.COMPLETED}, judger=_SlowJudger())
        samples = [self._state(1), self._state(2)]

        with patch("xtuner.v1.rl.agent_loop.single_turn_agent_loop.DEFAULT_JUDGER_CANCEL_TIMEOUT_S", 0.01):
            task = asyncio.create_task(loop.run_judger(samples))
            await asyncio.sleep(0)
            loop._pause_event.set()
            result = await task

        self.assertEqual([state.uid for state in result], [1, 2])
        self.assertTrue(all(state.status == Status.ABORTED for state in result))
        self.assertTrue(all(state.finish_reason == "abort" for state in result))
        self.assertTrue(all(state.reward is None for state in result))


if __name__ == "__main__":
    unittest.main()
