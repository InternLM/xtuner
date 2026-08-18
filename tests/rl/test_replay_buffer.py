# 当前 ReplayBuffer PR-fast 单测覆盖点：
#
# 1. 公共读写/消费契约：SyncReplayBufferConfig 和 AsyncReplayBufferConfig 都要保证
#    task 隔离、status 隔离、count/count_statuses、is_ready、take_batch 消费语义正确。
# 2. 空输入和零 size 操作是 no-op：put([])、get(0)、take_batch(..., 0)、is_ready(..., 0)
#    不应改变 buffer 状态，也不应报错。
# 3. 写入生成结果时会补齐训练版本信息：put(..., model_step, current_train_step)
#    会补齐 response_model_steps，并刷新 seq_staleness。
# 4. tail batch disabled 时将 EXPIRED 当作终态，释放重字段且不写入 buffer；enabled 时保留
#    prompt/mm_info，只重置 response 和 routed experts 以便重新 rollout。
# 5. 写入过期结果时会触发 rerollout：超过 stale_threshold 的 group 会被重置 response 相关字段，
#    并保留 prompt/message 等重新 rollout 所需的输入字段。
# 6. refresh_staleness 的公共契约：可以刷新 completed/aborted 记录，也要尊重显式传入的
#    status 过滤条件。
# 7. SyncReplayBufferConfig 的采样策略：按 FIFO 顺序返回 group。
# 8. AsyncReplayBufferConfig 的采样策略：优先返回 seq_staleness 更高的 group；
#    staleness 相同时使用 FIFO 作为 tie-breaker。
# 9. save/resume 保留采样顺序：sync 恢复后仍是 FIFO，async 恢复后仍按 staleness 排序。
# 10. save/resume 保留真实 RolloutState 字段：状态、response、tokens、logprobs、reward、
#    error_msg、extra_fields 等字段恢复后应一致。
# 11. save/resume 保留 Ray ObjectRef：直接 ObjectRef 和 dict(dict(ObjectRef)) 嵌套结构恢复后，
#     解引用得到的内容都应与保存前一致。

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig


REPLAY_BUFFER_CONFIGS = [
    ("sync_fifo", SyncReplayBufferConfig),
    ("async_staleness", AsyncReplayBufferConfig),
]


def make_rollout_state(
    uid: int,
    *,
    session_id: int | None = None,
    status: Status = Status.COMPLETED,
    seq_staleness: int = 0,
    prompt_ids: list[int] | None = None,
    response: str | None = None,
    response_ids: list[int] | None = None,
    response_model_steps: list[int] | None = None,
    response_mask: list[int] | None = None,
    logprobs: list[float] | None = None,
    reward: dict | None = None,
    error_msg: str | None = None,
    tokens: list[int] | None = None,
    routed_experts=None,
    mm_info: dict | None = None,
    extra_fields: dict | None = None,
    input_ids: list[int] | None = None,
    labels: list[int] | None = None,
) -> RolloutState:
    prompt_ids = list(prompt_ids) if prompt_ids is not None else [uid, uid + 1000]
    response_ids = list(response_ids) if response_ids is not None else [uid + 10]
    logprobs = list(logprobs) if logprobs is not None else [0.1 for _ in response_ids]
    return RolloutState(
        rollout_id=uid,
        group_id=uid,
        message=[{"role": "user", "content": f"prompt {uid}"}],
        prompt_ids=prompt_ids,
        session_id=session_id,
        tokens=list(tokens) if tokens is not None else list(prompt_ids),
        response=response if response is not None else f"response {uid}",
        response_ids=response_ids,
        response_model_steps=list(response_model_steps) if response_model_steps is not None else None,
        response_mask=list(response_mask) if response_mask is not None else [1 for _ in response_ids],
        logprobs=logprobs,
        routed_experts=routed_experts,
        finish_reason="stop" if status == Status.COMPLETED else None,
        reward=reward,
        error_msg=error_msg,
        seq_staleness=seq_staleness,
        status=status,
        mm_info=mm_info,
        extra_fields=dict(extra_fields or {}),
        input_ids=input_ids,
        labels=labels,
    )


def group_uids(groups: list[list[RolloutState]]) -> list[list[int]]:
    return [[state.rollout_id for state in group] for group in groups]


async def save_and_resume(
    replay_buffer_config_cls,
    save_path: Path,
    groups: list[list[RolloutState]],
    *,
    task_name: str = "task",
):
    original = replay_buffer_config_cls().build()
    for group in groups:
        await original.put(group, task_name)
    await original.save(save_path)

    resumed = replay_buffer_config_cls().build()
    await resumed.resume(save_path)
    return resumed


class TestReplayBuffer(unittest.IsolatedAsyncioTestCase):
    async def test_common_query_count_and_take_batch_contract(self):
        # ReplayBuffer 的公共读写契约：按 task/status 隔离统计，并且 take_batch 会消费已取出的数据。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                await replay_buffer.put([make_rollout_state(101, seq_staleness=1)], "task_a")
                await replay_buffer.put([make_rollout_state(102, seq_staleness=5)], "task_a")
                await replay_buffer.put([make_rollout_state(103, status=Status.FAILED)], "task_a")
                await replay_buffer.put([make_rollout_state(201, seq_staleness=2)], "task_b")

                assert await replay_buffer.count("task_a", Status.COMPLETED) == 2
                assert await replay_buffer.count("task_a", Status.FAILED) == 1
                assert await replay_buffer.count("task_b", Status.COMPLETED) == 1
                assert await replay_buffer.count("task_b", Status.FAILED) == 0
                assert await replay_buffer.is_ready({"task_a": 2, "task_b": 1})
                assert not await replay_buffer.is_ready({"task_a": 3})

                status_counts = await replay_buffer.count_statuses(
                    ["task_a", "task_b"],
                    [Status.COMPLETED, Status.FAILED],
                )
                assert status_counts["task_a"] == {Status.COMPLETED: 2, Status.FAILED: 1}
                assert status_counts["task_b"] == {Status.COMPLETED: 1, Status.FAILED: 0}

                batch_by_task, consumed_counts = await replay_buffer.take_batch(
                    {"task_a": 1, "task_b": 1, "task_c": 0}
                )

                assert consumed_counts == {"task_a": 1, "task_b": 1, "task_c": 0}
                assert len(batch_by_task["task_a"]) == 1
                assert len(batch_by_task["task_b"]) == 1
                assert batch_by_task["task_c"] == []
                assert batch_by_task["task_a"][0][0].status == Status.COMPLETED
                assert batch_by_task["task_b"][0][0].status == Status.COMPLETED
                assert await replay_buffer.count("task_a", Status.COMPLETED) == 1
                assert await replay_buffer.count("task_b", Status.COMPLETED) == 0
                assert await replay_buffer.count("task_a", Status.FAILED) == 1

    async def test_common_empty_and_zero_size_operations_are_noops(self):
        # 空写入和 0-size 读取不应该制造数据，也不应该阻塞上层 batch 编排。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                await replay_buffer.put([], "task")

                assert len(replay_buffer) == 0
                assert await replay_buffer.get(0, "task", Status.COMPLETED) == []
                assert await replay_buffer.count("task", Status.COMPLETED) == 0
                assert await replay_buffer.is_ready({"task": 0})

                batch_by_task, consumed_counts = await replay_buffer.take_batch({"task": 0})
                assert batch_by_task == {"task": []}
                assert consumed_counts == {"task": 0}

    async def test_common_put_normalizes_generated_rollout_state(self):
        # 入库时 replay buffer 会补齐 response token 的 model_step，并刷新训练侧可见的 staleness。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                generated = make_rollout_state(
                    1,
                    response_ids=[11, 12],
                    response_model_steps=[],
                    seq_staleness=99,
                )

                await replay_buffer.put(
                    [generated],
                    "task",
                    model_step=3,
                    current_train_step=5,
                )

                assert generated.response_model_steps == [3, 3]
                assert generated.seq_staleness == 1
                completed = await replay_buffer.get(1, "task", Status.COMPLETED)
                assert completed[0][0].response_model_steps == [3, 3]
                assert completed[0][0].seq_staleness == 1

    async def test_common_put_drops_expired_group_when_tail_batch_is_disabled(self):
        # tail batch disabled 时 EXPIRED 是终态，释放重字段后不再写入 replay buffer。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                stale = make_rollout_state(
                    1,
                    session_id=101,
                    prompt_ids=[101, 102],
                    tokens=[999],
                    response="stale response",
                    response_ids=[11, 12],
                    response_model_steps=[1, 1],
                    logprobs=[0.2, 0.3],
                    reward={"score": 1.0},
                    error_msg="old error",
                    mm_info={"pixel_values": np.ones((2, 3), dtype=np.float32)},
                    extra_fields={"train_prompt_ids": [101, 102]},
                )

                with patch(
                    "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
                    new=AsyncMock(return_value={"101"}),
                ) as release_sessions:
                    await replay_buffer.put(
                        [stale],
                        "task",
                        current_train_step=5,
                        stale_threshold=3,
                        expired_groups_retryable=False,
                    )

                release_sessions.assert_awaited_once_with(["101"])
                assert stale.status == Status.EXPIRED
                assert stale.prompt_ids is None
                assert stale.tokens is None
                assert stale.mm_info is None
                assert stale.extra_fields == {}
                assert await replay_buffer.count("task", Status.EXPIRED) == 0
                assert len(replay_buffer) == 0
                assert await replay_buffer.get(1, "task", Status.EXPIRED) == []

    async def test_common_put_defaults_to_retryable_expired_group(self):
        # standalone ReplayBuffer 不传 retryability 时保留原有语义：EXPIRED 可 rerollout。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                pixel_values = np.ones((2, 3), dtype=np.float32)
                stale = make_rollout_state(
                    1,
                    session_id=102,
                    prompt_ids=[101, 102],
                    tokens=[999],
                    response="stale response",
                    response_ids=[11, 12],
                    response_model_steps=[1, 1],
                    logprobs=[0.2, 0.3],
                    reward={"score": 1.0},
                    error_msg="old error",
                    routed_experts=np.ones((2, 2), dtype=np.int64),
                    mm_info={"pixel_values": pixel_values},
                    extra_fields={"train_prompt_ids": [101, 102]},
                )

                with patch(
                    "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
                    new=AsyncMock(),
                ) as release_sessions:
                    await replay_buffer.put(
                        [stale],
                        "task",
                        current_train_step=5,
                        stale_threshold=3,
                    )

                release_sessions.assert_not_awaited()
                expired = await replay_buffer.get(1, "task", Status.EXPIRED)
                reusable = expired[0][0]
                assert reusable.status == Status.EXPIRED
                assert reusable.seq_staleness == 3
                assert reusable.prompt_ids == [101, 102]
                assert reusable.tokens == [101, 102]
                assert reusable.response == ""
                assert reusable.response_ids == []
                assert reusable.response_model_steps == []
                assert reusable.logprobs == []
                assert reusable.reward is None
                assert reusable.error_msg is None
                assert reusable.routed_experts is None
                assert reusable.finish_reason is None
                assert reusable.response_mask is None
                assert reusable.mm_info is not None
                assert reusable.mm_info["pixel_values"] is pixel_values
                assert reusable.extra_fields == {"train_prompt_ids": [101, 102]}

    async def test_common_put_token_expiry_preserves_fresh_group_members(self):
        # token expiry 只清理真正全 token 过期的 state，外层 group 仍进入 EXPIRED pool。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                expired = make_rollout_state(
                    1,
                    response="expired response",
                    response_ids=[11, 12],
                    response_model_steps=[0, 0],
                    reward={"score": 0.1},
                )
                fresh = make_rollout_state(
                    2,
                    response="fresh response",
                    response_ids=[21, 22],
                    response_model_steps=[4, 4],
                    reward={"score": 0.9},
                )

                await replay_buffer.put(
                    [expired, fresh],
                    "task",
                    current_train_step=5,
                    stale_threshold=10,
                    token_stale_threshold=4,
                    expired_groups_retryable=True,
                )

                self.assertEqual(await replay_buffer.count("task", Status.COMPLETED), 0)
                self.assertEqual(await replay_buffer.count("task", Status.EXPIRED), 1)
                group = (await replay_buffer.get(1, "task", Status.EXPIRED))[0]
                self.assertEqual([item.status for item in group], [Status.EXPIRED, Status.COMPLETED])
                self.assertEqual(group[0].response, "")
                self.assertEqual(group[0].response_ids, [])
                self.assertIsNone(group[0].reward)
                self.assertEqual(group[1].response, "fresh response")
                self.assertEqual(group[1].response_ids, [21, 22])
                self.assertEqual(group[1].response_model_steps, [4, 4])
                self.assertEqual(group[1].reward, {"score": 0.9})

    async def test_common_put_skips_token_expiry_for_agentic_group(self):
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                state = make_rollout_state(
                    1,
                    response="agentic response",
                    response_model_steps=[0],
                    input_ids=[1, 2],
                    labels=[-100, 2],
                )

                await replay_buffer.put(
                    [state],
                    "task",
                    current_train_step=5,
                    stale_threshold=10,
                    token_stale_threshold=4,
                    expired_groups_retryable=True,
                )

                self.assertEqual(await replay_buffer.count("task", Status.COMPLETED), 1)
                self.assertEqual(await replay_buffer.count("task", Status.EXPIRED), 0)
                self.assertEqual(state.status, Status.COMPLETED)
                self.assertEqual(state.response, "agentic response")

    async def test_common_put_seq_expiry_preserves_fresh_group_members(self):
        # seq expiry 路由整组到 EXPIRED pool，但只标记和清理实际过期的 state。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                stale = make_rollout_state(
                    1,
                    response="stale response",
                    response_model_steps=[0],
                )
                fresh = make_rollout_state(
                    2,
                    response="fresh response",
                    response_model_steps=[4],
                )

                await replay_buffer.put(
                    [stale, fresh],
                    "task",
                    current_train_step=5,
                    stale_threshold=4,
                    expired_groups_retryable=True,
                )

                group = (await replay_buffer.get(1, "task", Status.EXPIRED))[0]
                self.assertEqual([item.status for item in group], [Status.EXPIRED, Status.COMPLETED])
                self.assertEqual([item.response for item in group], ["", "fresh response"])

    async def test_common_put_drops_entire_token_expired_group_when_rerollout_is_disabled(self):
        # 无 rerollout consumer 时，混合 group 整体终止并释放，缺口由新 prompt 补齐。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                expired = make_rollout_state(1, response_model_steps=[0])
                fresh = make_rollout_state(2, response_model_steps=[4])

                await replay_buffer.put(
                    [expired, fresh],
                    "task",
                    current_train_step=5,
                    stale_threshold=10,
                    token_stale_threshold=4,
                    expired_groups_retryable=False,
                )

                self.assertEqual(await replay_buffer.count("task", Status.EXPIRED), 0)
                self.assertEqual(len(replay_buffer), 0)
                self.assertEqual([item.status for item in (expired, fresh)], [Status.EXPIRED, Status.EXPIRED])
                self.assertIsNone(expired.prompt_ids)
                self.assertIsNone(fresh.prompt_ids)

    async def test_common_refresh_token_expiry_moves_mixed_group_to_expired_pool(self):
        # batch-start refresh 使用 consumer step 重新判定，并保留仍新鲜 state 的完整训练数据。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                expired = make_rollout_state(1, response_model_steps=[0], reward={"score": 0.1})
                fresh = make_rollout_state(2, response_model_steps=[4], reward={"score": 0.9})
                await replay_buffer.put([expired, fresh], "task")

                expired_counts = await replay_buffer.refresh_staleness(
                    task_stale_thresholds={"task": 10},
                    task_token_stale_thresholds={"task": 4},
                    expired_groups_retryable_by_task={"task": True},
                    current_train_step=5,
                )

                self.assertEqual(expired_counts, {"task": 1})
                self.assertEqual(await replay_buffer.count("task", Status.COMPLETED), 0)
                self.assertEqual(await replay_buffer.count("task", Status.EXPIRED), 1)
                group = (await replay_buffer.get(1, "task", Status.EXPIRED))[0]
                self.assertEqual([item.status for item in group], [Status.EXPIRED, Status.COMPLETED])
                self.assertEqual(group[0].response_ids, [])
                self.assertEqual(group[1].response_ids, [12])
                self.assertEqual(group[1].reward, {"score": 0.9})

    async def test_common_refresh_staleness_drops_only_non_retryable_expired_groups(self):
        # 同一轮 refresh 仍统计两类过期；只删除 non-retryable EXPIRED，保留 tail batch 可重试项。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                non_retryable_stale = make_rollout_state(
                    1,
                    session_id=201,
                    response_model_steps=[1],
                    mm_info={"pixel_values": np.ones((2, 3), dtype=np.float32)},
                )
                retryable_stale = make_rollout_state(
                    2,
                    session_id=202,
                    response_model_steps=[1],
                    mm_info={"pixel_values": np.ones((2, 3), dtype=np.float32)},
                )
                await replay_buffer.put([non_retryable_stale], "non_retryable_task")
                await replay_buffer.put([retryable_stale], "retryable_task")
                assert len(replay_buffer) == 2

                with patch(
                    "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
                    new=AsyncMock(return_value={"201"}),
                ) as release_sessions:
                    expired_counts = await replay_buffer.refresh_staleness(
                        task_stale_thresholds={"non_retryable_task": 2, "retryable_task": 2},
                        expired_groups_retryable_by_task={
                            "non_retryable_task": False,
                            "retryable_task": True,
                        },
                        current_train_step=4,
                    )

                release_sessions.assert_awaited_once_with(["201"])
                assert expired_counts == {"non_retryable_task": 1, "retryable_task": 1}
                assert non_retryable_stale.status == Status.EXPIRED
                assert non_retryable_stale.prompt_ids is None
                assert non_retryable_stale.mm_info is None
                assert retryable_stale.status == Status.EXPIRED
                assert retryable_stale.prompt_ids == [2, 1002]
                assert retryable_stale.mm_info is not None
                assert await replay_buffer.count("non_retryable_task", Status.COMPLETED) == 0
                assert await replay_buffer.count("non_retryable_task", Status.EXPIRED) == 0
                assert await replay_buffer.count("retryable_task", Status.EXPIRED) == 1
                assert len(replay_buffer) == 1
                assert await replay_buffer.get(1, "non_retryable_task", Status.EXPIRED) == []

    async def test_refresh_staleness_batches_non_retryable_release_outside_lock(self):
        replay_buffer = AsyncReplayBufferConfig().build()
        first = make_rollout_state(1, session_id=301, response_model_steps=[1])
        second = make_rollout_state(2, session_id=302, response_model_steps=[1])
        await replay_buffer.put([first], "non_retryable_task")
        await replay_buffer.put([second], "non_retryable_task")

        release_started = asyncio.Event()
        allow_release = asyncio.Event()

        async def delayed_release(session_ids):
            release_started.set()
            await allow_release.wait()
            return set(session_ids)

        with patch(
            "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
            new=AsyncMock(side_effect=delayed_release),
        ) as release_sessions:
            refresh_task = asyncio.create_task(
                replay_buffer.refresh_staleness(
                    task_stale_thresholds={"non_retryable_task": 2},
                    expired_groups_retryable_by_task={"non_retryable_task": False},
                    current_train_step=4,
                )
            )
            await asyncio.wait_for(release_started.wait(), timeout=1.0)
            try:
                # The non-retryable records are already removed and the buffer lock is
                # available while the trace-store RPC is still blocked.
                count = await asyncio.wait_for(
                    replay_buffer.count("non_retryable_task", Status.COMPLETED),
                    timeout=1.0,
                )
                assert count == 0
            finally:
                allow_release.set()
            expired_counts = await refresh_task

        release_sessions.assert_awaited_once_with(["301", "302"])
        assert expired_counts == {"non_retryable_task": 2}
        assert first.status == Status.EXPIRED
        assert second.status == Status.EXPIRED
        assert len(replay_buffer) == 0

    async def test_common_refresh_staleness_contract(self):
        # refresh_staleness 同时覆盖默认刷新 completed/aborted，以及 status filter 只刷新指定状态。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                await replay_buffer.put([make_rollout_state(1, response_model_steps=[1])], "task")
                await replay_buffer.put(
                    [make_rollout_state(2, status=Status.ABORTED, response_model_steps=[1])], "task"
                )

                expired_counts = await replay_buffer.refresh_staleness(
                    task_stale_thresholds={"task": 2},
                    current_train_step=4,
                )

                assert expired_counts == {"task": 2}
                assert await replay_buffer.count("task", Status.COMPLETED) == 0
                assert await replay_buffer.count("task", Status.ABORTED) == 0
                assert await replay_buffer.count("task", Status.EXPIRED) == 2
                expired = await replay_buffer.get(2, "task", Status.EXPIRED)
                assert {state.rollout_id for group in expired for state in group} == {1, 2}

                filtered_buffer = replay_buffer_config_cls().build()
                await filtered_buffer.put([make_rollout_state(3, response_model_steps=[1])], "task")
                await filtered_buffer.put(
                    [make_rollout_state(4, status=Status.ABORTED, response_model_steps=[1])], "task"
                )

                filtered_counts = await filtered_buffer.refresh_staleness(
                    task_stale_thresholds={"task": 2},
                    current_train_step=4,
                    statuses=[Status.ABORTED],
                )

                assert filtered_counts == {"task": 1}
                assert await filtered_buffer.count("task", Status.COMPLETED) == 1
                assert await filtered_buffer.count("task", Status.ABORTED) == 0
                assert await filtered_buffer.count("task", Status.EXPIRED) == 1
                completed = await filtered_buffer.get(1, "task", Status.COMPLETED)
                expired = await filtered_buffer.get(1, "task", Status.EXPIRED)
                assert completed[0][0].rollout_id == 3
                assert expired[0][0].rollout_id == 4

    async def test_sync_get_returns_fifo_order(self):
        # Sync replay 用于共卡按需生产，策略契约是同 task/status 下严格按入库顺序消费。
        replay_buffer = SyncReplayBufferConfig().build()
        await replay_buffer.put([make_rollout_state(1), make_rollout_state(2)], "task")
        await replay_buffer.put([make_rollout_state(3)], "task")
        await replay_buffer.put([make_rollout_state(4, status=Status.FAILED)], "task")

        completed = await replay_buffer.get(2, "task", Status.COMPLETED)
        failed = await replay_buffer.get(1, "task", Status.FAILED)

        assert group_uids(completed) == [[1, 2], [3]]
        assert group_uids(failed) == [[4]]

    async def test_async_get_prefers_higher_staleness(self):
        # Async replay 优先消费更旧模型生成的样本，帮助训练侧尽快清理高 staleness 数据。
        replay_buffer = AsyncReplayBufferConfig().build()
        await replay_buffer.put([make_rollout_state(1, seq_staleness=1)], "task")
        await replay_buffer.put([make_rollout_state(2, seq_staleness=5)], "task")
        await replay_buffer.put([make_rollout_state(3, seq_staleness=3)], "task")

        completed = await replay_buffer.get(3, "task", Status.COMPLETED)

        assert group_uids(completed) == [[2], [3], [1]]

    async def test_async_get_uses_fifo_as_staleness_tie_breaker(self):
        # staleness 相同的 async 样本仍按入库顺序消费，避免同版本样本被重排成不可预测顺序。
        replay_buffer = AsyncReplayBufferConfig().build()
        await replay_buffer.put([make_rollout_state(1, seq_staleness=3)], "task")
        await replay_buffer.put([make_rollout_state(2, seq_staleness=8)], "task")
        await replay_buffer.put([make_rollout_state(3, seq_staleness=3)], "task")
        await replay_buffer.put([make_rollout_state(4, seq_staleness=3)], "task")

        completed = await replay_buffer.get(4, "task", Status.COMPLETED)

        assert group_uids(completed) == [[2], [1], [3], [4]]

    async def test_sync_save_resume_preserves_fifo_sampling_order(self):
        # save/resume 后 Sync replay 的 FIFO 消费顺序必须不变，否则 checkpoint resume 会改变训练数据顺序。
        with tempfile.TemporaryDirectory() as tmp_dir:
            resumed = await save_and_resume(
                SyncReplayBufferConfig,
                Path(tmp_dir),
                [
                    [make_rollout_state(1)],
                    [make_rollout_state(2)],
                    [make_rollout_state(3)],
                ],
            )

            completed = await resumed.get(3, "task", Status.COMPLETED)
            assert group_uids(completed) == [[1], [2], [3]]

    async def test_async_save_resume_preserves_staleness_sampling_order(self):
        # save/resume 后 Async replay 仍要按 staleness 排序，否则恢复训练会消费不同优先级的数据。
        with tempfile.TemporaryDirectory() as tmp_dir:
            resumed = await save_and_resume(
                AsyncReplayBufferConfig,
                Path(tmp_dir),
                [
                    [make_rollout_state(1, seq_staleness=1)],
                    [make_rollout_state(2, seq_staleness=5)],
                    [make_rollout_state(3, seq_staleness=3)],
                ],
            )

            completed = await resumed.get(3, "task", Status.COMPLETED)
            assert group_uids(completed) == [[2], [3], [1]]

    async def test_save_resume_preserves_rollout_state_fields(self):
        # save/resume 应保留真实 RolloutState 字段，不再用 MockState.input_ids 代表训练样本内容。
        def state_signature(state: RolloutState) -> tuple:
            return (
                state.rollout_id,
                tuple(state.prompt_ids or []),
                tuple(state.response_ids or []),
                tuple(state.response_model_steps or []),
                state.seq_staleness,
                state.status,
            )

        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                original_state = make_rollout_state(
                    1,
                    prompt_ids=[101, 102],
                    response_ids=[201, 202],
                    response_model_steps=[3, 3],
                    seq_staleness=4,
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    resumed = await save_and_resume(
                        replay_buffer_config_cls,
                        Path(tmp_dir),
                        [[original_state]],
                    )

                    completed = await resumed.get(1, "task", Status.COMPLETED)
                    assert state_signature(completed[0][0]) == state_signature(original_state)

    async def test_save_resume_preserves_object_refs(self):
        # replay buffer checkpoint 需要递归展开并恢复 RolloutState 里直接和嵌套存放的 Ray ObjectRef。
        started_ray = False
        try:
            if not ray.is_initialized():
                # ObjectRef save/resume coverage needs an isolated local Ray cluster.
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                started_ray = True
        except Exception as exc:
            self.skipTest(f"Ray init failed for replay buffer ObjectRef test: {exc}")

        try:
            for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
                with self.subTest(replay_buffer_config=config_name):
                    routed_experts = np.array([[1, 2], [3, 4]], dtype=np.int64)
                    nested_payload = {"tokens": [11, 12], "scores": [0.25, 0.75]}
                    state = make_rollout_state(
                        1,
                        routed_experts=ray.put(routed_experts),
                        extra_fields={"outer": {"inner": ray.put(nested_payload)}},
                    )

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        resumed = await save_and_resume(
                            replay_buffer_config_cls,
                            Path(tmp_dir),
                            [[state]],
                        )

                        completed = await resumed.get(1, "task", Status.COMPLETED)
                        restored = completed[0][0]

                        assert isinstance(restored.routed_experts, ray.ObjectRef)
                        np.testing.assert_array_equal(ray.get(restored.routed_experts), routed_experts)

                        nested_ref = restored.extra_fields["outer"]["inner"]
                        assert isinstance(nested_ref, ray.ObjectRef)
                        assert ray.get(nested_ref) == nested_payload
        finally:
            if started_ray and ray.is_initialized():
                ray.shutdown()
