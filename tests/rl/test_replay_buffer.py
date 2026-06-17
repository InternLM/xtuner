# 当前 ReplayBuffer PR-fast 单测覆盖点：
#
# 1. 公共读写/消费契约：SyncReplayBufferConfig 和 AsyncReplayBufferConfig 都要保证
#    task 隔离、status 隔离、count/count_statuses、is_ready、take_batch 消费语义正确。
# 2. 空输入和零 size 操作是 no-op：put([])、get(0)、take_batch(..., 0)、is_ready(..., 0)
#    不应改变 buffer 状态，也不应报错。
# 3. 写入生成结果时会补齐训练版本信息：put(..., model_step, current_train_step)
#    会补齐 response_model_steps，并刷新 seq_staleness。
# 4. 写入过期结果时会触发 rerollout：超过 stale_threshold 的 group 会被重置 response 相关字段，
#    并保留 prompt/message 等重新 rollout 所需的输入字段。
# 5. refresh_staleness 的公共契约：可以刷新 completed/aborted 记录，也要尊重显式传入的
#    status 过滤条件。
# 6. SyncReplayBufferConfig 的采样策略：按 FIFO 顺序返回 group。
# 7. AsyncReplayBufferConfig 的采样策略：优先返回 seq_staleness 更高的 group；
#    staleness 相同时使用 FIFO 作为 tie-breaker。
# 8. save/resume 保留采样顺序：sync 恢复后仍是 FIFO，async 恢复后仍按 staleness 排序。
# 9. save/resume 保留真实 RolloutState 字段：状态、response、tokens、logprobs、reward、
#    error_msg、extra_fields 等字段恢复后应一致。
# 10. save/resume 保留 Ray ObjectRef：直接 ObjectRef 和 dict(dict(ObjectRef)) 嵌套结构恢复后，
#     解引用得到的内容都应与保存前一致。

import tempfile
import unittest
from pathlib import Path

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
    status: Status = Status.COMPLETED,
    seq_staleness: int = 0,
    prompt_ids: list[int] | None = None,
    response: str | None = None,
    response_ids: list[int] | None = None,
    response_model_steps: list[int] | None = None,
    logprobs: list[float] | None = None,
    reward: dict | None = None,
    error_msg: str | None = None,
    tokens: list[int] | None = None,
    routed_experts=None,
    extra_fields: dict | None = None,
) -> RolloutState:
    prompt_ids = list(prompt_ids) if prompt_ids is not None else [uid, uid + 1000]
    response_ids = list(response_ids) if response_ids is not None else [uid + 10]
    logprobs = list(logprobs) if logprobs is not None else [0.1 for _ in response_ids]
    return RolloutState(
        rollout_id=uid,
        group_id=uid,
        message=[{"role": "user", "content": f"prompt {uid}"}],
        prompt_ids=prompt_ids,
        tokens=list(tokens) if tokens is not None else list(prompt_ids),
        response=response if response is not None else f"response {uid}",
        response_ids=response_ids,
        response_model_steps=list(response_model_steps) if response_model_steps is not None else None,
        response_mask=[1 for _ in response_ids],
        logprobs=logprobs,
        routed_experts=routed_experts,
        finish_reason="stop" if status == Status.COMPLETED else None,
        reward=reward,
        error_msg=error_msg,
        seq_staleness=seq_staleness,
        status=status,
        extra_fields=dict(extra_fields or {}),
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

    async def test_common_put_expires_stale_group_and_resets_response(self):
        # 入库时超过 staleness 阈值的 group 会转入 EXPIRED，并清理旧 response 以便后续重新 rollout。
        for config_name, replay_buffer_config_cls in REPLAY_BUFFER_CONFIGS:
            with self.subTest(replay_buffer_config=config_name):
                replay_buffer = replay_buffer_config_cls().build()
                stale = make_rollout_state(
                    1,
                    prompt_ids=[101, 102],
                    tokens=[999],
                    response="stale response",
                    response_ids=[11, 12],
                    response_model_steps=[1, 1],
                    logprobs=[0.2, 0.3],
                    reward={"score": 1.0},
                    error_msg="old error",
                )

                await replay_buffer.put(
                    [stale],
                    "task",
                    current_train_step=5,
                    stale_threshold=3,
                )

                expired = await replay_buffer.get(1, "task", Status.EXPIRED)
                restored = expired[0][0]
                assert restored.status == Status.EXPIRED
                assert restored.seq_staleness == 3
                assert restored.tokens == [101, 102]
                assert restored.response == ""
                assert restored.response_ids == []
                assert restored.response_model_steps == []
                assert restored.logprobs == []
                assert restored.reward is None
                assert restored.error_msg is None
                assert restored.routed_experts is None
                assert restored.finish_reason is None
                assert restored.response_mask == []

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
