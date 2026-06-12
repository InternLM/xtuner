"""Sampler / ProduceContext / ProduceStrategy 的行为测试。

Good Tests:
- 通过 Sampler、ProduceContext、SyncProduceStrategy、AsyncProduceStrategy 的公开入口验证生产行为。
- 断言 replay buffer 中可见的 Rollout Group 状态、ProduceBatchStatus、版本/staleness/metrics 等业务结果。
- 对复杂异步行为只保留对最终可观察结果的断言；_PendingTasks 的并发协议放在独立测试文件中。

Bad Tests:
- 不直接测试 _PendingTasks 的内部集合或 claim/cancel 细节。
- 不把 sampler 调用次数、pending task 数量、mock 调用顺序当成核心契约，除非它们是当前 public 行为的唯一可观测信号。
- 不测试 AgentLoopManager 的状态机；manager 编排行为放在 test_multi_task_agent_loop_manager.py。

本文件主要覆盖的 public 行为:
- sampler 优先复用可重试 Rollout Group，耗尽后回退 dataloader。
- ProduceContext 统一处理生成结果落库、过滤、raw reward 和模型版本记录。
- SyncProduceStrategy / AsyncProduceStrategy 通过共卡入口完成生产，不返回状态控制信号。
- DisaggAsyncProduceStrategy 返回 UPDATE_WEIGHT_AND_ABORT、EXPIRED_BATCH 的后台生产状态。
- AsyncProduceStrategy 的 oversample、tail-batch、partial rollout、pause drain 和 staleness 结果。
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    DisaggAsyncProduceStrategyConfig,
    DisaggProduceContext,
    DisaggProduceProgress,
    ProduceBatchStatus,
    ProduceContext,
    ProduceProgress,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


def make_rollout_state(
    uid: int,
    *,
    seq_staleness: int = 1,
    status: Status = Status.COMPLETED,
    reward_score: float | None = None,
) -> RolloutState:
    return RolloutState(
        uid=uid,
        message_uid=uid,
        message=[{"role": "user", "content": f"prompt {uid}"}],
        prompt_ids=[uid],
        tokens=[uid],
        response="" if status in (Status.ABORTED, Status.EXPIRED) else f"response {uid}",
        response_ids=[],
        response_mask=[],
        finish_reason="abort" if status == Status.ABORTED else "stop",
        reward={"score": reward_score} if reward_score is not None else None,
        seq_staleness=seq_staleness,
        status=status,
        extra_fields={},
    )


class TestProducer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 1. 模拟 DataloaderConfig 和 Dataloader
        self.mock_dataloader_cfg = MagicMock()
        self.mock_dataloader = MagicMock()
        # 模拟 next(dataloader_iter) 返回 [RolloutState]
        self.mock_dataloader.__iter__.return_value = iter([[make_rollout_state(i)] for i in range(100)])
        self.mock_dataloader_cfg.build.return_value = self.mock_dataloader

        # 2. 模拟 Tokenizer
        self.mock_tokenizer = MagicMock()

        # 3. 准备 ReplayBuffer
        replay_buffer_cfg = AsyncReplayBufferConfig()
        self.replay_buffer = replay_buffer_cfg.build()

    def _build_sampler(self):
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        return sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

    def _build_progress(
        self,
        task_name: str,
        target: int,
        consumed: int = 0,
        target_upto_future_step: int | None = None,
    ) -> ProduceProgress:
        if consumed != 0 or target_upto_future_step is not None:
            raise ValueError("Use _build_disagg_progress for absolute consumed/target progress.")
        return ProduceProgress.build(
            task_names=[task_name],
            target_samples={task_name: target},
        )

    def _build_agent_loop(self, sleep_by_id: dict[int, float] | None = None):
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        async def mock_pause():
            await mock_agent_loop.rollout_ctl.pause_generation.remote()

        mock_agent_loop.pause = mock_pause

        sleep_by_id = sleep_by_id or {}

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(sleep_by_id.get(rs[0].message_uid, 0.0))
            for r in rs:
                r.seq_staleness = kwargs.get("model_step", kwargs.get("train_step", 0))
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen
        return mock_agent_loop

    def _build_context(
        self,
        strategy,
        task_name: str,
        agent_loop,
        sampler,
        *,
        batch_size: int,
        train_step: int = 0,
        model_step: int = 0,
        progress: ProduceProgress | None = None,
    ) -> ProduceContext:
        # 测试只走新的 ProduceContext 入口，不再覆盖旧散装参数兼容逻辑。
        if progress is None:
            progress = self._build_progress(task_name, target=batch_size)
        return ProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=self.replay_buffer,
            task_batch_size=batch_size,
            task_name=task_name,
            train_step=train_step,
            model_step=model_step,
            progress=progress,
            is_valid_sample_fn=strategy.is_valid_sample_fn,
            stale_threshold=getattr(strategy, "stale_threshold", None),
        )

    def _build_disagg_progress(
        self,
        task_name: str,
        target: int,
        train_step: int = 0,
        consumed: int = 0,
        producer_future_step: int | None = None,
        target_upto_future_step: int | None = None,
    ) -> DisaggProduceProgress:
        progress = DisaggProduceProgress.build([task_name])
        progress.next_consumer_step = train_step
        progress.producer_future_step = producer_future_step if producer_future_step is not None else train_step
        progress.consumed_samples[task_name] = consumed
        progress.target_samples[task_name] = target
        progress.target_upto_future_step = (
            target_upto_future_step if target_upto_future_step is not None else train_step
        )
        return progress

    def _build_disagg_context(
        self,
        strategy,
        task_name: str,
        agent_loop,
        sampler,
        *,
        batch_size: int,
        train_step: int = 0,
        model_step: int = 0,
        progress: DisaggProduceProgress | None = None,
        update_event: asyncio.Event | None = None,
    ) -> DisaggProduceContext:
        if progress is None:
            progress = self._build_disagg_progress(task_name, target=batch_size, train_step=train_step)
        if update_event is None:
            update_event = asyncio.Event()
        return DisaggProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=self.replay_buffer,
            task_batch_size=batch_size,
            task_name=task_name,
            train_step=train_step,
            update_event=update_event,
            model_step=model_step,
            progress=progress,
            is_valid_sample_fn=strategy.is_valid_sample_fn,
            stale_threshold=getattr(strategy, "stale_threshold", None),
        )

    async def test_contexts_keep_colocate_and_disagg_control_surface_separate(self):
        # 共卡 context 只表达一次本地生产窗口；update_event / abort / 绝对累计进度只属于非共卡 context。
        task_name = "test_context_surface"
        sampler = self._build_sampler()
        agent_loop = self._build_agent_loop()

        colocate_strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        colocate_ctx = self._build_context(
            colocate_strategy,
            task_name,
            agent_loop,
            sampler,
            batch_size=1,
            train_step=3,
            model_step=2,
            progress=ProduceProgress.build(
                task_names=[task_name],
                target_samples={task_name: 1},
            ),
        )
        self.assertEqual(colocate_ctx.batch_target, 1)
        for disagg_only_name in ("update_event", "should_abort", "available_count", "total_target"):
            self.assertFalse(hasattr(colocate_ctx, disagg_only_name), disagg_only_name)

        disagg_strategy = DisaggAsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        update_event = asyncio.Event()
        disagg_ctx = self._build_disagg_context(
            disagg_strategy,
            task_name,
            agent_loop,
            sampler,
            batch_size=1,
            train_step=3,
            model_step=2,
            progress=self._build_disagg_progress(task_name, target=2, train_step=3, consumed=1),
            update_event=update_event,
        )
        self.assertEqual(disagg_ctx.total_target, 2)
        self.assertEqual(await disagg_ctx.available_count(), 1)
        self.assertFalse(disagg_ctx.should_abort())
        update_event.set()
        self.assertTrue(disagg_ctx.should_abort())

    async def test_sampler_with_replay_buffer(self):
        # 验证 sampler 优先复用 replay buffer 中可重试的 rollout group，耗尽后回退 dataloader。
        task_name = "test_task"
        sampler = self._build_sampler()

        # 场景 A: ReplayBuffer 为空，从 Dataloader 拿
        data = await sampler.sample(task_name)
        self.assertEqual(data[0].message_uid, 0)

        # 场景 B: ReplayBuffer 有多个候选状态，按列表顺序优先拿
        aborted_item = make_rollout_state(999, status=Status.ABORTED)
        expired_item = make_rollout_state(1000, status=Status.EXPIRED)
        await self.replay_buffer.put([aborted_item], task_name)
        await self.replay_buffer.put([expired_item], task_name)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].message_uid, 1000)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].message_uid, 999)

        # 场景 C: ReplayBuffer 对应状态都为空，回退到 Dataloader
        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].message_uid, 1)

    async def test_put_generated_group_only_validates_completed_group(self):
        # 验证 ProduceContext 只对 completed group 执行业务过滤，aborted group 保持可重试状态。
        task_name = "test_valid_completed_only"
        valid_checked_statuses = []

        def is_valid_sample_fn(samples):
            valid_checked_statuses.append([sample.status for sample in samples])
            return False

        strategy = SyncProduceStrategyConfig(is_valid_sample_fn=is_valid_sample_fn).build()
        ctx = self._build_context(
            strategy,
            task_name,
            self._build_agent_loop(),
            self._build_sampler(),
            batch_size=1,
        )

        completed_group = [make_rollout_state(1, status=Status.COMPLETED)]
        self.assertFalse(await ctx.put_generated_group(completed_group))
        self.assertEqual(completed_group[0].status, Status.FILTERED)

        aborted_group = [make_rollout_state(2, status=Status.ABORTED)]
        self.assertFalse(await ctx.put_generated_group(aborted_group))
        self.assertEqual(aborted_group[0].status, Status.ABORTED)

        self.assertEqual(valid_checked_statuses, [[Status.COMPLETED]])
        self.assertEqual(await self.replay_buffer.count(task_name, Status.FILTERED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 1)

    async def test_put_generated_group_records_raw_rewards_before_filtering(self):
        # 验证 raw reward 在过滤前统计，filtered group 仍能贡献生成侧 reward 指标。
        task_name = "test_raw_reward_before_filter"

        def is_valid_sample_fn(samples):
            return False

        strategy = SyncProduceStrategyConfig(is_valid_sample_fn=is_valid_sample_fn).build()
        ctx = self._build_context(
            strategy,
            task_name,
            self._build_agent_loop(),
            self._build_sampler(),
            batch_size=1,
        )

        completed_group = [
            make_rollout_state(1, status=Status.COMPLETED, reward_score=0.25),
            make_rollout_state(2, status=Status.COMPLETED, reward_score=0.75),
        ]
        self.assertFalse(await ctx.put_generated_group(completed_group))

        self.assertEqual([item.status for item in completed_group], [Status.FILTERED, Status.FILTERED])
        self.assertEqual(ctx.progress.consume_raw_rewards(task_name), (1.0, 2))
        self.assertEqual(ctx.progress.consume_raw_rewards(task_name), (0.0, 0))
        self.assertEqual(await self.replay_buffer.count(task_name, Status.FILTERED), 1)

    async def test_sync_produce_strategy(self):
        # 验证同步生产策略会生产指定数量的 completed rollout group 并写入 replay buffer。
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.0, 1: 0.01})
        produce_strategy_cfg = SyncProduceStrategyConfig()
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        # 执行：生产 batch_size 为 2 的数据
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            train_step=4,
            model_step=3,
            progress=self._build_progress(task_name, target=2),
        )
        await strategy.produce_batch(ctx)

        # 验证：ReplayBuffer 中应该有 2 条 COMPLETED 数据
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 2)
        self.assertEqual(final_data[0][0].message_uid, 0)
        self.assertEqual(final_data[1][0].message_uid, 1)

    async def test_sync_produce_strategy_refills_after_filtered_and_aborted_groups(self):
        # 验证 filtered / aborted group 不占用 completed quota，sync producer 会继续补齐训练 batch。
        task_name = "test_sync_refill"

        def is_valid_sample_fn(samples):
            return samples[0].message_uid != 0

        async def mock_gen(rs, **kwargs):
            for r in rs:
                if r.message_uid == 1:
                    r.status = Status.ABORTED
                    r.response = ""
                    r.response_ids = []
                else:
                    r.status = Status.COMPLETED
                    r.response = "ok"
                    r.response_ids = [1, 2]
                    r.reward = {"score": 1.0}
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        strategy = SyncProduceStrategyConfig(is_valid_sample_fn=is_valid_sample_fn).build()
        sampler = self._build_sampler()
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            train_step=4,
            model_step=3,
            progress=self._build_progress(task_name, target=2),
        )

        await strategy.produce_batch(ctx)
        completed = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(completed), 2)
        self.assertEqual(sorted(group[0].message_uid for group in completed), [2, 3])
        self.assertEqual(await self.replay_buffer.count(task_name, Status.FILTERED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 1)

    async def test_async_produce_strategy_oversamples_and_retries_aborted_groups(self):
        # 验证异步生产策略会按超发预算生产，并优先重试 replay buffer 中的 aborted group。
        # 这个async_produce_strategy的测试主要验证超发逻辑 + staleness 优先get的逻辑
        # 异步的其他功能如 partial_rollout, tail_batch不在这里进行验证
        mock_agent_loop = MagicMock()
        mock_agent_loop.pause = AsyncMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        task_name = "test_task"
        call_count = 0
        async def mock_gen(rs, **kwargs):
            nonlocal call_count
            call_count += 1
            for r in rs:
                if r.message_uid == 999:
                    r.seq_staleness = 5
                else:
                    r.seq_staleness = call_count
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen

        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=1)
        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)
        strategy = produce_strategy_cfg.build()
        # 预处理
        aborted_item = make_rollout_state(999, status=Status.ABORTED)
        await self.replay_buffer.put([aborted_item], task_name)
        # 执行
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )
        await strategy.produce_batch(ctx)

        # 验证：ReplayBuffer 中应该有 4 条 COMPLETED 数据。
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 4)
        self.assertEqual(sorted(group[0].message_uid for group in final_data), [0, 1, 2, 999])

    async def test_async_produce_strategy_accepts_context_entrypoint(self):
        # 验证 AsyncProduceStrategy 通过 ProduceContext public 入口完成一次最小生产。
        task_name = "test_context_entry"
        mock_agent_loop = self._build_agent_loop()
        sampler = self._build_sampler()
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        progress = self._build_progress(task_name, target=1)
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=1,
            progress=progress,
        )

        await strategy.produce_batch(ctx)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)

    async def test_async_produce_strategy_uses_live_consumed_progress(self):
        # 验证策略读取 live consumed 绝对进度，consumer 已取走的 group 不会被误判为缺口。
        task_name = "test_live_consumed"
        call_count = 0

        async def mock_gen(rs, **kwargs):
            nonlocal call_count
            call_count += 1
            for r in rs:
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证版本记录顺序，放宽 stale 策略避免在生产入口提前返回。
        strategy = DisaggAsyncProduceStrategyConfig(over_sample_threshold=0.0, max_staleness=3).build()
        progress = self._build_disagg_progress(
            task_name,
            target=2,
            train_step=1,
            consumed=1,
            producer_future_step=2,
            target_upto_future_step=2,
        )

        ctx = self._build_disagg_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=2,
            model_step=1,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(call_count, 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)

    async def test_async_produce_strategy_uses_fixed_batch_oversample_budget(self):
        # 验证超发预算按当前 task batch size 固定计算，而不是按剩余缺口缩小。
        task_name = "test_fixed_oversample"
        for sample_id in range(9):
            await self.replay_buffer.put([make_rollout_state(sample_id, status=Status.COMPLETED)], task_name)
        sampler = MagicMock()
        sample_ids = iter(range(100, 200))

        async def sample(task_name, group_status=None):
            self.assertEqual(group_status, [Status.ABORTED])
            return [make_rollout_state(next(sample_ids), status=Status.ABORTED)]

        sampler.sample = AsyncMock(side_effect=sample)
        mock_agent_loop = self._build_agent_loop()
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        progress = self._build_progress(task_name, target=10)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=4,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        # 当前只缺 1 个样本，但 over-sample 预算固定为 over * batch_size = 4，
        # 因此本轮最多调度到 target + 4，对应初始发射 5 个任务。
        self.assertEqual(sampler.sample.await_count, 5)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 14)

    async def test_async_produce_strategy_tail_batch_is_static_and_no_oversample(self):
        # 验证 tail-batch 模式固定从 expired/aborted pool 补必要缺口，并禁用额外超发。
        task_name = "test_tail_static"
        for sample_id in (900, 901):
            await self.replay_buffer.put([make_rollout_state(sample_id, status=Status.EXPIRED)], task_name)

        sampler = self._build_sampler()
        original_sample = sampler.sample
        sampled_statuses: list[list[Status] | None] = []

        async def instrumented_sample(task_name, group_status=None):
            sampled_statuses.append(group_status)
            return await original_sample(task_name=task_name, group_status=group_status)

        sampler.sample = instrumented_sample
        mock_agent_loop = self._build_agent_loop()
        strategy = AsyncProduceStrategyConfig(
            over_sample_threshold=1.0,
            tail_batch_trigger_size=1,
        ).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )
        await strategy.produce_batch(ctx)
        # tail-batch 模式在本轮优先走 EXPIRED pool，并且不使用 over-sample 额外发射。
        self.assertEqual(sampled_statuses, [[Status.EXPIRED, Status.ABORTED], [Status.EXPIRED, Status.ABORTED]])
        completed = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(sorted(group[0].message_uid for group in completed), [900, 901])

    async def test_async_produce_strategy_fails_fast_on_invalid_progress(self):
        # 验证 progress 缺少当前 task key 时 fail fast，避免静默用 0 掩盖调度状态损坏。
        task_name = "test_invalid_progress"
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))

        missing_target = ProduceProgress(target_samples={})
        with self.assertRaisesRegex(KeyError, "target_samples"):
            ctx = self._build_context(
                strategy,
                task_name,
                mock_agent_loop,
                sampler,
                batch_size=1,
                train_step=1,
                model_step=0,
                progress=missing_target,
            )
            await strategy.produce_batch(ctx)

        disagg_strategy = DisaggAsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        missing_consumed = DisaggProduceProgress(
            task_names=[task_name],
            producer_future_step=1,
            next_consumer_step=1,
            consumed_samples={},
            target_samples={task_name: 1},
            target_upto_future_step=1,
        )
        with self.assertRaisesRegex(KeyError, "consumed_samples"):
            ctx = self._build_disagg_context(
                disagg_strategy,
                task_name,
                mock_agent_loop,
                sampler,
                batch_size=1,
                train_step=1,
                model_step=0,
                progress=missing_consumed,
            )
            await disagg_strategy.produce_batch(ctx)

    async def test_async_produce_strategy_records_sample_version_before_staleness_refresh(self):
        # 验证新生成 token 会先记录 Rollout Model Step，再按 consumer step 刷新 staleness。
        task_name = "test_sample_version"

        async def mock_gen(rs, **kwargs):
            self.assertNotIn("model_step", kwargs)
            for r in rs:
                r.response_ids = [10, 11]
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证版本记录顺序，放宽 stale 策略避免在生产入口提前返回。
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0, max_staleness=3).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1),
        )
        await strategy.produce_batch(ctx)
        completed = await self.replay_buffer.get(1, task_name, Status.COMPLETED)
        self.assertEqual(completed[0][0].response_model_steps, [3, 3])
        self.assertEqual(completed[0][0].seq_staleness, 1)

    async def test_async_produce_strategy_preserves_partial_rollout_old_versions(self):
        # 验证 partial rollout 保留旧 token 版本，新 token 使用本次调度的 Rollout Model Step。
        task_name = "test_partial_rollout_versions"
        partial_item = make_rollout_state(700, status=Status.ABORTED)
        partial_item.response_ids = [10]
        partial_item.response_model_steps = [1]
        await self.replay_buffer.put([partial_item], task_name)

        async def mock_gen(rs, **kwargs):
            self.assertNotIn("model_step", kwargs)
            # partial rollout 的历史 token 已有版本，新 token 应按本次调度时的模型版本补齐。
            rs[0].response_ids = [10, 11, 12]
            rs[0].status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证 partial rollout 版本拼接，放宽 stale 策略保留旧分段。
        strategy = AsyncProduceStrategyConfig(
            over_sample_threshold=0.0,
            enable_partial_rollout=True,
            max_staleness=3,
        ).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1),
        )
        await strategy.produce_batch(ctx)

        completed = await self.replay_buffer.get(1, task_name, Status.COMPLETED)
        self.assertEqual(completed[0][0].response_model_steps, [1, 3, 3])
        self.assertEqual(completed[0][0].seq_staleness, 3)

    async def test_async_produce_strategy_does_not_reclaim_previous_call_pending(self):
        # 共卡 async 的 pending 只属于一次 produce_batch；下一次调用不能回收上一次遗留结果。
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.05, 2: 0.05})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertGreater(strategy.pending_task_count(), 0)

        await asyncio.sleep(0.08)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertEqual(strategy.pending_task_count(), 0)

        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 1)
        self.assertEqual(final_data[0][0].message_uid, 0)
        for group in final_data:
            self.assertIn("group_generate_time_s", group[0].extra_fields)
            self.assertGreater(group[0].extra_fields["group_generate_time_s"], 0.0)

    async def test_async_produce_strategy_pause_produce_is_explicit(self):
        # 验证显式 pause_produce 会暂停 rollout controller、drain pending，并把结果落到 replay buffer。
        task_name = "test_cleanup"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.2, 2: 0.2})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertGreater(strategy.pending_task_count(), 0)

        pause_time_s = await strategy.pause_produce(ctx)

        self.assertGreaterEqual(pause_time_s, 0.0)
        self.assertEqual(strategy.pending_task_count(), 0)
        completed = await self.replay_buffer.count(task_name, Status.COMPLETED)
        aborted = await self.replay_buffer.count(task_name, Status.ABORTED)
        expired = await self.replay_buffer.count(task_name, Status.EXPIRED)
        self.assertEqual(completed + aborted + expired, 3)
        self.assertEqual(mock_agent_loop.rollout_ctl.pause_generation.remote.await_count, 1)

    async def test_async_produce_strategy_pause_produce_collects_without_cancelling(self):
        # 验证 pending task 在 pause 等待窗口内完成时会被收集，而不是直接取消丢失结果。
        task_name = "test_cleanup_without_cancel"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.03, 2: 0.03})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertGreater(strategy.pending_task_count(), 0)

        await strategy.pause_produce(ctx)

        self.assertEqual(strategy.pending_task_count(), 0)
        completed = await self.replay_buffer.count(task_name, Status.COMPLETED)
        aborted = await self.replay_buffer.count(task_name, Status.ABORTED)
        expired = await self.replay_buffer.count(task_name, Status.EXPIRED)
        self.assertEqual(completed + aborted + expired, 3)
        self.assertEqual(mock_agent_loop.rollout_ctl.pause_generation.remote.await_count, 1)

    async def test_async_produce_strategy_returns_update_abort_without_sampling(self):
        # 验证 update_event 已设置时策略立即返回 UPDATE_WEIGHT_AND_ABORT，不再采样新 rollout。
        task_name = "test_update_abort"
        strategy = DisaggAsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        update_event = asyncio.Event()
        update_event.set()

        ctx = self._build_disagg_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=1,
            model_step=1,
            update_event=update_event,
            progress=self._build_disagg_progress(task_name, target=1, train_step=1),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 0)

    async def test_async_produce_strategy_returns_update_abort_after_schedule_pause(self):
        # 验证调度临界区中途触发 pause 后，策略停止继续调度并返回 UPDATE_WEIGHT_AND_ABORT。
        task_name = "test_update_abort_after_schedule"
        strategy = DisaggAsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        mock_agent_loop = self._build_agent_loop({0: 0.05})
        sampler = MagicMock()
        update_event = asyncio.Event()
        progress = self._build_disagg_progress(task_name, target=1)

        async def sample(task_name, group_status=None):
            # 模拟 manager 在调度临界区中途触发 pause；当前样本会进入 pending，后续应停止继续调度。
            update_event.set()
            return [make_rollout_state(0, status=Status.ABORTED)]

        sampler.sample = AsyncMock(side_effect=sample)

        ctx = self._build_disagg_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            update_event=update_event,
            model_step=0,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT)
        self.assertEqual(sampler.sample.await_count, 1)

        await strategy.pause_produce(ctx)
        self.assertEqual(strategy.pending_task_count(), 0)

    async def test_disagg_async_produce_strategy_returns_expired_batch_before_processing_leftovers(self):
        # 验证非共卡 Rollout Model Step 过期时策略先返回 EXPIRED_BATCH，不消费已有 completed leftovers。
        task_name = "test_expired_batch"
        strategy = DisaggAsyncProduceStrategyConfig(max_staleness=0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        await self.replay_buffer.put([make_rollout_state(999, status=Status.COMPLETED)], task_name)

        ctx = self._build_disagg_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=3,
            model_step=1,
            progress=self._build_disagg_progress(task_name, target=1, train_step=3),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 0)
