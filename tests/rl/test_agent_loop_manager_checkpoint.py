"""AgentLoopManager checkpoint save/resume 的 public 行为测试。

Good Tests:
- 通过 AgentLoopManagerConfig.build() 构造真实 AgentLoopManager。
- 使用真实 Sampler、Dataloader、ReplayBuffer、ProduceStrategy 和 checkpoint 文件验证行为。
- 只用 fake rollout controller 替代真实推理服务，避免启动耗时外部 worker。

Bad Tests:
- 不直接读写 _status、_model_step、_produce_progress 等 manager 私有状态。
- 不 mock Sampler / ReplayBuffer / ProduceStrategy 内部模块。
- 不重复测试 ProduceProgress、_PendingTasks、ReplayBuffer 存储后端的细节。

本文件主要覆盖的 public 行为:
- 共卡 produce_batch 下，SyncProduceStrategy / AsyncProduceStrategy 都会在 resume 后
  继续同一段 sampler 序列。
- 非共卡 produce_loop/get_batch 下，AsyncProduceStrategy 也会在 resume 后
  继续同一段 sampler 序列。
- save/resume 后，checkpoint 中尚未消费的 completed rollout group 仍可通过 get_batch 取出。
- save 时如果 AsyncProduceStrategy 仍有 pending rollout task，会 fail fast，避免保存不完整状态。
- resume 后的 AsyncProduceStrategy 后台 producer 必须等 trainer 显式 continue_produce 后才恢复。
"""

import asyncio
import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig


QWEN3_4B_PATH = os.environ.get("QWEN3_4B_PATH")


class _RemoteMethod:
    def __init__(self, func=None):
        self.func = func
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _run():
            if self.func is None:
                return None
            result = self.func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return _run()


class _SyncRemoteMethod:
    def __init__(self, func=None):
        self.func = func
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.func is None:
            return None
        return self.func(*args, **kwargs)


class _FakeRolloutController:
    def __init__(self):
        self.generate = _RemoteMethod(self._generate)
        self.pause_generation = _RemoteMethod()
        self.continue_generation = _RemoteMethod()
        self.set_enable_partial_rollout = _SyncRemoteMethod()

    def _generate(self, rollout_state: RolloutState) -> RolloutState:
        rollout_state.status = Status.COMPLETED
        rollout_state.response = "ok"
        rollout_state.response_ids = [100, 101]
        rollout_state.reward = {"score": 1.0 if int(rollout_state.uid or 0) % 2 == 0 else 0.5}
        return rollout_state


class _BlockingRolloutController(_FakeRolloutController):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.generate = _RemoteMethod(self._blocking_generate)

    async def _blocking_generate(self, rollout_state: RolloutState) -> RolloutState:
        self.started.set()
        await self.release.wait()
        return self._generate(rollout_state)


class TestAgentLoopManagerCheckpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "rollout_data.jsonl"
        self._write_dataset(self.dataset_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_dataset(self, dataset_path: Path):
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "data_source": "unit",
                    "prompt": [{"role": "user", "content": f"question {idx}?"}],
                    "reward_model": {"style": "rule", "ground_truth": "ok"},
                    "extra_info": {"index": idx},
                }
            )
        dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def _build_manager(self, replay_buffer_config, *, rollout_controller=None, produce_strategy_config=None):
        assert QWEN3_4B_PATH is not None
        rollout_controller = rollout_controller or _FakeRolloutController()
        produce_strategy_config = produce_strategy_config or SyncProduceStrategyConfig()
        dataloader_cfg = DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(
                        name="unit",
                        anno_path=self.dataset_path,
                        enable_sequential_sampler=True,
                        disable_filter=True,
                    ),
                    "tokenize_fn": RLTextTokenizeFnConfig(max_length=128),
                }
            ],
            collator="fake_collator",
            pack_level="none",
            pack_to_max_length=False,
            pack_max_length=256,
            num_workers=0,
            round_up=False,
        )
        manager_cfg = AgentLoopManagerConfig(
            tasks=[
                {
                    "task_name": "unit_task",
                    "agent_loop_config": SingleTurnAgentLoopConfig(
                        hf_checkpoint=QWEN3_4B_PATH,
                        sample_params=SampleParams(max_tokens=2, temperature=0.0, top_k=1),
                    ),
                    "produce_strategy_config": produce_strategy_config,
                    "sampler_config": SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=2),
                }
            ]
        )
        return manager_cfg.build(
            rollout_controller=rollout_controller,
            tokenizer=QWEN3_4B_PATH,
            replay_buffer=replay_buffer_config.build(),
        )

    def _build_async_manager(self, *, rollout_controller=None):
        with patch("xtuner.v1.rl.agent_loop_manager.producer.ray.get", side_effect=lambda ref, *_, **__: ref):
            return self._build_manager(
                AsyncReplayBufferConfig(),
                rollout_controller=rollout_controller,
                produce_strategy_config=AsyncProduceStrategyConfig(over_sample_threshold=0.0),
            )

    def _build_sync_produce_batch_manager(self):
        return self._build_manager(SyncReplayBufferConfig())

    def _build_async_produce_batch_manager(self):
        return self._build_async_manager()

    def _rollout_index(self, rollout_group: list[RolloutState]) -> int:
        return int(rollout_group[0].extra_fields["index"])

    async def _produce_batch_index(self, manager, *, train_step: int, model_step: int) -> int:
        result = await manager.produce_batch(batch_size=1, train_step=train_step, model_step=model_step)
        return self._rollout_index(result.rollout_states[0])

    async def _consume_async_index(self, manager, *, train_step: int) -> int:
        result = await asyncio.wait_for(manager.get_batch(batch_size=1, train_step=train_step), timeout=3.0)
        self.assertEqual(len(result.rollout_states), 1)
        return self._rollout_index(result.rollout_states[0])

    async def _continue_and_consume_async_index(self, manager, *, train_step: int, model_step: int) -> int:
        await manager.continue_produce(model_step=model_step)
        return await self._consume_async_index(manager, train_step=train_step)

    async def _assert_produce_batch_resume_keeps_sampler_suffix(self, build_manager):
        manager = build_manager()
        sample1 = await manager.produce_batch(batch_size=1, train_step=1, model_step=0)
        sample1_index = self._rollout_index(sample1.rollout_states[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ckpt"
            await manager.save(checkpoint_path, model_step=1)

            expected_suffix = [
                await self._produce_batch_index(manager, train_step=2, model_step=1),
                await self._produce_batch_index(manager, train_step=3, model_step=2),
            ]

            restored_manager = build_manager()
            restored_model_step = await restored_manager.resume(checkpoint_path)
            actual_suffix = [
                await self._produce_batch_index(restored_manager, train_step=2, model_step=1),
                await self._produce_batch_index(restored_manager, train_step=3, model_step=2),
            ]

        self.assertEqual(restored_model_step, 1)
        self.assertEqual(len({sample1_index, *expected_suffix}), 3)
        self.assertEqual(actual_suffix, expected_suffix)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for AgentLoopManager checkpoint tests")
    async def test_produce_batch_resume_continues_same_sampler_suffix_for_sync_and_async_strategies(self):
        # 验证共卡 produce_batch:
        # 同一份 checkpoint 对 Sync/Async 生产策略都恢复到同一段后缀样本。
        cases = [
            ("sync", self._build_sync_produce_batch_manager),
            ("async", self._build_async_produce_batch_manager),
        ]

        for strategy_name, build_manager in cases:
            with self.subTest(strategy=strategy_name):
                await self._assert_produce_batch_resume_keeps_sampler_suffix(build_manager)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for AgentLoopManager checkpoint tests")
    async def test_async_produce_loop_resume_continues_same_sampler_suffix_after_checkpoint(self):
        # 验证非共卡 AsyncProduceStrategy: sample1 后保存 checkpoint，正常继续生产 sample2/sample3。
        # 从 checkpoint resume 后也必须继续生产同一段 sample2/sample3。
        manager = self._build_async_manager()
        produce_task = asyncio.create_task(manager.produce_loop(batch_size=1))

        try:
            sample1_index = await self._consume_async_index(manager, train_step=1)
            await manager.pause_produce(use_global_progress=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_path = Path(tmp_dir) / "ckpt"
                await manager.save(checkpoint_path, model_step=1)

                expected_suffix = [
                    await self._continue_and_consume_async_index(manager, train_step=2, model_step=1),
                    await self._continue_and_consume_async_index(manager, train_step=3, model_step=2),
                ]

                restored_manager = self._build_async_manager()
                restored_model_step = await restored_manager.resume(checkpoint_path)
                restored_produce_task = asyncio.create_task(restored_manager.produce_loop(batch_size=1))
                try:
                    actual_suffix = [
                        await self._continue_and_consume_async_index(
                            restored_manager, train_step=2, model_step=restored_model_step
                        ),
                        await self._continue_and_consume_async_index(
                            restored_manager, train_step=3, model_step=restored_model_step + 1
                        ),
                    ]
                finally:
                    restored_manager.shutdown()
                    await asyncio.wait_for(restored_produce_task, timeout=3.0)
        finally:
            manager.shutdown()
            await asyncio.wait_for(produce_task, timeout=3.0)

        self.assertEqual(restored_model_step, 1)
        self.assertEqual(len({sample1_index, *expected_suffix}), 3)
        self.assertEqual(actual_suffix, expected_suffix)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for AgentLoopManager checkpoint tests")
    async def test_resume_keeps_unconsumed_completed_groups_available_to_get_batch(self):
        # 验证 save 时 replay buffer 中未消费的 completed group，resume 后仍可被 get_batch 消费。
        manager = self._build_manager(AsyncReplayBufferConfig())
        buffered_group = [
            RolloutState(
                uid=9000 + idx,
                message_uid=90,
                message=[{"role": "user", "content": "buffered"}],
                prompt_ids=[1, 2, 3],
                response="ok",
                response_ids=[100 + idx],
                reward={"score": 1.0},
                status=Status.COMPLETED,
                extra_fields={"index": 90},
            )
            for idx in range(2)
        ]
        await manager.replay_buffer.put(buffered_group, "unit_task", model_step=4, current_train_step=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ckpt"
            await manager.save(checkpoint_path, model_step=4)

            restored_manager = self._build_manager(AsyncReplayBufferConfig())
            restored_model_step = await restored_manager.resume(checkpoint_path)
            result = await restored_manager.get_batch(batch_size=1, train_step=5)

        self.assertEqual(restored_model_step, 4)
        self.assertEqual(len(result.rollout_states), 1)
        self.assertEqual(self._rollout_index(result.rollout_states[0]), 90)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for AgentLoopManager checkpoint tests")
    async def test_save_rejects_while_async_rollout_task_is_pending(self):
        # 验证后台异步 rollout 还在进行时不能保存。
        # 否则 checkpoint 会丢失未入库的生产结果。
        rollout_controller = _BlockingRolloutController()
        with patch("xtuner.v1.rl.agent_loop_manager.producer.ray.get", side_effect=lambda ref, *_, **__: ref):
            manager = self._build_manager(
                AsyncReplayBufferConfig(),
                rollout_controller=rollout_controller,
                produce_strategy_config=AsyncProduceStrategyConfig(over_sample_threshold=0.0),
            )
        produce_task = asyncio.create_task(manager.produce_loop(batch_size=1))

        try:
            await asyncio.wait_for(rollout_controller.started.wait(), timeout=2.0)
            with tempfile.TemporaryDirectory() as tmp_dir:
                with self.assertRaisesRegex(RuntimeError, "pending rollout tasks"):
                    await manager.save(Path(tmp_dir) / "ckpt", model_step=0)
        finally:
            manager.shutdown()
            rollout_controller.release.set()
            await asyncio.wait_for(produce_task, timeout=2.0)

    @unittest.skipUnless(QWEN3_4B_PATH, "QWEN3_4B_PATH is required for AgentLoopManager checkpoint tests")
    async def test_resume_requires_continue_before_async_producer_generates(self):
        # 验证 resume 后 producer 仍处在等待同步完成状态。
        # 只有 continue_produce 后才恢复异步生产。
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "ckpt"
            rollout_controller = _FakeRolloutController()
            with patch("xtuner.v1.rl.agent_loop_manager.producer.ray.get", side_effect=lambda ref, *_, **__: ref):
                manager = self._build_manager(
                    AsyncReplayBufferConfig(),
                    rollout_controller=rollout_controller,
                    produce_strategy_config=AsyncProduceStrategyConfig(over_sample_threshold=0.0),
                )
            await manager.save(checkpoint_path, model_step=1)

            restored_rollout_controller = _FakeRolloutController()
            with patch("xtuner.v1.rl.agent_loop_manager.producer.ray.get", side_effect=lambda ref, *_, **__: ref):
                restored_manager = self._build_manager(
                    AsyncReplayBufferConfig(),
                    rollout_controller=restored_rollout_controller,
                    produce_strategy_config=AsyncProduceStrategyConfig(over_sample_threshold=0.0),
            )
            restored_model_step = await restored_manager.resume(checkpoint_path)
            produce_task = asyncio.create_task(restored_manager.produce_loop(batch_size=1))

            try:
                await asyncio.sleep(0.05)
                self.assertEqual(restored_rollout_controller.generate.calls, [])

                await restored_manager.continue_produce(model_step=restored_model_step)
                result = await asyncio.wait_for(restored_manager.get_batch(batch_size=1, train_step=2), timeout=3.0)
            finally:
                restored_manager.shutdown()
                await asyncio.wait_for(produce_task, timeout=2.0)

        self.assertEqual(len(result.rollout_states), 1)
        self.assertEqual(len(restored_rollout_controller.continue_generation.calls), 1)
        self.assertGreaterEqual(len(restored_rollout_controller.generate.calls), 1)


if __name__ == "__main__":
    unittest.main()
