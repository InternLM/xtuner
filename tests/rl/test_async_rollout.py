from __future__ import annotations

import os
import unittest

import ray
import torch

from transformers import AutoTokenizer

from xtuner.v1.data_proto import SampleParams, Status
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.rl.agent_loop import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    SingleTurnAgentLoopConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, AutoAcceleratorWorkers

MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH", "")
DATA_PATH = os.environ.get("ROLLOUT_DATA_PATH", "")
MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = 512
PACK_MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
EXPERIMENTAL_NAME = "async_rl_integration_test"

_RESOURCE_MAP = {"npu": "NPU", "cuda": "GPU"}


def _accelerator_type() -> str:
    return _RESOURCE_MAP[torch.accelerator.current_accelerator().type]


def _build_rollout_controller():
    """Build a RolloutController backed by a real inference engine."""
    resources_cfg = AcceleratorResourcesConfig(
        accelerator=_accelerator_type(),
        num_workers=1,
        num_cpus_per_worker=8,
        cpu_memory_per_worker=16 * 1024**3,
    )
    rollout_config = RolloutConfig(
        env=EXPERIMENTAL_NAME,
        device=resources_cfg.accelerator,
        model_path=MODEL_PATH,
        gpu_memory_utilization=0.8,
        context_length=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
        rollout_max_batch_size_per_instance=16,
        max_retry_per_sample=0,
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)
    rollout_ctl = ray.remote(RolloutController).remote(rollout_config, pg)
    return rollout_ctl


def _build_agent_loop_manager(
    rollout_ctl,
    task_name: str,
    over_sample_threshold: float = 0.0,
    enable_partial_rollout: bool = False,
    max_staleness: int = 0,
    tail_batch_trigger_size: int = 0,
    prompt_repeat_k: int = 1,
    max_tokens: int = MAX_RESPONSE_LENGTH,
):
    """Build an AgentLoopManager backed by a fresh AsyncReplayBuffer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    dataset_cfg = DatasetConfig(name=EXPERIMENTAL_NAME, anno_path=DATA_PATH)
    tokenizer_fn_cfg = RLTextTokenizeFnConfig(max_length=MAX_PROMPT_LENGTH)
    dataloader_cfg = DataloaderConfig(
        dataset_config_list=[{"dataset": dataset_cfg, "tokenize_fn": tokenizer_fn_cfg}],
        pack_max_length=PACK_MAX_LENGTH,
        collator="fake_collator",
        pack_level="none",
    )
    sampler_config = SamplerConfig(
        dataloader_cfg=dataloader_cfg,
        prompt_repeat_k=prompt_repeat_k,
    )

    sample_params = SampleParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        return_token_ids=True,
    )
    agent_loop_config = SingleTurnAgentLoopConfig(
        hf_checkpoint=MODEL_PATH,
        sample_params=sample_params,
    )

    produce_strategy_config = AsyncProduceStrategyConfig(
        over_sample_threshold=over_sample_threshold,
        enable_partial_rollout=enable_partial_rollout,
        max_staleness=max_staleness,
        tail_batch_trigger_size=tail_batch_trigger_size,
    )

    replay_buffer = AsyncReplayBufferConfig().build()

    manager_cfg = AgentLoopManagerConfig(
        tasks=[
            TaskSpecConfig(
                task_name=task_name,
                agent_loop_config=agent_loop_config,
                produce_strategy_config=produce_strategy_config,
                sampler_config=sampler_config,
            )
        ],
    )
    manager = manager_cfg.build(
        rollout_controller=rollout_ctl,
        tokenizer=tokenizer,
        replay_buffer=replay_buffer,
        logger=None,
    )
    return manager

class TestOversampling(unittest.IsolatedAsyncioTestCase):
    """Oversampling tests (mirrors debug_rollout=True: rollout only, no training).

    Why ABORTED samples are guaranteed:
      - over_sample_threshold=2.0 => data_concurrency = 3 * batch_size = 6 tasks
      - max_tokens=512             => long responses; most tasks still in-flight
                                      when the first batch_size completions arrive
      - _cleanup_pending_tasks()   => remaining tasks get abort-signalled and
                                      stored as ABORTED in the replay buffer
    """

    OVER_SAMPLE_THRESHOLD = 2.0        # data_concurrency = 3 * batch_size
    BATCH_SIZE = 2
    INITIAL_DATA_CONCURRENCY = int((1 + OVER_SAMPLE_THRESHOLD) * BATCH_SIZE)  # = 6

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("XTUNER_USE_FA3", "1")
        os.environ.setdefault("LMD_SKIP_WARMUP", "1")

    def setUp(self):
        ray.init(num_cpus=32, ignore_reinit_error=True)
        self.rollout_ctl = _build_rollout_controller()

    def tearDown(self):
        ray.shutdown()

    async def test_1_1_total_count_after_first_rollout(self):
        """1.1: After produce_batch round 1:

            remain_completed + remain_aborted == INITIAL_DATA_CONCURRENCY

        Flow:
          1. strategy starts INITIAL_DATA_CONCURRENCY tasks concurrently.
          2. As soon as BATCH_SIZE completions are collected, the while-loop
             exits; remaining pending tasks go through _cleanup_pending_tasks
             and are stored as ABORTED.
          3. produce_batch() then calls replay_buffer.get(BATCH_SIZE, COMPLETED)
             which consumes exactly BATCH_SIZE items.
          4. Any extras that completed during the abort window remain as
             COMPLETED in the buffer.

        Therefore:
            remain_completed + remain_aborted == INITIAL_DATA_CONCURRENCY - BATCH_SIZE

        Because every task either ends up COMPLETED or ABORTED in the buffer,
        and exactly BATCH_SIZE items are consumed by replay_buffer.get().
        """
        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name="test_1_1",
            over_sample_threshold=self.OVER_SAMPLE_THRESHOLD,
        )
        replay_buffer = manager.replay_buffer

        await manager.produce_batch(batch_size=self.BATCH_SIZE, train_step=1)

        remain_completed = await replay_buffer.count(
            task_name="test_1_1", group_status=Status.COMPLETED
        )
        remain_aborted = await replay_buffer.count(
            task_name="test_1_1", group_status=Status.ABORTED
        )

        # Primary assertion: items remaining in buffer after produce_batch consumes
        # BATCH_SIZE completed samples == INITIAL_DATA_CONCURRENCY - BATCH_SIZE
        expected_remaining = self.INITIAL_DATA_CONCURRENCY - self.BATCH_SIZE
        self.assertEqual(
            remain_completed + remain_aborted,
            expected_remaining,
            msg=(
                f"remain_completed={remain_completed}, remain_aborted={remain_aborted}, "
                f"expected total={expected_remaining} "
                f"(= INITIAL_DATA_CONCURRENCY {self.INITIAL_DATA_CONCURRENCY} "
                f"- BATCH_SIZE {self.BATCH_SIZE})"
            ),
        )

    async def test_1_2_second_rollout_does_not_convert_completed_leftovers(self):
        """1.2: Round 2 no longer destructively converts COMPLETED leftovers.

        AsyncProduceStrategy v2.2 keeps completed leftovers in the fresh window.
        Only existing ABORTED samples may be re-sampled through the ABORTED pool;
        completed samples are either consumed as completed or refreshed/expired by
        the manager consumer entry.
        """
        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name="test_1_2",
            over_sample_threshold=self.OVER_SAMPLE_THRESHOLD,
        )
        replay_buffer = manager.replay_buffer
        original_sample = manager.data_sampler.sample

        sampled_from_aborted = 0

        async def instrumented_sample(task_name, group_status=None, **kwargs):
            nonlocal sampled_from_aborted
            result = await original_sample(
                task_name=task_name, group_status=group_status, **kwargs
            )
            # Items fetched from the ABORTED pool still carry status==ABORTED.
            if result and result[0].status == Status.ABORTED:
                sampled_from_aborted += 1
            return result

        manager.data_sampler.sample = instrumented_sample

        # --- Round 1 ---
        await manager.produce_batch(batch_size=self.BATCH_SIZE, train_step=1)

        # After round 1: produce_batch consumed BATCH_SIZE completed items.
        # The leftover items (completed but not consumed) stay in the buffer.
        round1_remain_completed = await replay_buffer.count(
            task_name="test_1_2", group_status=Status.COMPLETED
        )
        round1_remain_aborted = await replay_buffer.count(
            task_name="test_1_2", group_status=Status.ABORTED
        )
        # Total leftover == INITIAL_DATA_CONCURRENCY - BATCH_SIZE
        expected_leftover = self.INITIAL_DATA_CONCURRENCY - self.BATCH_SIZE
        self.assertEqual(
            round1_remain_completed + round1_remain_aborted,
            expected_leftover,
            msg=(
                f"Round 1 leftover: completed={round1_remain_completed}, "
                f"aborted={round1_remain_aborted}, expected total={expected_leftover}"
            ),
        )
        # --- Round 2: reset counter then run ---
        sampled_from_aborted = 0
        await manager.produce_batch(batch_size=self.BATCH_SIZE, train_step=2)

        self.assertLessEqual(
            sampled_from_aborted,
            round1_remain_aborted,
            msg=(
                "Round 2 should not convert round-1 COMPLETED leftovers into the ABORTED queue: "
                f"sampled_from_aborted={sampled_from_aborted}, round1_remain_aborted={round1_remain_aborted}"
            ),
        )


class TestPartialRollout(unittest.IsolatedAsyncioTestCase):
    """Partial-rollout tests.

    All tests inject pre-constructed ABORTED samples directly into the
    replay buffer so that Sampler.sample(group_status=[ABORTED]) picks them
    up without any mocking.  The real AgentLoopManager.produce_batch() is
    used throughout.

    Key configuration:
      - over_sample_threshold=2.0  → data_concurrency = 3; guarantees concurrent
                                     tasks so the genuine oversampling + partial-
                                     rollout path is exercised (not just injected
                                     into a single-task environment).
      - enable_partial_rollout=True → ABORTED samples resume from existing
                                     response_ids instead of starting over.
    """

    BATCH_SIZE = 1
    OVER_SAMPLE = 2.0  # data_concurrency = int((1+2.0)*1) = 3; genuine oversampling
    # Short max_tokens for the max-exhausted short-circuit test; medium for multi-round.
    MAX_TOKENS_SHORT = 8
    MAX_TOKENS_MULTI = 32

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("XTUNER_USE_FA3", "1")
        os.environ.setdefault("LMD_SKIP_WARMUP", "1")

    def setUp(self):
        ray.init(num_cpus=32, ignore_reinit_error=True)
        self.rollout_ctl = _build_rollout_controller()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    def tearDown(self):
        ray.shutdown()

    def _make_aborted_state(self, uid: int, prompt: str, response_ids: list[int],
                            response_model_steps: list[int] | None = None,
                            max_tokens: int = MAX_RESPONSE_LENGTH) -> "RolloutState":
        """Helper: build an ABORTED RolloutState with given response_ids."""
        from xtuner.v1.data_proto import RolloutState, SampleParams, Status
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        state = RolloutState(
            uid=uid,
            message=[{"role": "user", "content": prompt}],
            prompt_ids=prompt_ids,
            sample_params=SampleParams(
                max_tokens=max_tokens,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                return_token_ids=True,
            ),
            status=Status.ABORTED,
            response_ids=response_ids,
            response="placeholder",
            logprobs=[0.0] * len(response_ids),
            response_mask=[1] * len(response_ids),
            response_model_steps=response_model_steps if response_model_steps is not None else [0] * len(response_ids),
            seq_staleness=0,
            extra_fields={},
        )
        return state

    async def test_2_1_partial_rollout_response_ids_are_concatenated(self):
        """2.1: Partial rollout 的 response_ids 前缀必须保持不变。

        Setup:
          - over_sample_threshold=2.0 → 3 个并发任务；注入的 ABORTED 样本与另外
            2 个 dataloader 新样本同时运行，真实触发 oversampling + partial-rollout 路径。
          - 注入 uid=9001 的 ABORTED 样本，response_ids=[1000,1001,1002,1003]。
          - 由于多任务竞争，注入样本可能多次被 abort 并在后续轮次继续；每次
            preprocess 以 existing response_ids 为前缀，postprocess 拼接新内容。

        断言: 最终完成的 uid=9001 样本的 response_ids 以初始 4 token 为前缀，
              且长度 > 4（确实生成了新内容）。
        """
        from xtuner.v1.data_proto import Status
        task_name = "test_2_1"
        initial_response_ids = [1000, 1001, 1002, 1003]
        injected_uid = 9001

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        replay_buffer = manager.replay_buffer

        state = self._make_aborted_state(
            uid=injected_uid,
            prompt="Count from one.",
            response_ids=initial_response_ids,
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        await replay_buffer.put([state], task_name)

        # Loop: with oversampling the injected sample may be aborted multiple times
        # before completing.  Search by uid across rounds.
        target_sample = None
        for train_step in range(1, 15):
            completed_groups = await manager.produce_batch(
                batch_size=self.BATCH_SIZE, train_step=train_step
            )
            for group in completed_groups.rollout_states:
                for sample in group:
                    if sample.uid == injected_uid:
                        target_sample = sample
            if target_sample is not None:
                break

        self.assertIsNotNone(
            target_sample,
            msg=f"Injected sample (uid={injected_uid}) never completed within 14 rounds",
        )
        final_response_ids = target_sample.response_ids
        self.assertGreater(
            len(final_response_ids), len(initial_response_ids),
            msg="Partial rollout should have appended new tokens",
        )
        self.assertEqual(
            final_response_ids[: len(initial_response_ids)],
            initial_response_ids,
            msg="response_ids must start with the original injected prefix",
        )

    async def test_2_2_eos_in_response_skips_inference_engine(self):
        """2.2: ABORTED 样本末尾为 EOS token → worker 短路，response_ids 不变。

        EOS 短路不调用推理引擎，注入样本几乎瞬间完成，在 3 个并发任务中
        必然最先完成，因此 completed_groups[0][0] 就是注入样本。

        断言: 返回样本的 response_ids 与注入时完全相同。
        """
        from xtuner.v1.data_proto import Status
        from xtuner.v1.rl.rollout.worker import get_eos_token

        task_name = "test_2_2"
        eos = get_eos_token(MODEL_PATH)
        eos_id = eos[0] if isinstance(eos, list) else eos
        initial_response_ids = [1000, 1001, eos_id]
        injected_uid = 9002

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        replay_buffer = manager.replay_buffer

        state = self._make_aborted_state(
            uid=injected_uid,
            prompt="Say hello.",
            response_ids=initial_response_ids,
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        await replay_buffer.put([state], task_name)

        # EOS short-circuit completes with no LLM call → always wins the race.
        completed_groups = await manager.produce_batch(
            batch_size=self.BATCH_SIZE, train_step=1
        )
        completed_groups = completed_groups.rollout_states

        self.assertEqual(len(completed_groups), self.BATCH_SIZE)
        final = completed_groups[0][0]
        self.assertEqual(final.uid, injected_uid,
                         msg="EOS short-circuit sample should be the first to complete")
        self.assertEqual(final.status, Status.COMPLETED)
        self.assertEqual(
            final.response_ids,
            initial_response_ids,
            msg="EOS short-circuit: response_ids must be identical to the injected ones",
        )

    async def test_2_3_max_tokens_exhausted_skips_inference_engine(self):
        """2.3: len(response_ids)==max_tokens → remaining_tokens==0 → worker 短路，response_ids 不变。

        与 test_2_2 同理，短路不调用推理引擎，注入样本在 3 个并发任务中必然最先完成。

        断言: 返回样本的 response_ids 与注入时完全相同。
        """
        from xtuner.v1.data_proto import Status
        task_name = "test_2_3"
        max_tokens = self.MAX_TOKENS_SHORT
        initial_response_ids = list(range(1010, 1010 + max_tokens))  # len == max_tokens
        injected_uid = 9003

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_tokens=max_tokens,
        )
        replay_buffer = manager.replay_buffer

        state = self._make_aborted_state(
            uid=injected_uid,
            prompt="Say hello.",
            response_ids=initial_response_ids,
            max_tokens=max_tokens,
        )
        await replay_buffer.put([state], task_name)

        # max_tokens short-circuit completes with no LLM call → always wins the race.
        completed_groups = await manager.produce_batch(
            batch_size=self.BATCH_SIZE, train_step=1
        )
        completed_groups = completed_groups.rollout_states

        self.assertEqual(len(completed_groups), self.BATCH_SIZE)
        final = completed_groups[0][0]
        self.assertEqual(final.uid, injected_uid,
                         msg="max_tokens short-circuit sample should be the first to complete")
        self.assertEqual(final.status, Status.COMPLETED)
        self.assertEqual(
            final.response_ids,
            initial_response_ids,
            msg="max_tokens exhausted: response_ids must be identical to the injected ones",
        )

    async def test_2_4_multi_round_response_ids_never_exceed_max_tokens(self):
        """2.4: 多轮 partial rollout 后 len(response_ids) <= max_tokens。

        over_sample_threshold=2.0 → 每轮 3 个并发任务；注入样本可能经历多次
        abort + continue 才能完成。无论经历几轮，最终 response_ids 长度不超过 max_tokens。

        按 uid 搜索目标样本，最多跑 14 轮。
        """
        from xtuner.v1.data_proto import Status
        task_name = "test_2_4"
        max_tokens = self.MAX_TOKENS_MULTI
        injected_uid = 9004

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_tokens=max_tokens,
        )
        replay_buffer = manager.replay_buffer

        state = self._make_aborted_state(
            uid=injected_uid,
            prompt="Count from one.",
            response_ids=[1020, 1021],  # 2 tokens initially; max_tokens=32
            max_tokens=max_tokens,
        )
        await replay_buffer.put([state], task_name)

        target_sample = None
        for train_step in range(1, 15):
            completed_groups = await manager.produce_batch(
                batch_size=self.BATCH_SIZE, train_step=train_step
            )
            for group in completed_groups.rollout_states    :
                for sample in group:
                    if sample.uid == injected_uid:
                        self.assertLessEqual(
                            len(sample.response_ids),
                            max_tokens,
                            msg=(
                                f"Step {train_step}: accumulated response_ids length "
                                f"{len(sample.response_ids)} exceeds max_tokens {max_tokens}"
                            ),
                        )
                        target_sample = sample
            if target_sample is not None:
                break

        self.assertIsNotNone(
            target_sample,
            msg=f"Injected sample (uid={injected_uid}) never completed within 14 rounds",
        )
        self.assertLessEqual(
            len(target_sample.response_ids),
            max_tokens,
            msg=f"Final response_ids length {len(target_sample.response_ids)} > max_tokens {max_tokens}",
        )


class TestTailBatch(unittest.IsolatedAsyncioTestCase):
    BATCH_SIZE = 2
    # 真实 lmdeploy 后端在大量并发 abort 时容易触发 session cleanup 异常；
    # 这里沿用 oversampling 覆盖里已稳定验证的并发规模，仍足够产生 leftover。
    OVER_SAMPLE = 2.0  # data_concurrency = (1 + 2.0) * BATCH_SIZE = 6

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("XTUNER_USE_FA3", "1")
        os.environ.setdefault("LMD_SKIP_WARMUP", "1")

    def setUp(self):
        ray.init(num_cpus=32, ignore_reinit_error=True)
        self.rollout_ctl = _build_rollout_controller()

    def tearDown(self):
        ray.shutdown()

    async def test_3_1_max_staleness_0_marks_expired(self):
        """3.1a: max_staleness=0 — 需要 3 轮才能在 buffer 中观察到 EXPIRED。

        staleness 积累路径（enable_partial_rollout=True）：
          Round 1 (step=1): 6 个并发任务，2 个完成后其余被 abort。
            被 abort 的样本携带 step=1 生成的分段 response，response_model_steps=[1,...].
          Round 2 (step=2): round1 的 ABORTED 样本被续写，多数在 round2 内完成（COMPLETED）。
            postprocess 只拼接 partial rollout 历史；producer put 前记录 response_model_steps 并刷新 staleness。
            该轮未消费的 COMPLETED 样本会留在 buffer 中。
          Round 3 (step=3): produce_batch 作为消费入口，先刷新 buffer 中的
            COMPLETED 样本，检查 seq_staleness=1 >= stale_threshold=1 → 标为 EXPIRED，
            放回 buffer。由于 trigger_size=0，EXPIRED 样本不在本轮被消费。

        断言: round3 结束后 buffer 中 expired > 0。
        """
        from xtuner.v1.data_proto import Status

        MAX_STALENESS = 0
        task_name = "test_3_1a"

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_staleness=MAX_STALENESS,
            tail_batch_trigger_size=0,  # 只测 EXPIRED 标记，不触发 tail-batch 模式
            # 测试用 rollout context_length 是 MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH；
            # max_tokens 不能超过这个测试配置，否则 lmdeploy 会进入超长请求的异常/abort 路径。
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        replay_buffer = manager.replay_buffer

        # 3 轮是让 staleness 自然积累并被 produce_batch 入口刷新标记的最少轮数：
        #   round1 产生 ABORTED（step=1 tokens）→ round2 续写完成并留作 COMPLETED
        #   → round3 开头刷新 completed 并标 EXPIRED（staleness=1 >= 1）
        for train_step in range(1, 5):
            await manager.produce_batch(batch_size=self.BATCH_SIZE, train_step=train_step)

        expired_count = await replay_buffer.count(
            task_name=task_name, group_status=Status.EXPIRED
        )
        aborted_count = await replay_buffer.count(
            task_name=task_name, group_status=Status.ABORTED
        )

        self.assertGreater(
            expired_count, 0,
            msg=(
                f"max_staleness=0: after 3 rounds (steps 1→3), leftover COMPLETED samples "
                f"with seq_staleness=1 should be marked EXPIRED by produce_batch entry refresh. "
                f"expired={expired_count}, aborted={aborted_count}"
            ),
        )

    async def test_3_2_tail_batch_mode_resets_staleness_to_zero(self):
        """3.2: 真实多轮循环自然触发 tail-batch 模式，验证 seq_staleness 重置为 0。

        配置:
          over_sample_threshold=2.0    → 每轮产生大量遗留样本
          max_staleness=0              → stale_threshold=1，一步即触发 EXPIRED
          tail_batch_trigger_size = BATCH_SIZE // 2 = 1 → expired >= 1 即进入 tail-batch

        流程 (最多 10 轮):
          - 在调用 produce_batch 之前读取 expired_before。
          - 若 expired_before >= trigger_size，本轮由 strategy 进入 tail-batch 模式：
              从 EXPIRED 池取样 → preprocess 重置 response_ids=[], response_model_steps=[]
              → 全新生成 → postprocess 拼接历史
              → producer put 前记录 response_model_steps 并刷新 staleness 为 0。
          - 取到第一个 tail-batch 完成样本后退出循环。

        断言:
          1. tail-batch 模式在 10 轮内被触发。
          2. 该轮返回的 COMPLETED 样本 seq_staleness == 0。
        """
        from xtuner.v1.data_proto import Status

        MAX_STALENESS = 0
        TRIGGER_SIZE = self.BATCH_SIZE // 2  # = 1
        task_name = "test_3_2"

        manager = _build_agent_loop_manager(
            self.rollout_ctl,
            task_name=task_name,
            over_sample_threshold=self.OVER_SAMPLE,
            enable_partial_rollout=True,
            max_staleness=MAX_STALENESS,
            tail_batch_trigger_size=TRIGGER_SIZE,
            # 保持在测试用 context_length 内，避免尾批测试被超长生成请求干扰。
            max_tokens=MAX_RESPONSE_LENGTH,
        )
        replay_buffer = manager.replay_buffer

        tail_batch_triggered = False
        completed_from_tail_batch = None

        for train_step in range(1, 11):
            expired_before = await replay_buffer.count(
                task_name=task_name, group_status=Status.EXPIRED
            )

            completed_groups = await manager.produce_batch(
                batch_size=self.BATCH_SIZE, train_step=train_step
            )
            completed_groups = completed_groups.rollout_states

            # 进入本轮前 expired >= trigger_size → 本轮就是 tail-batch 轮
            if expired_before >= TRIGGER_SIZE:
                tail_batch_triggered = True
                if completed_groups:
                    completed_from_tail_batch = completed_groups[0][0]
                    break

        self.assertTrue(
            tail_batch_triggered,
            msg="Tail-batch mode was never triggered within 10 rollout rounds.",
        )
        self.assertIsNotNone(
            completed_from_tail_batch,
            msg="Tail-batch round produced no completed samples.",
        )
        self.assertEqual(
            completed_from_tail_batch.seq_staleness, 0,
            msg=(
                f"Tail-batch sample must have seq_staleness=0 (fresh generation), "
                f"got seq_staleness={completed_from_tail_batch.seq_staleness}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
