"""PPO 角色化架构的行为测试。

覆盖的行为（一级为 TestClass，二级为 test 函数）：

- TestGAEAdvantageEstimation —— GAE 优势/回报估计的数值语义
  - test_gae_places_outcome_reward_on_the_last_supervised_token
  - test_gae_uses_one_discount_for_advantage_and_return
  - test_gae_resets_at_trajectory_boundaries_and_skips_observations
  - test_session_scope_places_one_reward_on_the_last_segment
  - test_normalization_does_not_rewrite_critic_returns
- TestTrainingSchedule —— 训练调度（多 pass 顺序 / update 切分 / warmup 计数）
  - test_critic_pass_order_shuffles_after_the_first_pass
  - test_absolute_warmup_counts_optimizer_updates
- TestRoleSwitching —— worker 侧角色切换与 RPC 守卫
  - test_switch_role_offloads_others_and_onloads_target
  - test_requires_role_rejects_unconfigured_and_inactive
- TestCriticConstruction —— critic 构造对宿主拓扑的复用
  - test_critic_reuses_host_fsdp_and_sp_topology
- TestWorkerSegmentGAE —— worker 本地 segment GAE 对缓存 values 的消费
  - test_compute_segment_gae_uses_cached_values
  - test_compute_segment_gae_non_contributing_replica_returns_no_stats
  - test_compute_segment_gae_rejects_missing_reward_score
- TestControllerPPOGuards —— controller 对 PPO 前置条件的守卫
  - test_gae_without_critic_is_rejected
- TestSessionGAEDispatch —— session GAE 跨 rank values 的散射对齐
  - test_session_gae_scatters_values_by_pack_plan
  - test_session_gae_rejects_value_count_mismatch
  - test_session_gae_rejects_uncovered_samples
- TestCheckpointOrchestration —— checkpoint 保存/恢复的角色编排
  - test_checkpoint_cover_every_configured_role
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.config.optim import LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.model.dense.qwen3 import Qwen3Dense0P6BConfig
from xtuner.v1.rl.advantage import GAEAdvantageConfig, GAEEstimator
from xtuner.v1.rl.advantage.gae import action_gae, terminal_rewards
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.critic_worker import CriticWorker, CriticWorkerConfig
from xtuner.v1.rl.trainer.pack import TokenBatchProto
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, partition_indices, pass_order
from xtuner.v1.rl.trainer.worker import TrainingWorker, TrainSample, requires_role
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _state(**overrides) -> RolloutState:
    fields = {
        "message": [],
        "labels": [0, 1, 2],
        "reward": {"score": 1.0},
        "group_id": 0,
        "rollout_id": 1,
        "finish_reason": "stop",
    }
    fields.update(overrides)
    return RolloutState(**fields)


class TestGAEAdvantageEstimation(unittest.TestCase):
    """GAE 优势/回报估计的数值语义（public: GAEEstimator.compute_segment_gae / compute_session_gae）。"""

    def test_gae_places_outcome_reward_on_the_last_supervised_token(self):
        # 行为：结局 reward 只落在最后一个被监督的 token 上
        estimator = GAEAdvantageConfig(
            gae_gamma=1.0,
            gae_lambda=1.0,
            normalize_actor_advantage=False,
        ).build()
        self.assertIsInstance(estimator, GAEEstimator)
        advantages, returns = estimator.compute_segment_gae(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0]),
            torch.tensor([False, True, True]),
            torch.tensor([0, 3]),
        )
        self.assertEqual(advantages.tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(returns.tolist(), [0.0, 1.0, 1.0])

    def test_gae_uses_one_discount_for_advantage_and_return(self):
        # 行为：advantage 与 return 使用同一套折扣参数
        estimator = GAEAdvantageConfig(gae_lambda=0.0, normalize_actor_advantage=False).build()
        advantages, returns = estimator.compute_segment_gae(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0]),
            torch.tensor([True, True]),
            torch.tensor([0, 2]),
        )
        self.assertEqual(advantages.tolist(), [0.0, 1.0])
        self.assertEqual(returns.tolist(), [0.0, 1.0])

    def test_gae_resets_at_trajectory_boundaries_and_skips_observations(self):
        # 行为：GAE 在轨迹边界重置且跳过观测位
        values = torch.tensor([0.0, 10.0])
        mask = torch.tensor([True, True])
        rewards = terminal_rewards(torch.tensor([1.0, 0.0]), mask, torch.tensor([0, 1, 2]))
        self.assertEqual(rewards.tolist(), [1.0, 0.0])
        advantages = action_gae(values, rewards, mask, torch.tensor([0, 1, 2]), gamma=1.0, gae_lambda=0.0)
        # One trajectory would bootstrap index 0 from the next value and produce 11.
        self.assertEqual(advantages.tolist(), [1.0, -10.0])

        observed = action_gae(
            torch.tensor([0.0, 99.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([True, False, True]),
            torch.tensor([0, 3]),
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertEqual(observed.tolist(), [1.0, 0.0, 1.0])

    def test_session_scope_places_one_reward_on_the_last_segment(self):
        # 行为：session 作用域下 reward 落在会话最后一个分段末尾
        estimator = GAEAdvantageConfig(
            gae_lambda=0.0,
            reward_scope="session",
            normalize_actor_advantage=False,
        ).build()
        states = [
            _state(
                rollout_id=1,
                session_id=7,
                labels=[0, 1, 2],
                extra_fields={"agent_trace_segment_index": 0},
            ),
            _state(
                rollout_id=2,
                session_id=7,
                labels=[0, 3, 4],
                extra_fields={"agent_trace_segment_index": 1},
            ),
        ]
        advantages, returns = estimator.compute_session_gae(
            states,
            [[0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(advantages, [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(returns, [[0.0, 0.0], [0.0, 1.0]])

    def test_normalization_does_not_rewrite_critic_returns(self):
        # 行为：advantage 归一化不改动 critic 的 returns
        estimator = GAEAdvantageConfig(gae_lambda=1.0, normalize_actor_advantage=True).build()
        states = [
            _state(rollout_id=1, group_id=0, reward={"score": 1.0}),
            _state(rollout_id=2, group_id=0, reward={"score": 3.0}),
        ]
        advantages, returns = estimator.compute_session_gae(
            states,
            [[0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(returns, [[1.0, 1.0], [3.0, 3.0]])
        flat = [value for sample in advantages for value in sample]
        self.assertAlmostEqual(sum(flat) / len(flat), 0.0, places=5)


class TestTrainingSchedule(unittest.TestCase):
    """训练调度行为（public: pass_order / partition_indices / build_lr_scheduler）。"""

    def test_critic_pass_order_shuffles_after_the_first_pass(self):
        # 行为：多 pass 复训首 pass 保序、后续 pass 洗牌；update 切分均匀
        self.assertEqual(pass_order(4, rollout_idx=1, pass_index=0, seed=0), [0, 1, 2, 3])
        shuffled = pass_order(4, rollout_idx=1, pass_index=1, seed=0)
        self.assertEqual(sorted(shuffled), [0, 1, 2, 3])
        self.assertNotEqual(shuffled, [0, 1, 2, 3])
        self.assertEqual(partition_indices([0, 1, 2, 3, 4], 2), [[0, 1, 2], [3, 4]])

    def test_absolute_warmup_counts_optimizer_updates(self):
        # 行为：warmup 按 optimizer 更新次数而非 rollout 步数计数
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = build_lr_scheduler(
            optimizer,
            LRConfig(lr_type="constant", warmup_ratio=2, lr_min=1e-6),
            total_steps=4,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5e-6)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5e-6)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)


class TestRoleSwitching(unittest.TestCase):
    """worker 侧角色切换与 RPC 守卫（public: switch_role / requires_role）。"""

    def test_switch_role_offloads_others_and_onloads_target(self):
        # 行为：切换角色时其余角色下卡、目标角色上卡，重复切换幂等重执行
        worker = TrainingWorker.__new__(TrainingWorker)
        actor, critic = MagicMock(), MagicMock()
        actor.role_name, critic.role_name = "actor", "critic"
        worker.actor, worker.critic = actor, critic
        worker._current_role = "actor"

        TrainingWorker.switch_role(worker, "critic")
        actor.offload.assert_called_once()
        actor.onload.assert_not_called()
        critic.onload.assert_called_once()
        critic.offload.assert_not_called()
        self.assertEqual(worker._current_role, "critic")

        # Declarative: re-switching to the same role re-runs ensure-state
        # (onload called again); the role methods decide what actually moves.
        TrainingWorker.switch_role(worker, "critic")
        self.assertEqual(critic.onload.call_count, 2)
        self.assertEqual(critic.offload.call_count, 0)
        self.assertEqual(worker._current_role, "critic")

        TrainingWorker.switch_role(worker, None)
        critic.offload.assert_called_once()
        actor.onload.assert_not_called()
        self.assertIsNone(worker._current_role)

        # 切换到未注册角色立即报 KeyError，且不改变当前记账。
        worker.critic = None
        with self.assertRaisesRegex(KeyError, "not registered"):
            TrainingWorker.switch_role(worker, "critic")
        self.assertIsNone(worker._current_role)

    def test_requires_role_rejects_unconfigured_and_inactive(self):
        # 行为：RPC 守卫区分『角色未配置』与『角色未激活』两种拒绝
        worker = TrainingWorker.__new__(TrainingWorker)
        actor = MagicMock()
        actor.role_name = "actor"
        worker.actor, worker.critic = actor, None
        worker._current_role = "actor"

        @requires_role("critic")
        def critic_rpc(self_):
            return "ok"

        # critic not configured on this worker
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            critic_rpc(worker)

        @requires_role("actor")
        def actor_rpc(self_):
            return "ok"

        # configured but not the active role
        worker._current_role = None
        with self.assertRaisesRegex(RuntimeError, "requires active role"):
            actor_rpc(worker)
        worker._current_role = "actor"
        self.assertEqual(actor_rpc(worker), "ok")


class TestCriticConstruction(unittest.TestCase):
    """critic 构造复用宿主拓扑（TrainEngine/build_lr_scheduler 为进程外重依赖，mock 之）。"""

    def test_critic_reuses_host_fsdp_and_sp_topology(self):
        # 行为：critic 构造复用宿主的 FSDP/SP 拓扑而非自带配置
        self.assertNotIn("fsdp_cfg", CriticWorkerConfig.model_fields)
        self.assertNotIn("sp_size", CriticWorkerConfig.model_fields)

        model_cfg = Qwen3Dense0P6BConfig(head_type="value_head")
        critic_cfg = CriticWorkerConfig(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            load_from="/tmp/critic",
        )
        fsdp_cfg = FSDPConfig(torch_compile=False)
        host = SimpleNamespace(config=SimpleNamespace(fsdp_cfg=fsdp_cfg), sp_mesh=object(), logger=MagicMock(), rank=0)

        with (
            patch("xtuner.v1.rl.trainer.critic_worker.TrainEngine") as train_engine,
            patch("xtuner.v1.rl.trainer.critic_worker.build_lr_scheduler") as build_scheduler,
        ):
            worker = CriticWorker(critic_cfg, host=host)

        self.assertIs(worker.host, host)
        self.assertIs(train_engine.call_args.kwargs["fsdp_cfg"], fsdp_cfg)
        self.assertIs(build_scheduler.call_args.args[0], train_engine.return_value.optimizer)
        train_engine.return_value.from_hf.assert_called_once_with("/tmp/critic", strict=False)
        # torch.compile gating lives in the host init (TrainingWorker.__init__),
        # not in role construction.


class TestWorkerSegmentGAE(unittest.TestCase):
    """worker 本地 segment GAE 的行为（public: CriticWorker.compute_segment_gae）。"""

    def test_compute_segment_gae_uses_cached_values(self):
        # 行为：segment GAE 消费缓存的 critic values 并产出 advantages/returns/统计
        labels = torch.tensor([[[-100, 2, 3]]]).reshape(1, 3)
        values = torch.tensor([[[0.2, 0.1, 0.3]]]).reshape(1, 3)
        sample = TrainSample(
            seq_ctx=MagicMock(),
            shifted_labels=labels,
            rollout_logprobs=None,
            teacher_targets={},
            reward_score=1.0,
            values=values,
        )
        host = SimpleNamespace(get_cached_batch=lambda: ([sample], [], {}))
        critic = CriticWorker.__new__(CriticWorker)
        critic.host = host

        stats = critic.compute_segment_gae(
            estimator=GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
            contribute_to_global_stats=True,
        )

        self.assertEqual(sample.advantages.shape, values.shape)
        self.assertEqual(sample.returns.shape, values.shape)
        self.assertEqual(sample.advantages[0, 0].item(), 0.0)
        self.assertEqual(stats["count"], 2.0)
        self.assertEqual(sample.advantages[:, 1:].sum().item(), stats["sum"])

    def test_compute_segment_gae_normalizes_with_global_moments(self):
        # 行为：归一化数学收敛在 estimator.normalize_with_moments，worker 只累矩
        labels = torch.tensor([[[-100, 2, 3]]]).reshape(1, 3)
        values = torch.tensor([[[0.2, 0.1, 0.3]]]).reshape(1, 3)
        sample = TrainSample(
            seq_ctx=MagicMock(),
            shifted_labels=labels,
            rollout_logprobs=None,
            teacher_targets={},
            reward_score=1.0,
            values=values,
        )
        host = SimpleNamespace(get_cached_batch=lambda: ([sample], [], {}))
        critic = CriticWorker.__new__(CriticWorker)
        critic.host = host

        stats = critic.compute_segment_gae(
            estimator=GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=True),
            contribute_to_global_stats=True,
        )

        # gamma=lambda=1: kept advantages 为 [0.9, 0.7]（reward 落在最后一个 token）
        for actual, expected in zip(sample.advantages[0].tolist(), [0.0, 1.0, -1.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertAlmostEqual(stats["sum"], 0.0, places=5)

    def test_compute_segment_gae_non_contributing_replica_returns_no_stats(self):
        # 行为：非贡献副本本地照常产出 advantage 但不返回全局统计
        sample = TrainSample(
            seq_ctx=MagicMock(),
            shifted_labels=torch.tensor([[[-100, 2]]]).reshape(1, 2),
            rollout_logprobs=None,
            teacher_targets={},
            reward_score=1.0,
            values=torch.tensor([[[0.2, 0.1]]]).reshape(1, 2),
        )
        host = SimpleNamespace(get_cached_batch=lambda: ([sample], [], {}))
        critic = CriticWorker.__new__(CriticWorker)
        critic.host = host

        stats = critic.compute_segment_gae(
            estimator=GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
            contribute_to_global_stats=False,
        )

        self.assertEqual(stats, {})
        self.assertIsNotNone(sample.advantages)
        self.assertIsNotNone(sample.returns)

    def test_compute_segment_gae_rejects_missing_reward_score(self):
        # 行为：缺少 reward score 的样本被明确拒绝
        sample = TrainSample(
            seq_ctx=MagicMock(),
            shifted_labels=torch.tensor([[[-100, 2]]]).reshape(1, 2),
            rollout_logprobs=None,
            teacher_targets={},
            values=torch.tensor([[[0.2, 0.1]]]).reshape(1, 2),
        )
        host = SimpleNamespace(get_cached_batch=lambda: ([sample], [], {}))
        critic = CriticWorker.__new__(CriticWorker)
        critic.host = host

        with self.assertRaisesRegex(ValueError, "requires a reward score"):
            critic.compute_segment_gae(
                estimator=GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
                contribute_to_global_stats=True,
            )


class TestControllerPPOGuards(unittest.TestCase):
    """controller 对 PPO 前置条件的守卫（public: TrainingController.fit）。"""

    def test_gae_without_critic_is_rejected(self):
        # 行为：GAE 估计器在没有 critic 角色时 fit 直接拒绝
        controller = TrainingController(workers=[], advantage_estimator=GAEEstimator())
        with self.assertRaises(RuntimeError):
            controller.fit([[]], pack_max_length=8, rollout_idx=0)


class TestCheckpointOrchestration(unittest.TestCase):
    """checkpoint 保存/恢复的角色编排（public: save/resume/save_hf）。"""

    def test_checkpoint_cover_every_configured_role(self):
        # 行为：save/resume/save_hf 以 role-first 目录覆盖每个已配置角色及 SFT 状态
        worker = TrainingWorker.__new__(TrainingWorker)
        actor, critic = MagicMock(), MagicMock()
        actor.role_name, critic.role_name = "actor", "critic"
        worker.actor, worker.critic = actor, critic
        worker.logger = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ckpt-step-1"
            checkpoint_path.mkdir()
            TrainingWorker.save(worker, checkpoint_path, no_save_optimizer=True)
            actor.save.assert_called_once_with(checkpoint_path / "actor", True)
            critic.save.assert_called_once_with(checkpoint_path / "critic", True)
            # SFT state is actor-owned and written at the checkpoint root.
            actor.save_sft_state.assert_called_once_with(checkpoint_path)

            cfg = LoadCheckpointConfig(checkpoint_path=checkpoint_path, load_optimizer_states=False)
            TrainingWorker.resume(worker, cfg)
            actor.resume.assert_called_once_with(checkpoint_path / "actor", cfg)
            critic.resume.assert_called_once_with(checkpoint_path / "critic", cfg)
            actor.resume_sft_state.assert_called_once_with(checkpoint_path)

            TrainingWorker.save_hf(worker, str(checkpoint_path / "hf"), save_dtype=torch.float16)
        actor.save_hf.assert_called_once_with(checkpoint_path / "hf" / "actor", torch.float16)
        critic.save_hf.assert_called_once_with(checkpoint_path / "hf" / "critic", torch.float16)


class TestSessionGAEDispatch(unittest.TestCase):
    """Session GAE must scatter per-rank critic values back to global sample order
    before running the estimator, reading only replica-0 batches."""

    @staticmethod
    def _controller() -> TrainingController:
        controller = TrainingController.__new__(TrainingController)
        controller.advantage_estimator = GAEEstimator(reward_scope="session", normalize_actor_advantage=False)
        controller._gae_reward_scope = "session"
        return controller

    def test_session_gae_scatters_values_by_pack_plan(self):
        # 行为：session GAE 按 pack plan 把各 rank values 散射回全局样本序且只读 replica-0
        # Four samples across two DP ranks with scrambled global indices; replica
        # positions 1 and 3 hold decoy values that must never be read.
        rollout_items = [
            _state(uid=0, rollout_id=10, labels=[-100, 5, 6], reward={"score": 1.0}),
            _state(uid=1, rollout_id=11, labels=[7, 8], reward={"score": 0.0}),
            _state(uid=2, rollout_id=12, labels=[9, 10, 11], reward={"score": 0.5}),
            _state(uid=3, rollout_id=13, labels=[12, 13, 14, 15], reward={"score": -1.0}),
        ]
        dense_values = [[0.1, 0.2], [0.3], [0.4, 0.5], [0.6, 0.7, 0.8]]
        pack_plan = {
            "dp_plans": {
                0: {"global_indices": [2, 0], "plan": [[[0, 1]]]},
                1: {"global_indices": [3, 1], "plan": [[[0, 1]]]},
            },
            "plan_log": {},
        }
        decoy = TokenBatchProto.concat([torch.tensor([9.9, 9.9])])
        value_batches = [
            TokenBatchProto.concat(
                [
                    torch.tensor(dense_values[2], dtype=torch.float32),
                    torch.tensor(dense_values[0], dtype=torch.float32),
                ]
            ),
            decoy,
            TokenBatchProto.concat(
                [
                    torch.tensor(dense_values[3], dtype=torch.float32),
                    torch.tensor(dense_values[1], dtype=torch.float32),
                ]
            ),
            decoy,
        ]

        sample_means, token_advantages, token_returns, advantage_stats = self._controller()._estimate_advantages(
            {"reward_groups": []}, rollout_items, pack_plan, value_batches, data_replicate_size=2
        )

        # Token rows keep each sample's shifted-label length.
        self.assertEqual([len(row) for row in token_advantages], [2, 1, 2, 3])
        self.assertEqual([len(row) for row in token_returns], [2, 1, 2, 3])
        # The scattered mapping matches running the estimator directly on dense values.
        expected_advantages, expected_returns = self._controller().advantage_estimator.compute_session_gae(
            rollout_items, dense_values
        )
        self.assertEqual(token_advantages, expected_advantages)
        self.assertEqual(token_returns, expected_returns)
        # Sample means average the supervised positions (sample 0 has one masked row).
        for state, mean, row in zip(rollout_items, sample_means, token_advantages):
            mask = [label != -100 for label in state.labels[1:]]
            supervised = [value for value, keep in zip(row, mask) if keep]
            self.assertAlmostEqual(mean, sum(supervised) / len(supervised))
        # Session-GAE stats are derived downstream, not here.
        self.assertIsNone(advantage_stats)

    def test_session_gae_rejects_value_count_mismatch(self):
        # 行为：values 行数与样本数不匹配时拒绝
        rollout_items = [_state(uid=0, rollout_id=10, labels=[-100, 5, 6], reward={"score": 1.0})]
        pack_plan = {"dp_plans": {0: {"global_indices": [0], "plan": [[[0]]]}}, "plan_log": {}}
        # Rank 0 reports two local value rows for one packed sample.
        value_batches = [
            TokenBatchProto.concat([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])]),
        ]
        with self.assertRaisesRegex(ValueError, "do not match the packed samples"):
            self._controller()._estimate_advantages(
                {"reward_groups": []}, rollout_items, pack_plan, value_batches, data_replicate_size=1
            )

    def test_session_gae_rejects_uncovered_samples(self):
        # 行为：plan 未覆盖全部样本时拒绝
        # Sample 1 is claimed by no DP rank, so its dense value slot stays empty.
        rollout_items = [
            _state(uid=0, rollout_id=10, labels=[-100, 5], reward={"score": 1.0}),
            _state(uid=1, rollout_id=11, labels=[6, 7], reward={"score": 1.0}),
        ]
        pack_plan = {"dp_plans": {0: {"global_indices": [0], "plan": [[[0]]]}}, "plan_log": {}}
        value_batches = [TokenBatchProto.concat([torch.tensor([0.1, 0.2])])]
        with self.assertRaisesRegex(RuntimeError, "do not cover every training sample"):
            self._controller()._estimate_advantages(
                {"reward_groups": []}, rollout_items, pack_plan, value_batches, data_replicate_size=1
            )
