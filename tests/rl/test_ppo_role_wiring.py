"""PPO 角色化架构的行为测试。

覆盖的行为（一级为 TestClass，二级为 test 函数）：

- TestGAEAdvantageEstimation —— GAE 优势/回报与 session 目标语义
- TestTrainingSchedule —— pass 顺序、update 切分与 warmup 计数
- TestResidencyAndGuards —— set_residency 跨角色搬运与 role placement 守卫
- TestCriticConstruction —— critic 复用宿主 FSDP/SP 拓扑并 strict 加载
- TestRoleCheckpoint —— role checkpoint 的计数/scheduler/RNG 契约
- TestCriticSegmentTargets —— TrainBatch 上的 segment targets 安装与统计
- TestControllerPPOGuards —— token estimator / critic 构造期配对校验
- TestCheckpointOrchestration —— checkpoint 的 role-first 目录编排
- TestSessionGAEDispatch —— session GAE 按 PackPlan 恢复全局序并安装目标
- TestGAEFitSkipsGroupAdvantages —— GAE 模式不调用 group advantage 估计
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.config.optim import LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.model.base import HeadConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense0P6BConfig
from xtuner.v1.rl.advantage import (
    GAEAdvantageConfig,
    GAEEstimator,
    GRPOAdvantageConfig,
    TokenStats,
    session_token_targets,
)
from xtuner.v1.rl.advantage.gae import action_gae, terminal_rewards
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.critic_worker import CriticWorker, CriticWorkerConfig
from xtuner.v1.rl.trainer.pack import PackPlan, RankPlan, TokenBatchProto
from xtuner.v1.rl.trainer.role_worker import RoleContext, RoleWorker
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, partition_indices, pass_order
from xtuner.v1.rl.trainer.train_batch import TrainBatch
from xtuner.v1.rl.trainer.worker import TrainingWorker
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
        advantages, returns = session_token_targets(
            estimator,
            states,
            [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
        )
        self.assertEqual([row.tolist() for row in advantages], [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual([row.tolist() for row in returns], [[0.0, 0.0], [0.0, 1.0]])

    def test_normalization_does_not_rewrite_critic_returns(self):
        # 行为：advantage 归一化不改动 critic 的 returns
        estimator = GAEAdvantageConfig(gae_lambda=1.0, normalize_actor_advantage=True).build()
        states = [
            _state(rollout_id=1, group_id=0, reward={"score": 1.0}),
            _state(rollout_id=2, group_id=0, reward={"score": 3.0}),
        ]
        advantages, returns = session_token_targets(
            estimator,
            states,
            [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
        )
        self.assertEqual([row.tolist() for row in returns], [[1.0, 1.0], [3.0, 3.0]])
        flat = [value for row in advantages for value in row.tolist()]
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


class _FakeRole:
    def __init__(self, name: str, *, model: bool, optimizer: bool) -> None:
        self.role_name = name
        self._model_on_device = model
        self._optimizer_on_device = optimizer
        self.calls: list[str] = []

    @property
    def model_on_device(self) -> bool:
        return self._model_on_device

    @property
    def optimizer_on_device(self) -> bool:
        return self._optimizer_on_device

    def onload_model(self) -> None:
        self.calls.append("onload_model")
        self._model_on_device = True

    def onload_optimizer(self) -> None:
        self.calls.append("onload_optimizer")
        self._optimizer_on_device = True

    def offload_model(self) -> None:
        self.calls.append("offload_model")
        self._model_on_device = False

    def offload_optimizer(self) -> None:
        self.calls.append("offload_optimizer")
        self._optimizer_on_device = False


class TestResidencyAndGuards(unittest.TestCase):
    """worker 跨角色驻留与 role 侧 RPC 守卫。"""

    def _worker(self):
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.actor = _FakeRole("actor", model=True, optimizer=True)
        worker.critic = _FakeRole("critic", model=False, optimizer=False)
        return worker

    def test_set_residency_offloads_losers_then_loads_winners(self):
        worker = self._worker()

        TrainingWorker.set_residency(worker, "critic", "critic")

        self.assertEqual(worker.actor.calls, ["offload_optimizer", "offload_model"])
        self.assertEqual(worker.critic.calls, ["onload_model", "onload_optimizer"])
        self.assertFalse(worker.actor.model_on_device)
        self.assertTrue(worker.critic.optimizer_on_device)

        # Repeating the request re-runs the idempotent onload but never unloads the target.
        TrainingWorker.set_residency(worker, "critic", "critic")
        self.assertEqual(worker.actor.calls, ["offload_optimizer", "offload_model"])
        self.assertEqual(worker.critic.calls, ["onload_model", "onload_optimizer", "onload_model", "onload_optimizer"])

        TrainingWorker.set_residency(worker, None, None)
        self.assertEqual(worker.critic.calls[-2:], ["offload_optimizer", "offload_model"])
        self.assertFalse(worker.critic.model_on_device)

    def test_set_residency_rejects_unconfigured_role(self):
        worker = self._worker()
        worker.critic = None
        with self.assertRaisesRegex(ValueError, "not configured"):
            TrainingWorker.set_residency(worker, "critic", "critic")

    def test_role_guards_check_actual_placement(self):
        actor = ActorWorker.__new__(ActorWorker)
        actor.role_name = "actor"
        actor._model_on_device = True
        actor._optimizer_on_device = True
        RoleWorker.require_train_residency(actor)

        actor._optimizer_on_device = False
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            RoleWorker.require_train_residency(actor)
        actor._model_on_device = False
        with self.assertRaisesRegex(RuntimeError, "model"):
            RoleWorker.require_model_residency(actor)


class TestCriticConstruction(unittest.TestCase):
    """critic 构造复用宿主拓扑，不再持有 host。"""

    def test_critic_uses_host_fsdp_and_strict_load(self):
        self.assertNotIn("fsdp_cfg", CriticWorkerConfig.model_fields)
        self.assertNotIn("sp_size", CriticWorkerConfig.model_fields)

        model_cfg = Qwen3Dense0P6BConfig(head_config=HeadConfig(type="value_head"))
        critic_cfg = CriticWorkerConfig(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            load_from="/tmp/critic",
        )
        fsdp_cfg = FSDPConfig(torch_compile=False)
        ctx = RoleContext(
            data_mesh={"sp": object()},  # type: ignore[arg-type]
            device=torch.device("cpu"),
            rank=0,
            seed=0,
            logger=MagicMock(),
            log_dir=None,
        )

        with (
            patch("xtuner.v1.rl.trainer.critic_worker.TrainEngine") as train_engine,
            patch("xtuner.v1.rl.trainer.critic_worker.build_lr_scheduler") as build_scheduler,
        ):
            worker = CriticWorker(critic_cfg, ctx, fsdp_cfg)

        self.assertFalse(hasattr(worker, "host"))
        self.assertIs(train_engine.call_args.kwargs["fsdp_cfg"], fsdp_cfg)
        self.assertIs(build_scheduler.call_args.args[0], train_engine.return_value.optimizer)
        train_engine.return_value.from_hf.assert_called_once_with("/tmp/critic", strict=True)


class TestCriticSegmentTargets(unittest.TestCase):
    """critic 在 TrainBatch 上计算并安装 segment targets。"""

    @staticmethod
    def _batch(*, reward: float | None = 1.0, normalize_values: bool = True) -> TrainBatch:
        state = RolloutState(
            rollout_id=1,
            group_id=0,
            message=[],
            prompt_ids=[1],
            response_ids=[2, 3],
            input_ids=[1, 2, 3, 4],
            labels=[-100, -100, 2, 3],
            reward={"score": reward} if reward is not None else None,
            status=Status.COMPLETED,
        )
        batch = TrainBatch(
            [state],
            RankPlan(global_indices=(0,), steps=(((0,),),)),
            {
                "rollout_idx": 0,
                "use_3d_position_ids": False,
                "pack_loss_keys": [],
                "has_routed_experts": False,
            },  # type: ignore[arg-type]
            DistillationTrainerAdapter(None),
            pack_max_length=3,
            model_cfg=SimpleNamespace(n_routed_experts=0, num_hidden_layers=1, num_experts_per_tok=1),  # type: ignore[arg-type]
            teacher_loss_keys=[],
            distillation_config=None,
        )
        batch.install_old_values([torch.tensor([0.2, 0.1, 0.3])])
        return batch

    @staticmethod
    def _critic() -> CriticWorker:
        critic = CriticWorker.__new__(CriticWorker)
        critic.device = torch.device("cpu")
        return critic

    def test_installs_complete_segment_targets(self):
        batch = self._batch()
        stats = self._critic().compute_segment_targets(
            batch,
            GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
            contribute=True,
        )

        batch.require_targets("critic")
        pack = batch.materialize((0,))
        torch.testing.assert_close(pack.full_data["advantages"], torch.tensor([[0.0, 0.9, 0.7]], dtype=torch.float32))
        torch.testing.assert_close(pack.full_data["returns"], torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32))
        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.sum, 1.6, places=5)
        self.assertAlmostEqual(stats.min, 0.7, places=5)
        self.assertAlmostEqual(stats.max, 0.9, places=5)

    def test_normalizes_actor_advantage_but_not_returns(self):
        batch = self._batch()
        stats = self._critic().compute_segment_targets(
            batch,
            GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=True),
            contribute=True,
        )

        pack = batch.materialize((0,))
        for actual, expected in zip(pack.full_data["advantages"][0, 1:].tolist(), [1.0, -1.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        torch.testing.assert_close(pack.full_data["returns"], torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32))
        self.assertAlmostEqual(stats.sum, 0.0, places=5)

    def test_non_contributing_replica_installs_targets_without_stats(self):
        batch = self._batch()
        stats = self._critic().compute_segment_targets(
            batch,
            GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
            contribute=False,
        )

        self.assertEqual(stats, TokenStats.empty())
        batch.require_targets("critic")

    def test_missing_reward_score_is_rejected(self):
        batch = self._batch(reward=None)
        with self.assertRaisesRegex(ValueError, "reward score"):
            self._critic().compute_segment_targets(
                batch,
                GAEEstimator(gae_gamma=1.0, gae_lambda=1.0, normalize_actor_advantage=False),
                contribute=True,
            )


class TestControllerPPOGuards(unittest.TestCase):
    """controller 在构造时校验 token estimator / critic 配对。"""

    def test_token_estimator_requires_critic(self):
        with self.assertRaisesRegex(ValueError, "configured together"):
            TrainingController(workers=[], advantage_estimator=GAEEstimator(), has_critic=False)

    def test_critic_requires_token_estimator(self):
        with self.assertRaisesRegex(ValueError, "configured together"):
            TrainingController(
                workers=[],
                advantage_estimator=GRPOAdvantageConfig().build(),
                has_critic=True,
            )


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
    """Session GAE scatters replica-0 values back to global order and installs one
    target pair per DP rank."""

    @staticmethod
    def _controller() -> TrainingController:
        controller = TrainingController.__new__(TrainingController)
        controller.advantage_estimator = GAEEstimator(reward_scope="session", normalize_actor_advantage=False)
        controller.logger = MagicMock()
        return controller

    def _run(self, rollout_items, plan, value_batches, data_replicate_size=2):
        workers = [
            SimpleNamespace(
                critic_collect_values=SimpleNamespace(remote=MagicMock(return_value=value)),
                install_ppo_targets=SimpleNamespace(remote=MagicMock(return_value=None)),
            )
            for value in value_batches
        ]
        controller = self._controller()
        controller.workers = workers
        with (
            patch("xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value),
            patch("xtuner.v1.rl.trainer.controller.ray.put", side_effect=lambda value: value),
        ):
            stats = controller._prepare_ppo_targets({"rollout_items": rollout_items}, plan, data_replicate_size, {})
        return stats, workers

    def test_session_gae_scatters_values_by_pack_plan(self):
        rollout_items = [
            _state(uid=0, rollout_id=10, labels=[-100, 5, 6], reward={"score": 1.0}),
            _state(uid=1, rollout_id=11, labels=[7, 8], reward={"score": 0.0}),
            _state(uid=2, rollout_id=12, labels=[9, 10, 11], reward={"score": 0.5}),
            _state(uid=3, rollout_id=13, labels=[12, 13, 14, 15], reward={"score": -1.0}),
        ]
        dense_values = [[0.1, 0.2], [0.3], [0.4, 0.5], [0.6, 0.7, 0.8]]
        pack_plan = PackPlan(
            ranks=(
                RankPlan(global_indices=(2, 0), steps=(((0, 1),),)),
                RankPlan(global_indices=(3, 1), steps=(((0, 1),),)),
            ),
            metrics={},
        )
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

        stats, workers = self._run(rollout_items, pack_plan, value_batches)

        expected_advantages, expected_returns = session_token_targets(
            self._controller().advantage_estimator,
            rollout_items,
            [torch.tensor(value, dtype=torch.float32) for value in dense_values],
        )
        expected_counts = sum(torch.tensor(rollout_items[i].labels[1:]).ne(-100).sum().item() for i in range(4))
        self.assertEqual(stats.count, int(expected_counts))

        for worker_index, worker in enumerate(workers):
            rank = pack_plan.ranks[worker_index // 2]
            payload = worker.install_ppo_targets.remote.call_args.args[0]
            advantage_rows = payload[0].chunk()
            return_rows = payload[1].chunk()
            self.assertEqual(len(advantage_rows), len(rank.global_indices))
            self.assertEqual(len(return_rows), len(rank.global_indices))
            for local_index, global_index in enumerate(rank.global_indices):
                torch.testing.assert_close(advantage_rows[local_index], expected_advantages[global_index])
                torch.testing.assert_close(return_rows[local_index], expected_returns[global_index])

    def test_session_gae_rejects_value_count_mismatch(self):
        rollout_items = [_state(uid=0, rollout_id=10, labels=[-100, 5, 6], reward={"score": 1.0})]
        pack_plan = PackPlan(ranks=(RankPlan(global_indices=(0,), steps=(((0,),),)),), metrics={})
        value_batches = [TokenBatchProto.concat([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])])]

        with self.assertRaisesRegex(ValueError, "Transport rows do not match RankPlan"):
            self._run(rollout_items, pack_plan, value_batches, data_replicate_size=1)

    def test_session_gae_rejects_uncovered_samples(self):
        rollout_items = [
            _state(uid=0, rollout_id=10, labels=[-100, 5], reward={"score": 1.0}),
            _state(uid=1, rollout_id=11, labels=[6, 7], reward={"score": 1.0}),
        ]
        pack_plan = PackPlan(ranks=(RankPlan(global_indices=(0,), steps=(((0,),),)),), metrics={})
        value_batches = [TokenBatchProto.concat([torch.tensor([0.1, 0.2])])]

        with self.assertRaisesRegex(ValueError, "same length"):
            self._run(rollout_items, pack_plan, value_batches, data_replicate_size=1)


class TestGAEFitSkipsGroupAdvantages(unittest.TestCase):
    """回归：GAE 模式下 fit 不能对 token estimator 调用 group advantage compute。"""

    def test_segment_gae_fit_does_not_call_group_estimator(self):
        def _rpc(returns=None):
            return SimpleNamespace(remote=MagicMock(return_value=returns))

        worker = SimpleNamespace(
            get_worker_cfg=_rpc(
                SimpleNamespace(optimizer_steps=1, pack_strategy="legacy", pack_seed=None, pack_extra_buffer_size=0)
            ),
            get_data_replicate_size=_rpc(1),
            load_train_batch=_rpc(None),
            release_train_batch=_rpc(None),
            set_residency=_rpc(None),
            critic_collect_values=_rpc(None),
            critic_compute_segment_targets=_rpc(TokenStats.empty()),
            critic_fit=_rpc([]),
            actor_fit=_rpc({}),
        )
        state = _state()
        prepared = {
            "rollout_items": [state],
            "reward_groups": [],
            "batch_attr": {},
            "cluster_rewards": [],
            "distillation_reward_observations": [],
        }
        controller = TrainingController.__new__(TrainingController)
        controller.workers = [worker]
        controller.advantage_estimator = GAEEstimator(reward_scope="segment")
        controller.distillation = SimpleNamespace(task_adv_weight=1.0)
        controller.logger = MagicMock()
        controller._token_advantages = True
        controller._has_critic = True
        controller._prepare_group_targets = MagicMock(side_effect=AssertionError("must not run in GAE mode"))

        with (
            patch.object(controller, "_prepare_rollout_items", return_value=prepared),
            patch.object(
                controller,
                "_build_pack_plan",
                return_value=PackPlan((RankPlan((0,), (((0,),),)),), {}),
            ),
            patch.object(controller, "_prepare_ppo_targets", return_value=TokenStats.empty()),
            patch.object(controller, "_build_trainer_log_info", return_value={}),
            patch.object(controller, "_critic_log_scalars", return_value={}),
            patch("xtuner.v1.rl.trainer.controller.ray.put", side_effect=lambda value: value),
            patch("xtuner.v1.rl.trainer.controller.ray.get", side_effect=lambda value, timeout=None: value),
        ):
            controller.fit([[state]], pack_max_length=16, rollout_idx=1)

        controller._prepare_group_targets.assert_not_called()


class _DummyRole(RoleWorker):
    role_name = "actor"

    def fit(self, batch):  # pragma: no cover - checkpoint tests only
        return {"train_entropy": 0.0, "train_metrics": []}


class TestRoleTrainUpdate(unittest.TestCase):
    """RoleWorker._train_update owns update counting and scheduler stepping."""

    def test_counts_only_applied_updates(self):
        role = _DummyRole.__new__(_DummyRole)
        role.role_name = "actor"
        role.rank = 0
        role.device = torch.device("cpu")
        role.logger = MagicMock()
        role.loss_cfg = None
        role._global_train_step = 0
        role.attempted_updates = 0
        role.completed_updates = 0
        role.lr_scheduler = MagicMock()
        role.engine = SimpleNamespace(
            train_step=MagicMock(
                side_effect=lambda **kwargs: {
                    "logs_info": {},
                    "extra_info": SimpleNamespace(get=lambda: {}),
                    "total_loss": 0.0,
                }
            ),
            clip_grad_norm=MagicMock(return_value=torch.tensor(1.0)),
            step_optimizer=MagicMock(return_value=True),
        )

        log_item = RoleWorker._train_update(role, [])

        self.assertEqual(role.attempted_updates, 1)
        self.assertEqual(role.completed_updates, 1)
        role.lr_scheduler.step.assert_called_once_with()
        self.assertIn("grad_norm", log_item)

        role.engine.step_optimizer = MagicMock(return_value=False)
        RoleWorker._train_update(role, [])
        self.assertEqual(role.attempted_updates, 2)
        self.assertEqual(role.completed_updates, 1)
        self.assertEqual(role.lr_scheduler.step.call_count, 1)


class TestRoleCheckpoint(unittest.TestCase):
    """Role checkpoint owns counters, scheduler definition and per-rank RNG."""

    @staticmethod
    def _role(*, scheduler_steps: int = 10, initial_lr: float = 1e-3) -> _DummyRole:
        role = _DummyRole.__new__(_DummyRole)
        role.config = SimpleNamespace(
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            scheduler_steps=scheduler_steps,
        )
        role.rank = 0
        role.seed = 0
        role.logger = MagicMock()
        role.attempted_updates = 0
        role.completed_updates = 0
        role._global_train_step = 0
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=initial_lr)
        role.engine = SimpleNamespace(
            optimizer=optimizer,
            save_dcp=MagicMock(),
            load_dcp=MagicMock(),
        )
        role.lr_scheduler = build_lr_scheduler(optimizer, role.config.lr_cfg, scheduler_steps)
        return role

    def test_save_resume_restores_counts_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actor"
            role = self._role()
            role.attempted_updates = 3
            role.completed_updates = 2
            role.lr_scheduler.step()
            expected_lr = role.lr_scheduler.get_last_lr()[0]
            torch.manual_seed(1234)
            expected_rng = torch.get_rng_state().clone()

            role.save(path)
            (path / ".metadata").touch()

            self.assertTrue((path / "role_state.pt").is_file())
            self.assertTrue((path / "rng_state_rank0.pt").is_file())

            torch.manual_seed(9999)
            resumed = self._role(initial_lr=5e-4)
            resumed.resume(path, LoadCheckpointConfig())

            self.assertEqual(resumed.attempted_updates, 3)
            self.assertEqual(resumed.completed_updates, 2)
            self.assertEqual(resumed.engine.optimizer.param_groups[0]["lr"], expected_lr)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_resume_without_scheduler_keeps_new_initial_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actor"
            role = self._role()
            role.lr_scheduler.step()
            role.save(path)
            (path / ".metadata").touch()

            resumed = self._role(initial_lr=5e-4)
            resumed.resume(path, LoadCheckpointConfig(load_scheduler=False))

            self.assertEqual(resumed.engine.optimizer.param_groups[0]["lr"], 5e-4)

    def test_resume_rejects_changed_scheduler_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "actor"
            role = self._role(scheduler_steps=10)
            role.save(path)
            (path / ".metadata").touch()

            resumed = self._role(scheduler_steps=20)
            with self.assertRaisesRegex(ValueError, "Scheduler configuration changed"):
                resumed.resume(path, LoadCheckpointConfig())
