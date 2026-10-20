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
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, partition_indices, pass_order
from xtuner.v1.rl.trainer.worker import TrainingWorker, TrainSample
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


class TestPPORoleWiring(unittest.TestCase):
    def test_gae_places_outcome_reward_on_the_last_supervised_token(self):
        estimator = GAEAdvantageConfig(
            gae_gamma=1.0,
            gae_lambda=1.0,
            normalize_actor_advantage=False,
        ).build()
        self.assertIsInstance(estimator, GAEEstimator)
        advantages, returns = estimator.compute_gae(
            [_state(labels=[0, -100, 1, 1], reward={"score": 1.0})],
            [[0.0, 0.0, 0.0]],
        )
        self.assertEqual(advantages, [[0.0, 1.0, 1.0]])
        self.assertEqual(returns, [[0.0, 1.0, 1.0]])

    def test_gae_uses_one_discount_for_advantage_and_return(self):
        estimator = GAEAdvantageConfig(gae_lambda=0.0, normalize_actor_advantage=False).build()
        advantages, returns = estimator.compute_gae(
            [_state(labels=[0, 1, 2], reward={"score": 1.0})],
            [[0.0, 0.0]],
        )
        self.assertEqual(advantages, [[0.0, 1.0]])
        self.assertEqual(returns, [[0.0, 1.0]])

    def test_gae_resets_at_trajectory_boundaries_and_skips_observations(self):
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
        advantages, returns = estimator.compute_gae(
            states,
            [[0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(advantages, [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(returns, [[0.0, 0.0], [0.0, 1.0]])

    def test_normalization_does_not_rewrite_critic_returns(self):
        estimator = GAEAdvantageConfig(gae_lambda=1.0, normalize_actor_advantage=True).build()
        states = [
            _state(rollout_id=1, group_id=0, reward={"score": 1.0}),
            _state(rollout_id=2, group_id=0, reward={"score": 3.0}),
        ]
        advantages, returns = estimator.compute_gae(
            states,
            [[0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(returns, [[1.0, 1.0], [3.0, 3.0]])
        flat = [value for sample in advantages for value in sample]
        self.assertAlmostEqual(sum(flat) / len(flat), 0.0, places=5)

    def test_critic_pass_order_shuffles_after_the_first_pass(self):
        self.assertEqual(pass_order(4, rollout_idx=1, pass_index=0, seed=0), [0, 1, 2, 3])
        shuffled = pass_order(4, rollout_idx=1, pass_index=1, seed=0)
        self.assertEqual(sorted(shuffled), [0, 1, 2, 3])
        self.assertNotEqual(shuffled, [0, 1, 2, 3])
        self.assertEqual(partition_indices([0, 1, 2, 3, 4], 2), [[0, 1, 2], [3, 4]])

    def test_absolute_warmup_counts_optimizer_updates(self):
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

    def test_attach_and_switch_route_to_host_workers(self):
        class Worker:
            def __init__(self):
                self.attached = {}
                self.role = "actor"

            def attach(self, name, worker_cls, worker_cfg):
                self.attached[name] = (worker_cls, worker_cfg)

            def switch_role_modules(self, role):
                self.role = role

        workers = [Worker(), Worker()]
        controller = TrainingController(workers=workers)
        controller.attach("critic", object, {"head": "value"})

        self.assertEqual(controller._roles, {"critic"})
        self.assertEqual(workers[0].attached["critic"][1], {"head": "value"})
        controller.switch_role_modules("critic")
        self.assertEqual([worker.role for worker in workers], ["critic", "critic"])
        controller.switch_role_modules("actor")
        self.assertEqual(workers[1].role, "actor")

    def test_critic_reuses_host_fsdp_and_sp_topology(self):
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
        host = SimpleNamespace(config=SimpleNamespace(fsdp_cfg=fsdp_cfg), sp_mesh=object())

        with (
            patch("xtuner.v1.rl.trainer.critic_worker.TrainEngine") as train_engine,
            patch("xtuner.v1.rl.trainer.critic_worker.build_lr_scheduler") as build_scheduler,
        ):
            worker = CriticWorker(critic_cfg, host=host)

        self.assertIs(worker.host, host)
        self.assertIs(worker.sp_mesh, host.sp_mesh)
        self.assertIs(train_engine.call_args.kwargs["fsdp_cfg"], fsdp_cfg)
        self.assertIs(build_scheduler.call_args.args[0], train_engine.return_value.optimizer)
        train_engine.return_value.from_hf.assert_called_once_with("/tmp/critic", strict=False)
        self.assertFalse(critic_cfg.model_cfg.compile_cfg)

    def test_critic_requires_host_worker(self):
        model_cfg = Qwen3Dense0P6BConfig(head_type="value_head")
        critic_cfg = CriticWorkerConfig(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            load_from="/tmp/critic",
        )
        with self.assertRaisesRegex(ValueError, "must be attached"):
            CriticWorker(critic_cfg, host=None)

    def test_segment_gae_scope_is_detected(self):
        controller = TrainingController(
            workers=[],
            advantage_estimator=GAEEstimator(reward_scope="segment"),
        )
        self.assertEqual(controller._gae_reward_scope, "segment")

        session_controller = TrainingController(
            workers=[],
            advantage_estimator=GAEEstimator(reward_scope="session"),
        )
        self.assertEqual(session_controller._gae_reward_scope, "session")

    def test_non_gae_estimator_has_no_reward_scope(self):
        controller = TrainingController(workers=[], advantage_estimator=None)
        self.assertIsNone(controller._gae_reward_scope)

    def test_compute_segment_gae_uses_cached_values(self):
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
            gae_gamma=1.0,
            gae_lambda=1.0,
            normalize_actor_advantage=False,
            contribute_to_global_stats=True,
        )

        self.assertEqual(sample.advantages.shape, values.shape)
        self.assertEqual(sample.returns.shape, values.shape)
        self.assertEqual(sample.advantages[0, 0].item(), 0.0)
        self.assertEqual(stats["count"], 2.0)
        self.assertEqual(sample.advantages[:, 1:].sum().item(), stats["sum"])

    def test_compute_segment_gae_non_contributing_replica_returns_no_stats(self):
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
            gae_gamma=1.0,
            gae_lambda=1.0,
            normalize_actor_advantage=False,
            contribute_to_global_stats=False,
        )

        self.assertEqual(stats, {})
        self.assertIsNotNone(sample.advantages)
        self.assertIsNotNone(sample.returns)

    def test_compute_segment_gae_rejects_missing_reward_score(self):
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
                gae_gamma=1.0,
                gae_lambda=1.0,
                normalize_actor_advantage=False,
                contribute_to_global_stats=True,
            )

    def test_save_hf_includes_attached_critic(self):
        worker = TrainingWorker.__new__(TrainingWorker)
        worker._engine = MagicMock()
        critic_worker = MagicMock()
        teacher_worker = MagicMock()
        worker._attached = {"critic": critic_worker, "teacher": teacher_worker}
        worker._sft_dataloader = None
        worker.logger = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ckpt-step-1"
            checkpoint_path.mkdir()
            actor_path = checkpoint_path / "actor"
            actor_path.mkdir()
            (actor_path / ".metadata").write_text("")
            (checkpoint_path / "critic").mkdir()
            (checkpoint_path / "teacher").mkdir()
            TrainingWorker.save(worker, checkpoint_path, no_save_optimizer=True)
            self.assertEqual(
                worker._engine.save_dcp.call_args.kwargs["weights_dir"],
                checkpoint_path / "actor",
            )
            critic_worker.save.assert_called_once_with(checkpoint_path / "critic", no_save_optimizer=True)
            teacher_worker.save.assert_not_called()

            TrainingWorker.resume(
                worker,
                LoadCheckpointConfig(checkpoint_path=checkpoint_path, load_optimizer_states=False),
            )
            critic_worker.resume.assert_called_once()
            teacher_worker.resume.assert_not_called()

            TrainingWorker.save_hf(worker, checkpoint_path / "hf", save_dtype=torch.float16)

        self.assertEqual(
            worker._engine.save_hf.call_args.args,
            (checkpoint_path / "hf" / "actor", torch.float16),
        )
        critic_worker.save_hf.assert_called_once_with(checkpoint_path / "hf" / "critic", torch.float16)
        teacher_worker.save_hf.assert_not_called()

        critic = CriticWorker.__new__(CriticWorker)
        critic._engine = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            critic_path = Path(tmpdir) / "critic"
            critic_path.mkdir()
            critic.save(critic_path, no_save_optimizer=True)
            critic_path.joinpath(".metadata").write_text("")
            critic.resume(critic_path)

        critic._engine.save_dcp.assert_called_once_with(
            weights_dir=critic_path,
            save_optimizer=False,
        )
        critic._engine.load_dcp.assert_called_once_with(
            weights_dir=critic_path,
            load_states=True,
            load_args=True,
        )
        critic.save_hf(Path("/tmp/critic-hf"), save_dtype=torch.float16)
        critic._engine.save_hf.assert_called_once_with(
            hf_dir=Path("/tmp/critic-hf"),
            save_dtype=torch.float16,
        )

    def test_actor_only_switch_rejects_unknown_role(self):
        controller = TrainingController(workers=[])
        with self.assertRaises(KeyError):
            controller.switch_role_modules("critic")

    def test_gae_without_critic_is_rejected(self):
        controller = TrainingController(workers=[], advantage_estimator=GAEEstimator())
        with self.assertRaises(RuntimeError):
            controller.fit([[]], pack_max_length=8, rollout_idx=0)
