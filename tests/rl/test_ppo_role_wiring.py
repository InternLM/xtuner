import unittest

import torch

from xtuner.v1.config.optim import LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.advantage import GAEAdvantageConfig, GAEEstimator
from xtuner.v1.rl.advantage.gae import action_gae, terminal_rewards
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, partition_indices, pass_order


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
            rollout_idx=0,
        )
        self.assertEqual(advantages, [[0.0, 1.0, 1.0]])
        self.assertEqual(returns, [[0.0, 1.0, 1.0]])

    def test_gae_uses_one_discount_for_advantage_and_return(self):
        estimator = GAEAdvantageConfig(gae_lambda=0.0, normalize_actor_advantage=False).build()
        advantages, returns = estimator.compute_gae(
            [_state(labels=[0, 1, 2], reward={"score": 1.0})],
            [[0.0, 0.0]],
            rollout_idx=0,
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
            rollout_idx=0,
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
            rollout_idx=0,
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

    def test_actor_only_switch_rejects_unknown_role(self):
        controller = TrainingController(workers=[])
        with self.assertRaises(KeyError):
            controller.switch_role_modules("critic")

    def test_gae_without_critic_is_rejected(self):
        controller = TrainingController(workers=[], advantage_estimator=GAEEstimator())
        with self.assertRaises(RuntimeError):
            controller.fit([[]], pack_max_length=8, rollout_idx=0)
