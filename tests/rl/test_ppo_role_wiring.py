import unittest

from xtuner.v1.rl.advantage import GAEAdvantageConfig, GAEEstimator
from xtuner.v1.rl.trainer.controller import TrainingController


class TestPPORoleWiring(unittest.TestCase):
    def test_gae_places_outcome_reward_on_the_last_supervised_token(self):
        estimator = GAEAdvantageConfig(gamma=1.0, lam=1.0).build()
        self.assertIsInstance(estimator, GAEEstimator)
        advantages, returns = estimator.compute_token_gae(
            values=[0.0, 0.0, 0.0],
            mask=[False, True, True],
            reward=1.0,
        )
        self.assertEqual(advantages, [0.0, 1.0, 1.0])
        self.assertEqual(returns, [0.0, 1.0, 1.0])

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
