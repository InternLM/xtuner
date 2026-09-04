"""Smoke tests for the PPO example configuration.

Loads `examples/v1/config/rl_ppo_qwen3p5_gsm8k.py` and asserts the wiring a
reader is meant to copy: that selecting GAE really does produce a critic, that
the critic is a scalar value model derived from the actor, and that the pieces
which must agree actually do.
"""

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples/v1/config/rl_ppo_qwen3p5_gsm8k.py"

_ENV = {
    "XTUNER_USE_LMDEPLOY": "1",
    "WORK_DIR": "/tmp/xtuner-ppo-example",
    "MODEL_PATH": "/tmp/xtuner-ppo-example/model",
    "DATA_PATH": "/tmp/xtuner-ppo-example/train.jsonl",
    "EVAL_DATA_PATH": "/tmp/xtuner-ppo-example/eval.jsonl",
}


class TestPPOExampleConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, _ENV):
            cls.namespace = runpy.run_path(str(CONFIG_PATH))
        cls.trainer = cls.namespace["trainer"]
        cls.worker = cls.trainer.train_worker_cfg

    def test_config_loads(self) -> None:
        self.assertEqual(type(self.trainer).__name__, "RLColocateTrainerConfig")

    def test_gae_and_critic_are_enabled_together(self) -> None:
        from xtuner.v1.rl.advantage import GAEAdvantageConfig

        self.assertIsInstance(self.trainer.advantage_estimator_config, GAEAdvantageConfig)
        self.assertIsNotNone(self.worker.critic_cfg)

    def test_gamma_and_lambda_reach_the_worker(self) -> None:
        """They are configured once, on the trainer, and pushed down."""
        trainer_cfg = self.trainer.advantage_estimator_config
        worker_cfg = self.worker.advantage_cfg
        self.assertIsNotNone(worker_cfg)
        self.assertEqual(worker_cfg.gamma, trainer_cfg.gamma)
        self.assertEqual(worker_cfg.gae_lambda, trainer_cfg.gae_lambda)

    def test_critic_is_a_scalar_value_model(self) -> None:
        from xtuner.v1.model.value import wants_scalar_value_head

        self.assertTrue(wants_scalar_value_head(self.worker.critic_cfg.model_cfg.text_config))

    def test_critic_uses_a_distinct_device_mesh(self) -> None:
        # Colliding mesh names would make the two models share EP/FSDP meshes.
        actor_prefix = self.worker.model_cfg.text_config.mesh_prefix
        critic_prefix = self.worker.critic_cfg.model_cfg.text_config.mesh_prefix
        self.assertNotEqual(actor_prefix, critic_prefix)

    def test_critic_disables_mtp_and_tying(self) -> None:
        text_cfg = self.worker.critic_cfg.model_cfg.text_config
        self.assertIsNone(text_cfg.mtp_config)
        self.assertFalse(text_cfg.tie_word_embeddings)

    def test_kl_is_applied_to_the_reward_not_the_loss(self) -> None:
        # Doing both would penalize divergence twice.
        self.assertIsNotNone(self.worker.kl_reward_cfg)
        self.assertFalse(self.worker.loss_cfg.use_kl_loss)

    def test_pack_length_is_divisible_by_sp_size(self) -> None:
        # GAE runs on the gathered sequence, so the gather must return exactly
        # pack_max_length.
        self.assertEqual(self.worker.pack_max_length % self.worker.sp_size, 0)

    def test_host_memory_budget_accounts_for_two_models(self) -> None:
        # The GRPO examples use 16 GiB; PPO keeps a second model and its Adam
        # state resident on the host.
        self.assertGreaterEqual(self.trainer.resources.cpu_memory_per_worker, 32 * 1024**3)

    def test_critic_warmup_is_enabled_by_default(self) -> None:
        self.assertGreater(self.worker.critic_cfg.warmup_steps, 0)
