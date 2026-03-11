import os
import unittest
import shutil
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from xtuner.v1.train.rl_colocate_trainer import RLColocateTrainer
from xtuner.v1.train.trainer import XTunerMeta, ExpInfo, LoadCheckpointConfig


class TestRLColocateCheckpoint(unittest.TestCase):
    """Test checkpoint save/resume logic for RLColocateTrainer."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exp_dir = Path(self.temp_dir) / "exp_tracking" / "exp-0"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_save_structure(self):
        """Test that _maybe_save_checkpoint creates correct directory structure."""
        # Create a minimal trainer instance with mocked dependencies
        trainer = Mock(spec=RLColocateTrainer)
        trainer.exp_dir = self.exp_dir
        trainer._CHECKPOINT_DIR = "checkpoints"
        trainer._SAVE_TRAIN_STATE_PATH = "train_state.json"
        trainer._META_PATH = ".xtuner_rl_colocate_trainer"
        trainer._checkpoint_interval = 1
        trainer._checkpoint_maxkeep = -1
        trainer._checkpoint_no_save_optimizer = False

        # Mock the agent_loop_manager and train_controller
        trainer.agent_loop_manager = Mock()
        trainer.agent_loop_manager.save = Mock()
        trainer.train_controller = Mock()
        trainer.logger = Mock()

        # Mock ray.get to return None
        with patch('ray.get', return_value=None):
            # Create meta with a minimal ExpInfo
            exp_info = ExpInfo(
                history=[],
                exp_dir=str(self.exp_dir),
                checkpoint_list=[]
            )
            trainer._meta = XTunerMeta(exps=[exp_info])

            # Call the actual method
            RLColocateTrainer._maybe_save_checkpoint(trainer, cur_step=1)

        # Verify checkpoint directory was created
        checkpoint_path = self.exp_dir / "checkpoints" / "ckpt-step-1"
        self.assertTrue(checkpoint_path.exists())

        # Verify train_state.json was created with correct content
        train_state_path = checkpoint_path / "train_state.json"
        self.assertTrue(train_state_path.exists())

        with open(train_state_path) as f:
            train_state = json.load(f)
            self.assertEqual(train_state["cur_step"], 1)

        # Verify agent_loop_manager.save was called
        trainer.agent_loop_manager.save.assert_called_once_with(checkpoint_path)

    def test_checkpoint_interval_skip(self):
        """Test that checkpoint is skipped when interval doesn't match."""
        trainer = Mock(spec=RLColocateTrainer)
        trainer.exp_dir = self.exp_dir
        trainer._CHECKPOINT_DIR = "checkpoints"
        trainer._checkpoint_interval = 5  # Save every 5 steps
        trainer.agent_loop_manager = Mock()

        # Call with step that doesn't match interval
        RLColocateTrainer._maybe_save_checkpoint(trainer, cur_step=3)

        # Verify no checkpoint was created
        checkpoint_path = self.exp_dir / "checkpoints" / "ckpt-step-3"
        self.assertFalse(checkpoint_path.exists())

        # Verify agent_loop_manager.save was NOT called
        trainer.agent_loop_manager.save.assert_not_called()

    def test_checkpoint_pruning(self):
        """Test checkpoint pruning with checkpoint_maxkeep."""
        trainer = Mock(spec=RLColocateTrainer)
        trainer.exp_dir = self.exp_dir
        trainer._CHECKPOINT_DIR = "checkpoints"
        trainer._SAVE_TRAIN_STATE_PATH = "train_state.json"
        trainer._META_PATH = ".xtuner_rl_colocate_trainer"
        trainer._checkpoint_interval = 1
        trainer._checkpoint_maxkeep = 2  # Keep only last 2 checkpoints
        trainer._checkpoint_no_save_optimizer = False
        trainer.agent_loop_manager = Mock()
        trainer.train_controller = Mock()
        trainer.logger = Mock()

        with patch('ray.get', return_value=None):
            exp_info = ExpInfo(
                history=[],
                exp_dir=str(self.exp_dir),
                checkpoint_list=[]
            )
            trainer._meta = XTunerMeta(exps=[exp_info])

            # Save 3 checkpoints
            RLColocateTrainer._maybe_save_checkpoint(trainer, cur_step=1)
            RLColocateTrainer._maybe_save_checkpoint(trainer, cur_step=2)
            RLColocateTrainer._maybe_save_checkpoint(trainer, cur_step=3)

        checkpoint_dir = self.exp_dir / "checkpoints"

        # Verify only last 2 checkpoints remain
        self.assertFalse((checkpoint_dir / "ckpt-step-1").exists())
        self.assertTrue((checkpoint_dir / "ckpt-step-2").exists())
        self.assertTrue((checkpoint_dir / "ckpt-step-3").exists())

        # Verify meta only tracks last 2 checkpoints
        self.assertEqual(len(trainer._meta.latest_exp.checkpoint_list), 2)

    def test_resolve_load_checkpoint_cfg_auto_resume(self):
        """Test _resolve_load_checkpoint_cfg with auto_resume=True."""
        trainer = Mock(spec=RLColocateTrainer)

        # Create a mock meta with a checkpoint
        mock_meta = Mock()
        mock_meta.latest_checkpoint = str(self.exp_dir / "checkpoints" / "ckpt-step-5")
        trainer._meta = mock_meta

        # Test auto_resume=True
        load_cfg = LoadCheckpointConfig()
        result = RLColocateTrainer._resolve_load_checkpoint_cfg(
            trainer, auto_resume=True, load_checkpoint_cfg=load_cfg
        )

        # Verify checkpoint_path was set from meta
        self.assertEqual(result.checkpoint_path, Path(mock_meta.latest_checkpoint))

    def test_resolve_load_checkpoint_cfg_no_auto_resume(self):
        """Test _resolve_load_checkpoint_cfg with auto_resume=False."""
        trainer = Mock(spec=RLColocateTrainer)

        mock_meta = Mock()
        mock_meta.latest_checkpoint = str(self.exp_dir / "checkpoints" / "ckpt-step-5")
        trainer._meta = mock_meta

        # Test auto_resume=False
        load_cfg = LoadCheckpointConfig()
        result = RLColocateTrainer._resolve_load_checkpoint_cfg(
            trainer, auto_resume=False, load_checkpoint_cfg=load_cfg
        )

        # Verify checkpoint_path was NOT set
        self.assertIsNone(result.checkpoint_path)


if __name__ == "__main__":
    unittest.main()
