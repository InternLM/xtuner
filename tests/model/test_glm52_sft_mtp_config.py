"""Regression coverage for the canonical GLM-5.2 SFT recipe.

The released checkpoint advertises one physical MTP layer through
``num_nextn_predict_layers=1``.  The SFT recipe must keep that checkpoint
layout while overriding XTuner's logical/recurrent depth to seven.
"""

import os
import runpy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock


RECIPE_PATH = Path(__file__).resolve().parents[2] / "examples/v1/config/sft_glm5p2.py"


class _CaptureTrainerConfig:
    """Small stand-in that records recipe arguments without initializing a trainer."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _run_glm52_sft_recipe_regression() -> None:
    """The SFT recipe should run one physical MTP block for seven depths."""
    model_cfg = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_config=SimpleNamespace(num_layers=1, share_weights=True),
        attention=SimpleNamespace(),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        recipe_env = {
            "GLM5_2_MODEL_PATH": str(tmp_path / "model"),
            "ALPACA_PATH": str(tmp_path / "alpaca"),
            "WORK_DIR": str(tmp_path / "work"),
            "EP_SIZE": "1",
            "INTRA_LAYER_MICRO_BATCH": "1",
            "WORLD_SIZE": "1",
            "GLOBAL_BATCH_SIZE": "1",
            "SAMPLE_MAX_LENGTH": "4096",
            "PACK_MAX_LENGTH": "16384",
            "TOTAL_STEP": "10",
            "LOSS_MODE": "chunk",
            "LOSS_CHUNK_SIZE": "1024",
            "FP8": "0",
            "MODEL_COMPILE": "0",
            "DATASET_TYPE": "alpaca",
            "DATASET_SAMPLE_RATIO": "1.0",
            "CACHE_TAG": "test",
            "PACK_LEVEL": "soft",
            "PACK_CHUNK_SIZE": "10000",
            "DATALOADER_NUM_WORKERS": "0",
            "PACK_WORKERS": "0",
            "GLOBAL_PACK": "1",
            "GROUP_BY_LENGTH": "1",
            "LR": "1e-6",
            "OPTIMIZER": "adamw",
            "ADAMW_FOREACH": "0",
            "SWAP_OPTIMIZER": "0",
            "LR_TYPE": "cosine",
            "WARMUP_RATIO": "0",
            "CPU_OFFLOAD": "0",
            "TORCH_COMPILE": "0",
            "DISPATCHER": "none",
            "SPARSE_MLA_BACKEND": "torch",
            "SP_SIZE": "1",
            "CHECKPOINT_INTERVAL": "200",
            "CHECKPOINT_MAX_KEEP": "3",
            "HF_INTERVAL": "200",
            "HF_MAX_KEEP": "3",
            "PROFILE_MEMORY": "0",
            "PROFILE_TIME": "0",
            "PROFILE_STEP": "2,3",
            "DEBUG_SKIP_SAVE": "0",
        }
        # Deliberately omit these so they exercise the recipe's true defaults.
        with mock.patch.dict(os.environ, recipe_env, clear=False):
            os.environ.pop("STRICT_LOAD", None)
            os.environ.pop("LOAD_CHECKPOINT_PATH", None)
            with (
                mock.patch("xtuner.v1.model.get_model_config_from_hf", return_value=model_cfg),
                mock.patch("xtuner.v1.train.TrainerConfig", _CaptureTrainerConfig),
            ):
                namespace = runpy.run_path(str(RECIPE_PATH), run_name="__glm52_sft_recipe_test__")

    trainer_cfg = namespace["trainer"]
    recipe_model_cfg = namespace["model_cfg"]

    assert recipe_model_cfg is model_cfg
    assert trainer_cfg.model_cfg is model_cfg
    assert model_cfg.num_nextn_predict_layers == 1
    assert model_cfg.mtp_config.num_layers == 7
    assert model_cfg.mtp_config.share_weights is True
    assert model_cfg.mtp_config.detach_mtp_lm_head_weight is False
    assert model_cfg.mtp_config.detach_mtp_inputs is False
    assert model_cfg.mtp_config.loss_scaling_factor == 0.1
    assert trainer_cfg.strict_load is True


class TestGlm52SftMtpConfig:
    def test_glm52_sft_recipe_uses_recurrent_mtp_depth_without_changing_checkpoint_depth(self) -> None:
        """The SFT recipe should run one physical MTP block for seven depths."""
        _run_glm52_sft_recipe_regression()
