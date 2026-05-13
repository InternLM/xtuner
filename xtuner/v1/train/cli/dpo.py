# Copyright (c) OpenMMLab. All rights reserved.
"""
CLI entry point for DPO (Direct Preference Optimization) training.

This module provides a command-line interface for DPO training, similar to
the SFT and RL training interfaces.

Usage:
    torchrun --nproc_per_node=8 xtuner/v1/train/cli/dpo.py --config config.py
"""

from pathlib import Path
from typing import Annotated

import torch.distributed as dist
from cyclopts import App, Parameter
from cyclopts.group import Group

from xtuner.v1.train.dpo_trainer import DPOTrainer
from xtuner.v1.utils import Config

import torch._dynamo
torch._dynamo.config.disable = True



app = App(
    help="XTuner's entry point for DPO (Direct Preference Optimization) training.",
)


@app.default()
def main(
    *,
    config: Annotated[Path, Parameter(group=Group("config-path", sort_key=0))],
):
    """Run DPO training with the given configuration file.
    
    The config file should export a 'trainer' variable of type DPOTrainerConfig.
    
    Example config file:
        from xtuner.v1.train.dpo_trainer import DPOTrainerConfig
        trainer = DPOTrainerConfig(...)
    
    Args:
        config: Path to the DPO training configuration file.
    """
    from xtuner.v1.train.dpo_trainer import DPOTrainerConfig
    
    # Load configuration
    cfg = Config.fromfile(config)
    # The config file should export a 'trainer' key (same pattern as rl.py)
    if "trainer" in cfg:
        trainer_cfg = cfg["trainer"]
    elif "config" in cfg:
        trainer_cfg = cfg["config"]
    elif "trainer_config" in cfg:
        trainer_cfg = cfg["trainer_config"]
    else:
        raise ValueError(
            "Config file must export a 'trainer' variable of type DPOTrainerConfig. "
            "Example: trainer = DPOTrainerConfig(...)"
        )
    # Validate config type
    if not isinstance(trainer_cfg, DPOTrainerConfig):
        raise TypeError(
            f"Expected DPOTrainerConfig, got {type(trainer_cfg).__name__}. "
            "Please ensure your config file exports: trainer = DPOTrainerConfig(...)"
        )
    
    # Create trainer from config
    trainer = DPOTrainer.from_config(trainer_cfg)
    
    # Run training
    trainer.fit()
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app(exit_on_error=False)
