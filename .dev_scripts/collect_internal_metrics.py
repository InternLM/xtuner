"""Collect internal metrics from the first batch of the dataloader configured in a trainer config.

Loads model weights from ``--hf-path`` and runs a single dummy forward on the first batch,
then prints the flattened metrics as JSON.

Usage:
    torchrun --nproc-per-node <N> .dev_scripts/collect_internal_metrics.py \\
        <config-path> --hf-path <hf-path> [--output <output.json>]
"""

import json

import torch.distributed as dist
from cyclopts import App, Parameter
from pathlib import Path
from typing import Annotated

from xtuner.v1.train.trainer import (
    LoadCheckpointConfig,
    ResumeConfig,
    Trainer,
)
from xtuner.v1.utils import Config
from xtuner.v1.utils.internal_metrics import InternalMetricsConfig, flatten_internal_metrics_for_logs
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


usage = """Usage
torchrun --nproc-per-node <N> .dev_scripts/collect_internal_metrics.py <config-path> --hf-path <hf-path> [--output <output.json>]

Arguments:
  <config-path>     XTuner Python config file (must expose a `trainer` key of type TrainerConfig)
  --hf-path         HuggingFace model path, overrides `load_from` in the config
  --output          Optional path to save flattened metrics as JSON (default: print to stdout on rank 0)
"""

cli = App(usage=usage)


@cli.default
def collect_internal_metrics(
    config_path: Annotated[
        Path,
        Parameter(help="Path to the XTuner Python config file"),
    ],
    hf_path: Annotated[
        Path,
        Parameter(help="HuggingFace model path, overrides load_from in the config"),
    ],
    output: Annotated[
        Path | None,
        Parameter(help="Optional path to save flattened metrics as JSON"),
    ] = None,
) -> None:
    monkey_patch_hf_modules_cache()

    trainer_cfg = Config.fromfile(config_path)["trainer"]

    trainer_cfg.load_from = hf_path
    trainer_cfg.resume_cfg = ResumeConfig()
    trainer_cfg.load_checkpoint_cfg = LoadCheckpointConfig()

    if trainer_cfg.internal_metrics_cfg is None:
        trainer_cfg.internal_metrics_cfg = InternalMetricsConfig()
    trainer_cfg.internal_metrics_cfg.internal_metrics_interval = 1

    trainer = Trainer.from_config(trainer_cfg)

    data_batch = next(iter(trainer._data_iter()))
    engine_input = trainer._prepare_model_input(data_batch)

    # fp8 models require scales to be precomputed before every forward (normally done inside
    # train_step); call it explicitly here since we skip train_step entirely.
    trainer._engine._maybe_precompute_float8_dynamic_scale_for_fsdp()

    # pop_metrics runs its own dummy forward; no train_step / optimizer step needed.
    internal_metrics = trainer._metrics_recorder.pop_metrics(engine_input)

    if dist.get_rank() == 0:
        flat = flatten_internal_metrics_for_logs(internal_metrics)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(flat, indent=2))
            print(f"Internal metrics saved to {output}")
        else:
            print(json.dumps(flat, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cli()
