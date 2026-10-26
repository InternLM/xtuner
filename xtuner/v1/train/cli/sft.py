# `xtuner.compile_cache` only uses the standard library and must run before any
# third-party import: when local compile-cache mode is enabled it repoints
# XTUNER_COMPILE_CACHE_DIR to the node-local warmed cache, which `xtuner/v1/__init__.py`
# reads while setting the per-compiler cache paths.
from xtuner import compile_cache


compile_cache.activate_local_cache()

from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import torch.distributed as dist  # noqa: E402
from cyclopts import App, Parameter  # noqa: E402
from cyclopts.group import Group  # noqa: E402

from xtuner.v1.train.arguments import TrainingArguments  # noqa: E402
from xtuner.v1.train.trainer import Trainer  # noqa: E402
from xtuner.v1.utils import Config  # noqa: E402
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache  # noqa: E402


app = App(
    help="XTuner's entry point for fine-tuning and training, launched using configuration files or arguments.",
)


@app.default()
def main(
    *,
    config: Annotated[Path | None, Parameter(group=Group("config-path", sort_key=0))] = None,
    arguments: Annotated[
        TrainingArguments | None, Parameter(group=Group("Training Arguments", sort_key=1), name="*")
    ] = None,
):
    monkey_patch_hf_modules_cache()

    if arguments is not None:
        if config is not None:
            raise ValueError("Cannot specify both `config` and `arguments`.")
        trainer_cfg = arguments.to_trainer_config()
    else:
        if config is None:
            raise ValueError("Must specify either `config` or `arguments`.")
        trainer_cfg = Config.fromfile(config)["trainer"]

    trainer = Trainer.from_config(trainer_cfg)
    trainer.fit()

    # fit() ends with a global barrier; on abnormal exit this is skipped and the shared
    # cache keeps its last consistent state.
    cache_state = compile_cache.writeback(barrier=dist.barrier if dist.is_initialized() else None)

    if dist.is_initialized():
        dist.destroy_process_group()

    # A shared-filesystem merge can outlast the process-group timeout; keep it outside
    # distributed synchronization while retaining automatic write-back in the CLI.
    if cache_state is not None and cache_state.global_rank == 0:
        compile_cache.merge_staging(cache_state)


if __name__ == "__main__":
    app(exit_on_error=False)
