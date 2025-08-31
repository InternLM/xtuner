from pathlib import Path
from typing import Annotated

import torch.distributed as dist
from cyclopts import App, Parameter
from cyclopts.group import Group

from xtuner.v1.train.arguments import TrainingArguments
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils import Config


app = App(
    name="entrypoint of sft & pretrain",
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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app(exit_on_error=False)
