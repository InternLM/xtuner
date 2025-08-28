from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from cyclopts.group import Group

import torch.distributed as dist
from xtuner.v1.train.arguments import TrainingArguments
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils import Config


app = App(name="entrypoint of sft & pretrain")


@app.default()
def main(
    *,
    trainer_cfg_path: Annotated[Path | None, Parameter(group=Group("config-path", sort_key=0))] = None,
    arguments: Annotated[
        TrainingArguments | None, Parameter(group=Group("Training Arguments", sort_key=1), name="*")
    ] = None,
):
    if arguments is not None:
        if trainer_cfg_path is not None:
            raise ValueError("Cannot specify both `trainer_cfg_path` and `arguments`.")
        trainer_cfg = arguments.to_trainer_config()
    else:
        if trainer_cfg_path is None:
            raise ValueError("Must specify either `trainer_cfg_path` or `arguments`.")
        trainer_cfg = Config.fromfile(trainer_cfg_path)["trainer"]

    trainer = Trainer.from_config(trainer_cfg)
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app()
