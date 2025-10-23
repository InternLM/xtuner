import os
from pathlib import Path
from typing import Annotated

import ray
import torch.distributed as dist
from cyclopts import App, Parameter
from cyclopts.group import Group

from xtuner.v1.train.rl_trainer import RLTrainer
from xtuner.v1.utils import Config


app = App(
    help="XTuner's entry point for fine-tuning and training, launched using configuration files or arguments.",
)


@app.default()
def main(
    *,
    config: Annotated[Path, Parameter(group=Group("config-path", sort_key=0))],
):
    if not ray.is_initialized():
        master_addr = os.getenv("RAY_MASTER_ADDR", "127.0.0.1")
        client_port = os.getenv("RAY_CLIENT_PORT", "10001")
        ray_head_address = f"ray://{master_addr}:{client_port}"
        ray.init(address=ray_head_address)
    trainer_cfg = Config.fromfile(config)["trainer"]
    trainer = RLTrainer.from_config(trainer_cfg)
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app(exit_on_error=False)
