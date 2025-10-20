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
    ray_cluster_url = os.getenv("RAY_CLUSTER_URL", "auto")
    if ray_cluster_url == "auto":
        ray.init(num_cpus=128, ignore_reinit_error=True)
    else:
        ray.init(address=ray_cluster_url, ignore_reinit_error=True)

    trainer_cfg = Config.fromfile(config)["trainer"]
    trainer = RLTrainer.from_config(trainer_cfg)
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app(exit_on_error=False)
