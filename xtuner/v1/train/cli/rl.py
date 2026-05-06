import os
import threading
import time
from pathlib import Path
from typing import Annotated

import ray
import torch.distributed as dist
from cyclopts import App, Parameter
from cyclopts.group import Group

from xtuner.v1.rl.utils import register_cleanup
from xtuner.v1.utils import Config
from xtuner.v1.utils.track_rl_mem import monitor_actor_memory


app = App(
    help="XTuner's entry point for fine-tuning and training, launched using configuration files or arguments.",
)


def rl_monitor_actor_memory(work_dir, interval: int = 60):
    while True:
        try:
            ray.init(address="auto")
            time.sleep(interval)
            break
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception:
            print("连接 Ray 集群失败, 等等")

    monitor_actor_memory(work_dir=work_dir, interval=interval)


@app.default()
def main(
    *,
    config: Annotated[Path, Parameter(group=Group("config-path", sort_key=0))],
    work_dir: str | None = None,
    num_workers: int | None = None,
):
    if not ray.is_initialized():
        ray.init(address="auto")

    if os.getenv("XTUNER_RL_MEM_DIR"):
        print("Start to monitor actor memory")
        track_thread = threading.Thread(target=rl_monitor_actor_memory, args=(os.getenv("XTUNER_RL_MEM_DIR"),))
        track_thread.daemon = True
        track_thread.start()

    cfg = Config.fromfile(config)
    if work_dir is not None:
        cfg.trainer.work_dir = work_dir
    if num_workers is not None:
        cfg.trainer.resources.num_workers = num_workers
    trainer = cfg.trainer.build()
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_cleanup()
    app(exit_on_error=False)
