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
from xtuner.v1.train.rl_async_trainer import RLTrainer
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
):
    if not ray.is_initialized():
        if os.getenv("RAY_MASTER_ADDR"):
            master_addr = os.getenv("RAY_MASTER_ADDR", "127.0.0.1")
            client_port = os.getenv("RAY_CLIENT_PORT", "10001")
            ray_head_address = f"ray://{master_addr}:{client_port}"
            ray.init(address=ray_head_address)
        else:
            ray.init(num_cpus=128)

    if os.getenv("XTUNER_RL_MEM_DIR"):
        print("Start to monitor actor memory")
        track_thread = threading.Thread(target=rl_monitor_actor_memory, args=(os.getenv("XTUNER_RL_MEM_DIR"),))
        track_thread.daemon = True
        track_thread.start()

    trainer_cfg = Config.fromfile(config)["trainer"]
    trainer = RLTrainer.from_config(trainer_cfg)

    require_batches = trainer_cfg.require_batches
    if require_batches ==0:
        print("Start fit")
        trainer.fit()
    else:
        print(f"Start async fit, require_batches={require_batches}")
        trainer.async_fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_cleanup()
    app(exit_on_error=False)
