import os

from xtuner.v1.engine import EngineConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    FTDPTokenizeFnConfig,
    Float8Config,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
    ZLossConfig,
)
import ray
from xtuner.v1.ray.train import TrainingController, TrainingWorker
from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers, AcceleratorResourcesConfig



if __name__ == "__main__":
    ray.init()
    QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
    ALPACA_PATH = os.environ["ALPACA_PATH"]

    moe_cfg = Qwen3MoE30BA3Config(
            ep_size=1,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
        )
    
    optim_cfg: AdamWConfig = AdamWConfig()
    fsdp_cfg: FSDPConfig = FSDPConfig(
        torch_compile=True,
        cpu_offload=False,
        ep_size=1,
    )
    dataset_config = [
        dict(dataset=DatasetConfig(name='alpaca', anno_path=ALPACA_PATH, sample_ratio=1.0),
                tokenize_fn=FTDPTokenizeFnConfig()),
    ]

    dataloader_config = DataloaderConfig(
            pack_max_length=512,
            max_length=512,
        )

    engine_cfg = EngineConfig(
        model_cfg=moe_cfg,
        fsdp_cfg=fsdp_cfg,
        optim_cfg=optim_cfg,
    )

    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=8,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    workers, pg = AutoAcceleratorWorkers.from_config(
        TrainingWorker, engine_cfg, resources
    )
    futures = [ worker.test_all_reduce.remote() for worker in workers ]
    print(ray.get(futures))
    train_controller = TrainingController.remote(
        workers=workers,
        dataset_config=dataset_config,
        dataloader_config=dataloader_config,
        tokenizer=QWEN3_MOE_PATH,
        global_batch_size=16,
        work_dir="/tmp/qwen3_moe_test",
        sp_size=1,
        epoch_num=1,
        resume_config=None,
        seed=42,
        debug=False,
    )

    ray.get(train_controller.fit.remote())
    
