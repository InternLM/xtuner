import os
import argparse

from xtuner.v1.engine import EngineConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    Float8Config,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
    ZLossConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
import ray
from xtuner.v1.ray.train import TrainingWorker
from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers, AcceleratorResourcesConfig
from xtuner.v1.config.trainer import TrainerConfig
from xtuner.v1.train.trainer import Trainer
import torch


def test_ray(trainer_cfg):
    ray.init()
    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=8,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    workers, pg = AutoAcceleratorWorkers.from_config(
        TrainingWorker, trainer_cfg, resources
    )
    futures = [ worker.test_all_reduce.remote() for worker in workers ]
    print(ray.get(futures))
    handles = [worker.fit.remote() for worker in workers]
    print(ray.get(handles))
    return


def test_torchrun(rank, trainer_cfg):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "8"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29000"
    os.environ["LOCAL_RANK"] = str(rank)
    trainer = Trainer.from_config(trainer_cfg)
    trainer.fit()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--model-path', type=str, default=os.environ["QWEN3_MOE_PATH"])
    parser.add_argument('--data-path', type=str, default=os.environ["ALPACA_PATH"])
    args = parser.parse_args()

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
    dataset_cfg = [
        dict(dataset=DatasetConfig(name='alpaca', anno_path=args.data_path, sample_ratio=1.0),
                tokenize_fn=FTDPTokenizeFnConfig()),
    ]

    dataloader_cfg = DataloaderConfig(
            pack_max_length=512,
            max_length=512,
        )

    engine_cfg = EngineConfig(
        model_cfg=moe_cfg,
        fsdp_cfg=fsdp_cfg,
        optim_cfg=optim_cfg,
    )
    lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)

    trainer_cfg = TrainerConfig(
        model_cfg=moe_cfg,
        load_from=args.model_path,
        tokenizer_path=args.model_path,
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        optim_cfg=optim_cfg,
        lr_cfg=lr_cfg,
        fsdp_cfg=fsdp_cfg,
        global_batch_size=16,
        epoch_num=1,
        work_dir="/tmp/qwen3_moe_test",
        seed=42,
    )

    if args.ray:
        test_ray(trainer_cfg)
    else:
        torch.multiprocessing.spawn(test_torchrun, args=(trainer_cfg, ), nprocs=8)
        test_torchrun(trainer_cfg)
    