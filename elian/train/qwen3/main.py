import torch.distributed as dist
from cyclopts import App

from xtuner.v1.train.trainer import Trainer
from train_config import TrainConfig  
from xtuner.v1.model import Qwen3Dense4BConfig
from xtuner.v1.config import LRConfig, AdamWConfig

app = App(
    name="entrypoint of sft & pretrain",
    help="Elian-XTuner's entry point for fine-tuning and training, launched using configuration files or arguments.",
)


@app.default()
def main():
    cfg = TrainConfig()
    model_cfg = Qwen3Dense4BConfig()
    optim_cfg = AdamWConfig(lr=cfg.lr)
    lr_cfg = LRConfig(lr_type=cfg.lr_type, lr_min=cfg.lr_min)
    trainer = Trainer(
        **cfg.to_trainer_kwargs(model_cfg, optim_cfg, lr_cfg)
    )
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    app(exit_on_error=False)
