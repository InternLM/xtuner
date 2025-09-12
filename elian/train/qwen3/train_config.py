from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from xtuner.v1.train.trainer import ResumeConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.datasets.config import DatasetCombine, DatasetConfig, DataloaderConfig
from xtuner.v1.datasets.sft_tokenize_fn.openai import OpenaiTokenizeFunctionConfig

@dataclass
class TrainConfig:
    # base path
    model_path: str = "/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-4b/instruct-base"
    work_dir: str = "/data1/nuist_llm/TrainLLM/SFT-elian/xtuner/elian/save/model-01"
    log_dir: str = "/data1/nuist_llm/TrainLLM/SFT-elian/xtuner/elian/save/model-01"

    # data params
    dataset_cfg: list = field(default_factory=lambda: [
        "/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/code/Nemotron-Post-Training-V2-code-coldStart.jsonl",
        "/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/code/Nemotron-Post-Training-V2-code.jsonl",
        "/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/math/Nemotron-Post-Training-V2-math.jsonl",
        "/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/other/Nemotron-Post-Training-V2-math-coldStart.jsonl"
    ] + [f"/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/other/chat-0000{i}-of-00012.jsonl" for i in range(10)] + [
        "/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/other/chat-00010-of-00012.jsonl","/data1/nuist_llm/TrainLLM/datasets/SFT/math/category/other/chat-00011-of-00012.jsonl"
    ])
    cache_dir: str = "/data1/nuist_llm/cacheTem/elianXtuner"
    class_name: str = "JsonlDataset" # TODO @elian: new parquest
    sample_ratio: float = 1.0
    cache_tag: str = "elian-xtuner"
    message_template: str = "qwen3"
    max_token_size: int = 4096
    max_position_embeddings: int = 4096
    collator: str = "sft_llm_collator" # ["sft_llm_collator", "sft_vllm_collator", "fake_collator"]
    pack_level: str = "soft" # ["soft", "none"] # soft is True, none is False for Pack
    pack_max_length: int = 8024 # max_position_size
    pack_workers: int = 8
    num_workers: int = 8


    # train params
    global_batch_size: int = 1
    total_epoch: int = 1

    # fsdp params
    sp_size: int = 2
    tp_size: int = 2
    ep_size: int = 1
    recompute_ratio:float = 1.0
    cpu_offload: bool = False

    # loss params
    mode: str = "chunk"
    chunk_size: int = 1024
    loss_reduction: str = "token" # ["token", "sample", "square"]

    # resume params
    resume_from: Optional[str] = None
    auto_resume: bool = False
    load_optimizer: bool = True
    load_dataset: bool = True
    load_scheduler: bool = True
    strict_load: bool = False

    # save checkpoint step
    hf_interval: Optional[int] = 2000
    hf_max_keep: Optional[int] = 1
    checkpoint_interval: Optional[int] = 1000
    checkpoint_maxkeep: Optional[int] = 2

    # profiling
    profile_step: Optional[int] = 1
    profile_time: bool = True
    profile_memory: bool = True
    intra_layer_micro_batch: int = 1

    # other
    seed: int = 42
    debug: bool = False
    backend: str = "nccl"
    exp_tracker: str = "tensorboard"

    # optim
    lr: float = 6e-5
    weight_decay: float = 0.001
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    lr_type: str = "cosine" # ["cosine", "linear", "constant"]
    warmup_ratio: float = 0.03
    lr_min: float = 1e-6

    def build_resume_cfg(self) -> Optional[ResumeConfig]:
        if self.resume_from or self.auto_resume:
            return ResumeConfig(
                resume_from=self.resume_from,
                auto_resume=self.auto_resume,
                load_optimizer=self.load_optimizer,
                load_dataset=self.load_dataset,
                load_scheduler=self.load_scheduler,
            )
        return None
    
    def build_fsdp_cfg(self) -> Optional[FSDPConfig]:
        if self.tp_size > 1 or self.ep_size > 1:
            return FSDPConfig(
                tp_size = self.tp_size,
                sp_size = self.sp_size,
                ep_size = self.ep_size,
                cpu_offload = self.cpu_offload
            )
        else:
            return None

    def build_loss_cfg(self) -> Optional[CELossConfig]:
        if self.mode!="eager" or self.loss_reduction!="token":
            return CELossConfig(
                mode = self.mode,
                chunk_size = self.chunk_size,
                loss_reduction = self.loss_reduction
            )
        else:
            return None
        
    def build_datasets_cfg(self) -> list[DatasetCombine]:
        all_datasets = []
        for data_file in self.dataset_cfg:
            data_path = Path(data_file)
            name = data_path.stem
            tokenize_fn_cfg = OpenaiTokenizeFunctionConfig(
                        chat_template=self.message_template,
                        max_length=self.max_token_size
                    )
            all_datasets.append(
                {
                "dataset":DatasetConfig(
                    anno_path=data_file,
                    cache_dir=self.cache_dir,
                    name=name,
                    cache_tag=self.cache_tag,
                    class_name=self.class_name,
                    sample_ratio=self.sample_ratio
                ),
                "tokenize_fn":tokenize_fn_cfg
                }
            )
        return all_datasets
    
    def build_dataloader(self) -> DataloaderConfig:
        return DataloaderConfig(
            collator = self.collator,
            pack_level = self.pack_level,
            pack_max_length = self.pack_max_length,
            pack_workers = self.pack_workers,
            num_workers = self.num_workers
        )
        
    def to_trainer_kwargs(self, model_cfg, optim_cfg, lr_cfg):
        return dict(
            model_cfg=model_cfg,
            tokenizer_path=self.model_path,
            load_from=self.model_path,
            optim_cfg=optim_cfg,
            lr_cfg=lr_cfg,
            global_batch_size = self.global_batch_size,
            work_dir = self.work_dir,
            log_dir = self.log_dir,
            sp_size = self.sp_size,
            total_epoch = self.total_epoch,
            checkpoint_interval = self.checkpoint_interval,
            checkpoint_maxkeep = self.checkpoint_maxkeep,
            hf_interval = self.hf_interval,
            hf_max_keep = self.hf_max_keep,
            exp_tracker = self.exp_tracker,
            profile_step = self.profile_step,
            profile_time = self.profile_time,
            profile_memory = self.profile_memory,
            intra_layer_micro_batch = self.intra_layer_micro_batch,
            seed = self.seed,
            debug = self.debug,
            backend = self.backend,
            resume_cfg=self.build_resume_cfg(),
            fsdp_cfg=self.build_fsdp_cfg(),
            loss_cfg=self.build_loss_cfg(),
            dataset_cfg=self.build_datasets_cfg(),
            dataloader_cfg=self.build_dataloader()
        )

    def __str__(self):
        cfg_dict = asdict(self)
        max_key_len = max(len(k) for k in cfg_dict.keys())
        lines = []
        for k, v in cfg_dict.items():
            lines.append(f"{k:<{max_key_len}} : {v}")
        return "\n".join(lines)

if __name__ == "__main__":
    cfg = TrainConfig()
    print(cfg)