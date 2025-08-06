# flake8: noqa=E501
from pathlib import Path
from typing import Annotated, Literal, cast

from cyclopts import Parameter
from cyclopts.group import Group
from mmengine import list_dir_or_file
from pydantic import BaseModel, ConfigDict

from xtuner.v1.config import FSDPConfig, TrainerConfig
from xtuner.v1.config.base_model import TransformerConfig
from xtuner.v1.config.data import DataloaderConfig, DatasetConfig, DatasetConfigList
from xtuner.v1.config.optim import AdamWConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.model import get_model_config, get_model_config_from_hf
from xtuner.v1.utils import Config, get_logger, is_hf_model_path


logger = get_logger()


# Define groups with explicit ordering
model_group = Group("model", sort_key=0)
dataset_group = Group("dataset", sort_key=1)
optimizer_group = Group("optimizer", sort_key=2)
lr_scheduler_group = Group("lr-scheduler", sort_key=3)
training_group = Group("training", sort_key=4)
checkpoint_group = Group("checkpoint", sort_key=5)
parallel_group = Group("fsdp-parallel", sort_key=6)


@Parameter(name="*")
class TrainingArguments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # model
    load_from: Annotated[str | None, Parameter(group=model_group, help="load checkpoint from")] = None
    model_cfg: Annotated[
        str | None,
        Parameter(group=model_group, help="model config path or name, choose one of from ['qwen3-moe-30BA3']"),
    ] = None
    tokenizer_path: Annotated[
        Path | None,
        Parameter(
            group=model_group,
            help="tokenizer path. If load_from is an `transformer` model dir, this arguments is not required ",
        ),
    ] = None
    # dataset
    dataset: Annotated[Path, Parameter(group=dataset_group, help="dataset config path or jsonl file or dir")]
    dataloader: Annotated[
        DataloaderConfig, Parameter(group=dataset_group, help="dataset config path or jsonl file or dir", name="*")
    ]
    # optimizer
    lr: Annotated[float, Parameter(group=optimizer_group, help="learning rate")] = 6e-5
    optim: Annotated[Literal["AdamW"], Parameter(group=optimizer_group, help="optimizer type")] = "AdamW"
    # lr-scheduler
    lr_min: Annotated[float, Parameter(group=lr_scheduler_group, help="minimum learning rate")] = 1e-6
    scheudler_type: Annotated[
        Literal["cosine", "linear", "constant"], Parameter(group=lr_scheduler_group, help="scheduler type")
    ] = "cosine"
    warmup_ratio: Annotated[float, Parameter(group=lr_scheduler_group, help="warmup ratio")] = 0.03
    # training
    total_step: Annotated[int | None, Parameter(group=training_group, help="total training steps")] = None
    epoch_num: Annotated[int, Parameter(group=training_group, help="number of epochs")] = 1
    work_dir: Annotated[Path | None, Parameter(group=training_group, help="working directory of trainer")] = None
    global_batch_size: Annotated[
        int | None, Parameter(group=training_group, help="Global training batch size, defaults to `dp` size")
    ] = None
    # checkpoint
    load_model: Annotated[bool, Parameter(group=checkpoint_group, help="load model from checkpoint")] = True
    load_optimizer: Annotated[bool, Parameter(group=checkpoint_group, help="load optimizer from checkpoint")] = True
    load_dataset: Annotated[bool, Parameter(group=checkpoint_group, help="load dataset state from checkpoint")] = True
    load_scheduler: Annotated[bool, Parameter(group=checkpoint_group, help="load scheduler state from checkpoint")] = (
        True
    )
    fsdp_config: Annotated[FSDPConfig | None, Parameter(group=parallel_group, help="FSDP configuration")] = None

    def model_post_init(self, _):
        if self.tokenizer_path is None:
            load_from = self.load_from
            assert load_from is not None, "Transformer model path should be set if `tokenizer_path` is None"
            assert is_hf_model_path(load_from), (
                "Transformer model path should be a valid HuggingFace model path if `tokenizer_path` is None"
            )
            self.tokenizer_path = cast(Path, Path(load_from))

    def to_trainer_config(self):
        # Load dataset config
        dataset_cfg = self._get_dataset_config()

        # Create optimizer config
        optim_cfg = AdamWConfig(lr=self.lr)

        # Create learning rate scheduler config
        lr_cfg = LRConfig(lr_type=self.scheudler_type, warmup_ratio=self.warmup_ratio, lr_min=self.lr_min)

        # Create dataloader config (using defaults for now)
        dataloader_cfg = self.dataloader
        model_cfg = self._get_model_config()
        resume_cfg = None

        return TrainerConfig(
            model_cfg=model_cfg,
            load_from=self.load_from,
            tokenizer_path=self.tokenizer_path,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            optim_cfg=optim_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=self.fsdp_config,
            global_batch_size=self.global_batch_size,
            total_step=self.total_step,
            epoch_num=self.epoch_num,
            resume=resume_cfg,
            work_dir=self.work_dir,
        )

    def _get_dataset_config(self) -> DatasetConfigList:
        dataset_cfg = self.dataset
        if dataset_cfg.suffix == ".py":
            config = Config.fromfile(dataset_cfg)
            if "datasets" not in config:
                raise ValueError(f"Dataset config file {dataset_cfg} does not contain `datasets` key.")
            return config["datasets"]
        else:
            dataset_path = dataset_cfg
            dataset_path.exists()

            jsonl_list: list[Path]

            if dataset_path.is_file():
                assert dataset_path.suffix in [".jsonl"], "Dataset file must be a JSONL or JSON file."
                jsonl_list = [dataset_path]
            else:
                assert dataset_path.is_dir(), "Dataset path must be a directory or a JSONL file."
                jsonl_list = [dataset_path / f for f in list_dir_or_file(dataset_path, suffix="jsonl", list_dir=False)]

            ret: DatasetConfigList = []
            for jsonl_file in jsonl_list:
                if not jsonl_file.exists():
                    raise FileNotFoundError(f"Dataset file {jsonl_file} does not exist.")
                tokenize = FTDPTokenizeFnConfig()
                dataset = DatasetConfig(anno_path=jsonl_file)
                ret.append({"dataset": dataset, "tokenize_fn": tokenize})
            return ret

    def _get_resume_config(self):
        # TODO:
        ...

    def _get_model_config(self) -> TransformerConfig:
        if self.model_cfg is not None:
            if self.model_cfg.endswith(".py"):
                model_cfg = Config.fromfile(self.model_cfg)
                assert "model" in model_cfg, "Model configuration must contain 'model' key."
                return model_cfg["model"]
            else:
                if (model_cfg := get_model_config(self.model_cfg)) is not None:
                    return model_cfg

        assert self.load_from is not None, "`load_from` must be set if `model_cfg` is not set"

        model_path = Path(self.load_from)
        if is_hf_model_path(self.load_from):
            return get_model_config_from_hf(model_path)
        else:
            raise NotImplementedError
