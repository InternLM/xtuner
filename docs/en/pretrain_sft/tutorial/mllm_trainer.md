# Fine-tuning Multimodal Large Models with Trainer

In the previous [tutorial](../../get_started/mllm_sft.md), we launched a fine-tuning training session through the command line in the simplest way, and behind this quick start is XTuner's core component `Trainer` at work. In this section, we will get to know Trainer and use a more granular approach to control various aspects of training.

Before reading this tutorial, please read [Fine-tuning Large Models with Trainer](./llm_trainer.md).

## Selecting a Model:

Trainer builds models through configuration files. Let's take XTuner's built-in support for `Intern-S1-mini` as an example to quickly obtain a model configuration instance

```{code-block} python
:caption: Building Model Configuration

from xtuner.v1.model import InternS1MiniConfig

model_cfg = InternS1MiniConfig()
```

If we want to modify certain parameters of the model, such as reducing the number of model layers, we can do this:

```{tip}
:class: margin

Why not jump to the source code of InternS1MiniConfig to see what parameters can be configured?
```

```{code-block} python
:caption: Modifying Model Layers

from xtuner.v1.model import Qwen3Dense8BConfig
text_cfg = Qwen3Dense8BConfig(num_hidden_layers=16)
model_cfg = InternS1MiniConfig(text_config=text_cfg)
```

Note: If the number of layers is modified, the weights cannot be fully loaded.

## Selecting a Dataset:

Trainer also builds datasets through configuration files. Let's take the jsonl format data used in the previous tutorial as an example to quickly obtain a dataset configuration instance

Dataset format reference [documentation](../../get_started/mllm_sft.md#sft-dataset)


```{tip}
:class: margin

Data loading too slow every time you start? Why not set `cache_dir`?
```

```{code-block} python
:caption: Building Data Configuration

from xtuner.v1.datasets import (
    DataloaderConfig,
    DatasetConfig,
    InternS1VLTokenizeFnConfig
)

sample_max_length = 8192 # Maximum length of a single sample, will be truncated if exceeded, and warning will be output
pack_max_length = 16384 # Maximum length that can be contained in one training iter, pack mechanism will try to concatenate multiple samples together to reduce padding
# If your GPU memory is insufficient, you can appropriately reduce the above two parameters, but please ensure sample_max_length <= pack_max_length

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text', # Data alias
                                 # Annotation file path, can be a single jsonl or a folder, will automatically traverse all jsonl files in the current folder
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl', # Pure text data
                                 sample_ratio=5.0, # Data sampling ratio, here is repeated 5 times, can be a decimal
                                 class_name='VLMJsonlDataset'), # Corresponding dataset class name
        # A dataset needs to be paired with a corresponding tokenizer fun function to process single item data output by the dataset
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media', # Data alias
                                 anno_path='tests/resource/mllm_sft_single_image_example_data.jsonl', # Multimodal data
                                 media_root='tests/',
                                 sample_ratio=20.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
# dataloader configuration
dataloader_config = DataloaderConfig(dataset_config_list=dataset_config,
                                     pack_max_length=pack_max_length,
                                     num_workers=8,
                                     collator='intern_s1_vl_sft_collator')

```

The above builds a relatively general dataset example, where `dataset_config` defines 2 datasets, namely pure text data and multimodal data, and specifies their respective sampling ratios, while `dataloader_config` defines the relevant parameters of the data loader.

Through the above flexible configuration combination method, users can easily configure various datasets and control their respective sampling ratios.


## Selecting Optimizer and Learning Rate Scheduler:

```{code-block} python
:caption: Optimizer & LR Scheduler

from xtuner.v1.config import LRConfig, AdamWConfig

optim_cfg = AdamWConfig(lr=1e-6, foreach=False) # Different modules have different device meshes, foreach must be False
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)
```

## Building Trainer Configuration


After completing the above steps and building the core components of Trainer, we can build a Trainer instance:

```{code-block} python
:caption: Building Trainer

from xtuner.v1.train import Trainer
from xtuner.v1.loss import CELossConfig

load_from = "<model path>" # If in fine-tuning mode, must specify, otherwise will train from scratch
tokenizer = "<tokenizer path, usually same as model path>"

trainer = TrainerConfig(
    load_from=load_from, # If in fine-tuning mode, must specify, otherwise will train from scratch
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    # Global batch size
    # Assuming 8-card training, then each card's forward shape is (1, pack_max_length), gradient accumulation times is 1
    # Assuming 4-card training, then each card's forward shape is (1, pack_max_length), gradient accumulation times is 2 (automatically converted)
    global_batch_size=8,
    epoch_num=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024), # Can significantly reduce GPU memory usage, recommended to always enable
)

```


## Launch Training

Complete code is as follows:

````{toggle}
```python
from xtuner.v1.model import InternS1MiniConfig
from xtuner.v1.train import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    LRConfig
)

from xtuner.v1.datasets import InternS1VLTokenizeFnConfig, DataloaderConfig, DatasetConfig,
from xtuner.v1.loss import CELossConfig

# model config
model_cfg = InternS1MiniConfig()

# dataset and dataloader config
sample_max_length = 8192
pack_max_length = 16384

dataset_config = [
    {
        "dataset": DatasetConfig(name='pure_text',
                                 anno_path='tests/resource/mllm_sft_text_example_data.jsonl',
                                 sample_ratio=5.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
    {
        "dataset": DatasetConfig(name='media',
                                 anno_path='tests/resource/mllm_sft_single_image_example_data.jsonl',
                                 media_root='tests/',
                                 sample_ratio=20.0,
                                 class_name='VLMJsonlDataset'),
        "tokenize_fn": InternS1VLTokenizeFnConfig(model_cfg=model_cfg, max_length=sample_max_length),
    },
]
dataloader_config = DataloaderConfig(dataset_config_list=dataset_config,
                                     pack_max_length=pack_max_length,
                                     num_workers=8,
                                     pack_level="expand_soft",
                                     collator='intern_s1_vl_sft_collator')

# optimizer and lr config
optim_cfg = AdamWConfig(lr=1e-6, foreach=False)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=0)

load_from = "<model path>" # If in fine-tuning mode, must specify, otherwise will train from scratch
tokenizer = "<tokenizer path, usually same as model path>"

# trainer config
trainer = TrainerConfig(
    load_from=load_from,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    tokenizer_path=tokenizer,
    global_batch_size=8,
    epoch_num=2,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024)
)
trainer.fit()
```
````

After writing the above Python script, name it `toy_train.py`, and we can launch distributed training through `torchrun`:

```{code-block} bash
:caption: Launch Training

torchrun --nproc_per_node=8 toy_train.py
```

Congratulations, you have implemented a XTuner training entry yourself! You can fully customize your training parameters in this script.