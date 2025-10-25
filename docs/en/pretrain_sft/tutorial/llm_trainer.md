(trainer-sft)=
# Fine-tuning Large Models with Trainer

In the previous [tutorial](../../get_started/sft.md), we launched a fine-tuning training session through the command line in the simplest way, and behind this quick start is XTuner's core component `Trainer` at work. In this section, we will get to know Trainer and use a more granular approach to control various aspects of training.


(model-cfg)=
## Selecting a Model:

Trainer builds models through configuration files. Let's take XTuner's built-in support for `Qwen3 8B` as an example to quickly obtain a model configuration instance

```{code-block} python
:caption: Building Model Configuration

from xtuner.v1.model import Qwen3Dense8BConfig

model_cfg = Qwen3Dense8BConfig()
```

If we want to modify certain parameters of the model, such as reducing the number of model layers, we can do this:

```{tip}
:class: margin

Why not jump to the source code of Qwen3Dense8BConfig to see what parameters can be configured?
```

```{code-block} python
:caption: Modifying Model Layers

model_cfg = Qwen3Dense8BConfig(num_hidden_layers=16)
```


## Selecting a Dataset:

Trainer also builds datasets through configuration files. Let's take the jsonl format data used in the previous tutorial as an example to quickly obtain a dataset configuration instance

Dataset format reference [documentation](../../get_started/sft.md#sft-dataset)

```{tip}
:class: margin

Data loading too slow every time you start? Why not set `cache_dir`?
```

```{code-block} python
:caption: Building Data Configuration


dataset_cfg = []
dataloader_cfg =

```


## Selecting Optimizer and Learning Rate Scheduler:

```{code-block} python
:caption: Optimizer & LR Scheduler

from xtuner.v1.config import LRConfig, AdamWConfig


optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
```

## Building Trainer Configuration


After completing the above steps and building the core components of Trainer, we can build a Trainer instance:

```{code-block} python
:caption: Building Trainer

from xtuner.v1.train import Trainer


load_from = "<model path>" # If in fine-tuning mode, must specify, otherwise will train from scratch
tokenizer = "<tokenizer path, usually same as model path>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
)

```


## Launch Training

Complete code is as follows:

````{toggle}
```python
from xtuner.v1.model import Qwen3Dense8BConfig
from xtuner.v1.config import LRConfig, AdamWConfig
from xtuner.v1.train import Trainer


model_cfg = Qwen3Dense8BConfig()
dataset_cfg = []
dataloader_cfg =
optim_cfg = AdamWConfig(lr=6e-05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)

load_from = "<model path>" # If in fine-tuning mode, must specify, otherwise will train from scratch
tokenizer = "<tokenizer path, usually same as model path>"

trainer = Trainer(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
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