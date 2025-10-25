# Training Configuration

In the previous [tutorial](../../get_started/sft.md), we launched a fine-tuning training session through the command line in the simplest way. However, in actual use, we often need more fine-grained control over the training process, such as setting learning rate, parallel strategy, sequence length and other parameters. At this time, using a configuration file to manage experiment configuration will be more convenient! After reading this tutorial, I believe you will gain something.

## Command Line Launch

Before officially introducing the configuration file, let's review the previous command line entry:

```{code-block} bash
python xtuner/v1/train/cli/sft.py --help
```

```{hint}
:class: margin
If you think the training entry doesn't suit your needs? You can also DIY it yourself.
```

There are quite a few parameters, right? Let's sort them out. Currently, the training entry supports two startup methods: passing in a configuration file or directly passing command line parameters, **these two methods are mutually exclusive**. That is to say, if you use a configuration file, you cannot use command line parameters anymore, and vice versa. Simply summarize:

- **Command Line Parameters**
  - 😊 Simple and fast, suitable for simple and quick experiments. For example, just changing the dataset, model path, parallel strategy or training steps, etc.
  - 😅 Limited scalability, cannot use custom modules, configuration granularity is relatively coarse

- **Configuration File**
  - 😊 Super fine configuration granularity, strong scalability, supports custom modules, version management is also very convenient
  - 😅 Have to write the configuration file yourself, a bit of a threshold

```{note}
:class: margin
Actually, the command line parameter function is also very rich, it is recommended to look at the `--help` output, there may be surprises.
```

Seeing this, I believe you already have a clear idea, next let's focus on how to use the configuration file.


## Configuration File

XTuner adopts Python-style configuration files, allowing you to fully utilize Python's syntax features to import core components, and enjoy a smoother configuration experience in editors that support syntax prompts.

```{tip}
:class: margin
If you have already seen the [trainer tutorial](./trainer.md), you can directly jump to the [Building TrainerConfig](TrainerConfig) section.
```

Simply put, XTuner's training configuration revolves around `Trainer`. Writing a training configuration is essentially building a `TrainerConfig` instance. Next, let's get it done step by step!

### Building Model Configuration

Taking XTuner's built-in `Qwen3 8B` as an example, getting the model configuration is this simple:

```{code-block} python
:caption: Building Model Configuration

from xtuner.v1.model import Qwen3Dense8BConfig

model_cfg = Qwen3Dense8BConfig()
```

### Building Data Configuration

Dataset format reference [documentation](../../get_started/sft.md#sft-dataset)

```{tip}
:class: margin

Data loading too slow every time you start? Why not set `cache_dir`?
```

```{code-block} python
:caption: Building Data Configuration

# Dataset configuration
dataset_cfg = []
# Data loader configuration
dataloader_cfg =
```

### Building Optimizer and Learning Rate Scheduler

```{code-block} python
:caption: Optimizer & Learning Rate Scheduler Configuration

from xtuner.v1.config import LRConfig, AdamWConfig

# Optimizer configuration
optim_cfg = AdamWConfig(lr=6e-05)
# Learning rate scheduler configuration
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
```

(TrainerConfig)=
### Building TrainerConfig

Finally, we need to integrate all configurations! Look here:

```{code-block} python
:caption: Building Complete TrainerConfig

from xtuner.v1.train import TrainerConfig

# Basic path configuration
load_from = "<model path>"  # Must specify in fine-tuning mode, otherwise will train from scratch
tokenizer_path = "<tokenizer path, usually same as model path>"

# Integrate all configurations
trainer = TrainerConfig(
    model_cfg=model_cfg,
    tokenizer_path=tokenizer_path,
    load_from=load_from,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    work_dir="<target working directory>",
)
```

## Launch Training

After the configuration file is ready, launching training is a piece of cake! Just pass the configuration file path to the training entry:

```{code-block} bash
:caption: Launch Training

python xtuner/v1/train/cli/sft.py --config <configuration file path>
```