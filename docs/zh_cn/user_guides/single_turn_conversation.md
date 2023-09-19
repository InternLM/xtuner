# 单轮对话 data pipeline

单轮对话指令微调旨在提升模型回复特定指令的能力，其数据处理流程可以分为以下两部分：

1. 按照相应数据集格式构造数据
2. 向数据集中插入对话模板（可选）

XTuner 支持使用 HuggingFace Hub 数据集、Alpaca 格式的自定义数据集以及其他格式的自定义数据集进行 SFT（Supervised FineTune）。三者的主要区别在于：

1. 使用 HuggingFace Hub 数据集时需要将原始数据映射为 XTuner 定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)；
2. 使用 Alpaca 格式的自定义数据集时，需要保证自定义数据集至少包含'instruction', 'input', 'output'三列；
3. 对于自定义数据集则推荐用户按照[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)构造数据集，**这会大幅度缩小数据预处理所消耗的时间**。

## 使用 HuggingFace Hub 数据集

### Step 1, 映射原始数据集为标准格式

由于不同数据集的格式各有不同，因此需要将原始数据映射为 XTuner 定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)。XTuner 支持通过 map function 来实现格式的映射。下面以 [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 数据集为例介绍如何实现数据映射。

alpaca 数据集格式如下所示：

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='tatsu-lab/alpaca')
>>> ds['train']
Dataset({
    features: ['instruction', 'input', 'output', 'text'],
    num_rows: 52002
})
```

由此可见，alpaca train dataset 有 52002 行，4 列，列名分别为 'instruction', 'input', 'output', 'text'。'instruction' 和 'input' 给出了问题描述，'output' 为对应 GroundTruth 回答。[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)中介绍了单轮对话指令微调过程中，数据格式应该为：

```json
[{
    "conversation":[
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

因此，可以通过下面的 map function 将原始数据映射为标准格式：

```python
# 假设将该函数存放在./map_fn.py文件中
def alpaca_map_fn(example):
    """
    >>> train_ds = ds['train'].map(alpaca_map_fn)
    >>> train_ds
    Dataset({
        features: ['instruction', 'input', 'output', 'text', 'conversation'],
        num_rows: 52002
    })
    >>> train_ds[0]['conversation']
    [{'input': 'xxx', 'output': 'xxx'}]
    """
    if example.get('output', '') == '<nooutput>':
        return {'conversation': [{'input': '', 'output': ''}]}
    else:
        return {
            'conversation': [{
                'input': example['input'],
                'output': example['output']
            }]
        }
```

### Step 2, 列出候选模型名字

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称。

### Step 3, 复制 config 文件

如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

例如通过下列命令将名为 `internlm_7b_qlora_alpaca_e3` 的 config 导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

### Step 4 设置对话模板（可选）

对话模板是指用于生成对话的预定义模式或结构。这些模板可以包含问句、回答或多轮对话中的不同角色的发言。在训练数据集中加入对话模板有利于模型生成有结构和逻辑的对话，并提供更准确、一致和合理的回答。

不同数据集、不同语言模型可能对应着不同的对话模板。例如，[alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 数据集的对话模板如下：

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
xxx

### Assistant:
xxx
```

XTuner提供了一系列对话模板，你可以在 `xtuner/utils/templates.py` 中找到。其中，`INSTRUCTION_START` 和 `INSTRUCTION` 分别代表第一轮对话和后续若干轮对话所使用的对话模板。在单轮对话数据集（如 `alpaca`）中只会用到 `INSTRUCTION_START`。

### Step 5, 修改 config 文件

对Step 3 复制得到的 config 文件需要进行如下修改：

1. 导入 Step 1 中实现的映射函数 `alpaca_map_fn`
2. 用 `alpaca_map_fn` 替换 `train_dataset` 中的 `dataset_map_fn`
3. （可选）通过 `prompt_template = PROMPT_TEMPLATE.alpaca` 来设置 `alpaca` 数据集对应的对话模板。
4. 调整原始数据集的路径，关于 `load_dataset` 的相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
+ from mmengine.config import read_base
+ with read_base():
+     from .map_fn import alpaca_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- alpaca_en_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/data'

+ prompt_template = PROMPT_TEMPLATE.alpaca
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
+   dataset_map_fn=alpaca_map_fn,
+   template_map_fn=dict(
+       type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
...
```

#### Step 6, 打印数据集（可选）

在修改配置文件后，可以打印处理后数据集的第一条数据，以验证数据集是否正确构建。

```bash
xtuner log-dataset $CONFIG
```

其中 `$CONFIG` 是 Step 5 修改过的 config 的文件路径。

## 使用自定义数据集

### 使用 Alpaca 格式自定义数据集

若自定义数据集的数据格式满足`alpaca`格式，可以参考以下步骤进行 SFT 训练。

#### Step 1，列出候选模型名字

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称（如`baichuan`、`llama`）。

#### Step 2, 复制 config 文件

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

由于自定义数据集满足 Alpaca 格式，因此`CONFIG_NAME`应该从 Step 1 列出的候选模型名字中选择与 Alpaca 相关的。例如通过下列命令将名为 `internlm_7b_qlora_alpaca_e3` 的 config 导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

#### Step 3, 设置对话模板（可选）

参考[设置对话模板](#step-4-设置对话模板可选)

#### Step 4, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- alpaca_en_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/json/data'

prompt_template = PROMPT_TEMPLATE.alpaca
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(
+       type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=alpaca_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
...
```

### 使用其他格式自定义数据集

#### Step 1, 数据集准备

按照 XTuner 定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)准备自定义数据：

```json
[{
    "conversation":[
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

#### Step 2, 列出候选模型名字

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称。

#### Step 3, 复制 config 文件

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

#### Step 4, 设置对话模板（可选）

参考[设置对话模板](#step-4-设置对话模板可选)

#### Step 5, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

1. 调整原始数据集的路径
2. 由于数据集格式已经是标准格式了，需要将 `train_dataset` 中的 `dataset_map_fn` 置为 None
3. 设置对话模板

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- alpaca_en_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/json/data'

+ prompt_template = PROMPT_TEMPLATE.alpaca
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(
+       type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
+   dataset_map_fn=None,
+   template_map_fn=dict(
+       type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
...
```

#### Step 6, 检查数据集（可选）

在修改配置文件后，可以运行`xtuner/tools/check_custom_dataset.py`脚本验证数据集是否正确构建。

```bash
xtuner check-custom-dataset $CONFIG
```

其中 `$CONFIG` 是 Step 5 修改过的 config 的文件路径。
