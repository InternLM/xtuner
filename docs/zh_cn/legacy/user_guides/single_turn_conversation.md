# 单轮对话 data pipeline

- [使用 HuggingFace Hub 数据集](#使用-huggingface-hub-数据集)
- [使用自定义数据集](#使用自定义数据集)
  - [使用 Alpaca 格式的自定义数据集](#使用-alpaca-格式的自定义数据集)
  - [使用其他格式自定义数据集](#使用其他格式自定义数据集)

单轮对话指令微调旨在提升模型回复特定指令的能力，在数据处理阶段需要将原始数据转换为XTuner支持的数据集格式。

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
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

因此，可以通过下面的 map function 将原始数据映射为标准格式：

```python
# 假设将该函数存放在./map_fn.py文件中
SYSTEM_ALPACA = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.\n')
def custom_map_fn(example):
    if example.get('output') == '<nooutput>':
        return {'conversation': []}
    else:
        return {
            'conversation': [{
                'system': SYSTEM_ALPACA,
                'input': f"{example['instruction']}\n{example['input']}",
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

### Step 4, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

1. 导入 Step 1 中实现的映射函数 `custom_map_fn`
2. 用 `custom_map_fn` 替换 `train_dataset` 中的 `dataset_map_fn`
3. 调整原始数据集的路径，关于 `load_dataset` 的相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
+ from mmengine.config import read_base
+ with read_base():
+     from .map_fn import custom_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/data'
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=custom_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
...
```

### Step 5, 检查数据集（可选）

在修改配置文件后，可以运行`xtuner/tools/check_custom_dataset.py`脚本验证数据集是否正确构建。

```bash
xtuner check-custom-dataset $CONFIG
```

其中 `$CONFIG` 是 Step 4 修改过的 config 的文件路径。

## 使用自定义数据集

### 使用 Alpaca 格式的自定义数据集

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

#### Step 3, 修改 config 文件

对 Step 2 复制得到的 config 文件需要进行如下修改：

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/json/data'
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
...
```

### 使用其他格式自定义数据集

#### Step 1, 数据集准备

按照 XTuner 定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)准备自定义数据：

```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
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

#### Step 4, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

1. 调整原始数据集的路径
2. 由于数据集格式已经是标准格式了，需要将 `train_dataset` 中的 `dataset_map_fn` 置为 `None`

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/json/data'
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
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
...
```

#### Step 5, 检查数据集（可选）

在修改配置文件后，可以运行`xtuner/tools/check_custom_dataset.py`脚本验证数据集是否正确构建。

```bash
xtuner check-custom-dataset $CONFIG
```

其中 `$CONFIG` 是 Step 4 修改过的 config 的文件路径。
