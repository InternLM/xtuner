# 多轮对话 data pipeline

- [使用 HuggingFace Hub 数据集](#使用-huggingface-hub-数据集)
- [使用自定义数据集](#使用自定义数据集)

多轮对话指令微调旨在提升模型的多轮对话能力，在数据处理阶段需要将原始数据转换为XTuner支持的数据集格式。

XTuner 支持使用 HuggingFace Hub 数据集或自定义数据集进行 SFT（Supervised FineTune）。二者的主要区别在于，使用 HuggingFace Hub 数据集时需要将原始数据映射为 XTuner 定义的[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)，而对于自定义数据集则推荐用户按照[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)构造数据集。

## 使用 HuggingFace Hub 数据集

### Step 1, 映射原始数据集为标准格式

由于不同数据集的格式各有不同，因此需要将原始数据映射为 XTuner 定义的[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)。XTuner 支持通过 map function 来实现格式的映射。下面以 [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) 数据集为例介绍如何实现数据映射。

oasst1 数据集格式如下所示：

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='timdettmers/openassistant-guanaco')
>>> ds['train']
Dataset({
    features: ['text'],
    num_rows: 9846
})
>>> ds['train'][0]['text']
'### Human: xxx ### Assistant: xxx ###Human: xxx ###Assistant: xxx'
```

由此可见，oasst1 数据集既可以当做增量预训练数据集让模型学会一些基本的语言知识，又可以在经过一些处理后作为多轮对话数据集培养模型的多轮对话能力。[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)中介绍了多轮对话指令微调过程中，数据格式应该为：

```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
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
        },
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
SYSTEM_OASST1 = ''  # oasst1 并未使用 system 字段
def custom_map_fn(example):
    r"""
    Example before preprocessing:
        example['text'] = '### Human: Can you explain xxx'
                          '### Assistant: Sure! xxx'
                          '### Human: I didn't understand how xxx'
                          '### Assistant: It has to do with a process xxx.'

    Example after preprocessing:
        example['conversation'] = [
            {
                'input': 'Can you explain xxx',
                'output': 'Sure! xxx'
            },
            {
                'input': 'I didn't understand how xxx',
                'output': 'It has to do with a process xxx.'
            }
        ]
    """
    data = []
    for sentence in example['text'].strip().split('###'):
        sentence = sentence.strip()
        if sentence[:6] == 'Human:':
            data.append(sentence[6:].strip())
        elif sentence[:10] == 'Assistant:':
            data.append(sentence[10:].strip())
    if len(data) % 2:
        # The last round of conversation solely consists of input
        # without any output.
        # Discard the input part of the last round, as this part is ignored in
        # the loss calculation.
        data.pop()
    conversation = []
    for i in range(0, len(data), 2):
        system = SYSTEM_OASST1 if i == 0 else ''
        single_turn_conversation = {
            'system': system,
            'input': data[i],
            'output': data[i + 1]}
        conversation.append(single_turn_conversation)
    return {'conversation': conversation}
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

例如通过下列命令将名为 `internlm_7b_qlora_oasst1_e3` 的 config 导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

1. 导入 Step 1 中实现的映射函数 `custom_map_fn`
2. 用 `custom_map_fn` 替换 `train_dataset` 中的 `dataset_map_fn`
3. 调整原始数据集的路径，关于 `load_dataset` 的相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
+ from mmengine.config import read_base
+ with read_base():
+     from .map_fn import custom_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
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
-   dataset_map_fn=oasst1_map_fn,
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

在使用自定义多轮对话数据集进行指令微调时，我们推荐将数据集构造为 XTuner 定义的[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)。若自定义数据集格式为 `oasst1` 等其他格式，可参考[使用 HuggingFace Hub 数据集](#使用-huggingface-hub-数据集)一节。

### Step 1, 数据集准备

按照 XTuner 定义的[多轮对话数据格式](./dataset_format.md#多轮对话数据集格式)准备自定义数据：

```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
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
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

### Step 2, 列出候选模型名字

```bash
xtuner list-cfg -p internlm
```

`-p` 为模糊查找，若想训练其他模型，可以修改 `internlm` 为 XTuner 支持的其他模型名称。

### Step 3, 复制 config 文件

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, 修改 config 文件

对 Step 3 复制得到的 config 文件需要进行如下修改：

1. 调整原始数据集的路径
2. 由于数据集格式已经是标准格式了，需要将 `train_dataset` 中的 `dataset_map_fn` 置为 `None`

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
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
-   dataset_map_fn=oasst1_map_fn,
+   dataset_map_fn=None,
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
