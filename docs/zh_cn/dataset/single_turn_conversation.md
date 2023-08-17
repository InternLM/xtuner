# 单轮对话data pipeline

单轮对话指令微调旨在提升模型回复特定指令的能力，其数据处理流程可以分为以下两部分：

1. 按照相应数据集格式构造数据
2. 向数据集中插入对话模板（可选）

## 数据集构建

xTuner支持使用HuggingFace Hub数据集或自定义数据集进行SFT（Supervised FineTune）。二者的主要区别在于，使用HuggingFace Hub数据集时需要将原始数据映射为xTuner定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)，而对于自定义数据集则需要用户按照[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)构造数据集。

### 使用HuggingFace Hub数据集

#### Step 1 映射原始数据集为标准格式

由于不同数据集的格式各有不同，因此需要将原始数据映射为xTuner定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)。xTuner支持通过map function来实现格式的映射。下面以[alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)数据集为例介绍如何实现数据映射。

oasst1数据集格式如下所示：

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='tatsu-lab/alpaca')
>>> ds['train']
Dataset({
    features: ['instruction', 'input', 'output', 'text'],
    num_rows: 52002
})
```

由此可见，oasst1 train dataset有52002行，4列，列名分别为'instruction', 'input', 'output', 'text'。'instruction'和'input'给出了问题描述，'output'为对应groundtruth回答。[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)中介绍了单轮对话指令微调过程中，数据格式应该为：

```json
[
    {
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
    },
]
```

因此，可以通过下面的map function将原始数据映射为标准格式：
<a id="alpaca_map_fn"></a>

```python
>>> def alpaca_map_fn(example):
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

#### Step 2 列出候选模型名字

```bash
xtuner list-cfg -p internlm
```

`-p`为模糊查找，若想训练其他模型，可以修改`internlm`为xtuner支持的其他模型名称。

#### Step 3 复制config文件

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 xtuner/configs/internlm/internlm_7b/
```

#### Step 4 修改config文件

对step 3复制得到的config文件需要进行如下修改：

1. import Step 1 中实现的map function `alpaca_map_fn`
2. 用`alpaca_map_fn`替换`train_dataset`中的map_fn
3. 修改原始数据集路径，load_dataset相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

```diff
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
+ from xtuner.datasets.map_fns import alpaca_map_fn
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='path/to/your/data'),
    tokenizer=tokenizer,
    max_length=max_length,
+   map_fn=alpaca_map_fn,
    pack_to_max_length=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
...
```

### 使用自定义数据集

#### Step 1 数据集准备

按照xTuner定义的[单轮对话数据格式](./dataset_format.md#单轮对话数据集格式)准备自定义数据：

```json
[
    {
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
    },
]
```

#### Step2 列出候选模型名字

```bash
xtuner list-cfg -p internlm
```

`-p`为模糊查找，若想训练其他模型，可以修改`internlm`为xtuner支持的其他模型名称。

#### Step3 复制config文件

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 xtuner/configs/internlm/internlm_7b/
```

#### Step4 修改config文件

修改step 3复制得到的config文件中的原始数据集路径即可：

```diff
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='path/to/your/data'),
    tokenizer=tokenizer,
    max_length=max_length,
    map_fn=None,
    pack_to_max_length=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
...
```

## 插入对话模板（可选）

对话模板是指用于生成对话的预定义模式或结构。这些模板可以包含问句、回答或多轮对话中的不同角色的发言。

不同数据集、不同语言模型可能对应着不同的对话模板。例如，[alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)数据集的对话模板如下：

```
Below is an instruction that describes a task. 'Write a response that appropriately completes the request.

### Instruction:
我想知道如何读取CSV文件。

### Assistant:
要读取CSV文件，您可以使用pandas库的read_csv()函数。
```

在训练数据集中加入对话模板有利于模型生成有结构和逻辑的对话，并提供更准确、一致和合理的回答。

将前文介绍的[alpaca_map_fn](#alpaca_map_fn)替换为下方代码即可：

```python
>>> def alpaca_map_fn(example):
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
        PROMPT = {
            'with_input':
            ('Below is an instruction that describes a task, paired with an '
            'input that provides further context. '
            'Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n'
            '### Response: '),
            'without_input':
            ('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{instruction}\n\n'
            '### Response: ')
        }
        if example.get('input', '') != '':
            prompt_template = PROMPT['with_input']
        else:
            prompt_template = PROMPT['without_input']

        if example.get('output', '') == '<nooutput>':
            return {'conversation': [{'input': '', 'output': ''}]}
        else:
            return {
                'conversation': [{
                    'input': prompt_template.format(**example),
                    'output': example['output']
                }]
            }
```
