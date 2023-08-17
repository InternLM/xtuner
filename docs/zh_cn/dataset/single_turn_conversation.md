# 单轮对话data pipeline

单轮对话指令微调旨在提升模型回复特定指令的能力，其数据处理流程可以分为：

1. 按照相应数据集格式构造数据
2. 向数据集中插入对话模板（可选）

## 数据集构建

xTuner已经支持使用以下数据集进行训练：

- [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)：增量预训练数据集，多轮对话指令微调数据集
- [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)：单轮对话指令微调数据集
- [alpaca_zh](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese)：单轮对话指令微调数据集
- [openorca](https://huggingface.co/datasets/Open-Orca/OpenOrca)：单轮对话指令微调数据集
- [arxiv](https://kaggle.com/datasets/Cornell-University/arxiv)：单轮对话指令微调数据集，数据集中包含arxiv文章摘要与对应标题
- [cmd](https://github.com/Toyhom/Chinese-medical-dialogue-data/raw/master/Data_数据/)：单轮对话指令微调数据集，数据集中包含医疗相关数据
- [moss](https://huggingface.co/datasets/fnlp/moss-003-sft-data)：工具使用数据集

若要使用其他已有数据集或自定义数据集进行SFT（Supervised FineTune），可以参考下面的文档。

### 使用其他已有数据集

#### Step1 列出候选模型名字

```bash
xtuner list-cfg -p internlm
```

`-p`为模糊查找，若想训练其他模型，可以修改`internlm`为xtuner支持的其他模型名称。

#### Step2 复制config文件

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 xtuner/configs/internlm/internlm_7b/
```

#### Step 3 修改config文件

step2复制得到的config文件如下所示：

```python
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
################ Modification 1 ###########
from xtuner.datasets.map_fns import oasst1_map_fn
############################################
...
#######################################################################
#                          STEP 1  Settings                           #
#######################################################################
...
#######################################################################
#                      STEP 2  Model & Tokenizer                      #
#######################################################################
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    ################ Modification 2 ###########
    dataset=dict(type=load_dataset, path=data_path),
    ############################################
    tokenizer=tokenizer,
    max_length=max_length,
    ################ Modification 3 ###########
    map_fn=oasst1_map_fn,
    ############################################
    pack_to_max_length=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
#######################################################################
#                            STEP 4  Scheduler                        #
#######################################################################
...
#######################################################################
#                           STEP 5  Runtime                           #
#######################################################################
...
```

需要进行以下三点修改：

- 实现map_fn将原始数据集映射为xtuner标准数据集格式，并**在config中import进来**（对应Modification 1）
- 用import进来的map_fn替换掉`train_dataset`中的map_fn（对应Modification 3）
- 修改原始数据集路径（对应Modification 2），load_dataset相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

下面介绍如何实现数据集对应的map_fn。

由于不同数据集的格式各有不同，因此需要将原始数据映射为xTuner定义的[单轮对话数据格式](./dataset_format.md##单轮对话数据集格式)。xTuner支持通过map function来实现格式的映射。下面以[alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)数据集为例介绍如何实现数据映射。

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

由此可见，oasst1 train dataset有52002行，4列，列名分别为'instruction', 'input', 'output', 'text'。'instruction'和'input'给出了问题描述，'output'为对应groundtruth回答。[单轮对话数据格式](./dataset_format.md##单轮对话数据集格式)中介绍了单轮对话指令微调过程中，数据格式应该为：

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
    // ...
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

### 使用自定义数据集

#### Step1 数据集准备

按照xTuner定义的[单轮对话数据格式](./dataset_format.md##单轮对话数据集格式)准备自定义数据：

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
    // ...
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

修改step2复制得到的config文件中的原始数据集路径（对应Modification 1）即可：

```python
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
...
#######################################################################
#                          STEP 1  Settings                           #
#######################################################################
...
#######################################################################
#                      STEP 2  Model & Tokenizer                      #
#######################################################################
...
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    ################ Modification 1 ###########
    dataset=dict(type=load_dataset, path=data_path),
    ############################################
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
#######################################################################
#                            STEP 4  Scheduler                        #
#######################################################################
...
#######################################################################
#                           STEP 5  Runtime                           #
#######################################################################
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
