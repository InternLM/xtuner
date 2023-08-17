# 单轮对话data pipeline

多轮对话指令微调旨在提升模型的多轮对话能力，其数据处理流程可以分为以下两部分：

1. 按照相应数据集格式构造数据
2. 向数据集中插入对话模板（可选）

## 数据集构建

xTuner支持使用HuggingFace Hub数据集或自定义数据集进行SFT（Supervised FineTune）。二者的主要区别在于，使用HuggingFace Hub数据集时需要将原始数据映射为xTuner定义的[多轮对话数据格式](./dataset_format.md##多轮对话数据集格式)，而对于自定义数据集则需要用户按照[多轮对话数据格式](./dataset_format.md##多轮对话数据集格式)构造数据集。

### 使用HuggingFace Hub数据集

#### Step 1 映射原始数据集为标准格式

由于不同数据集的格式各有不同，因此需要将原始数据映射为xTuner定义的[多轮对话数据格式](./dataset_format.md##多轮对话数据集格式)。xTuner支持通过map function来实现格式的映射。下面以[oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)数据集为例介绍如何实现数据映射。

oasst1数据集格式如下所示：

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

由此可见，oasst1数据集既可以当做增量预训练数据集让模型学会一些基本的语言知识，又可以在经过一些处理后作为多轮对话数据集培养模型的多轮对话能力。[多轮对话数据格式](./dataset_format.md##多轮对话数据集格式)中介绍了多轮对话指令微调过程中，数据格式应该为：

```json
[
    {
        "conversation":[
            {
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
                "input": "xxx",
                "output": "xxx"
            },
            {
                "input": "xxx",
                "output": "xxx"
            }
        ]
    },
]
```

因此，可以通过下面的map function将原始数据映射为标准格式：

```python
>>> def oasst1_map_fn(example):
        r"""
        >>> train_ds = ds['train'].map(oasst1_map_fn)
        >>> train_ds
        Dataset({
            features: ['text', 'conversation'],
            num_rows: 9846
        })
        >>> train_ds[0]['conversation']
        [{'input': '### Human: xxx', 'output': '### Assistant: xxx'}, {'input': '### Human: xxx', 'output': '### Assistant: xxx.'}]
        """
    data = [
        '### ' + sentence.strip()
        for sentence in example['text'].strip().split('###') if sentence != ''
    ]
    if len(data) % 2:
        # The last round of conversation solely consists of input
        # without any output.
        # Discard the input part of the last round, as this part is ignored in
        # the loss calculation.
        data.pop()
    conversation = []
    for i in range(0, len(data), 2):
        single_turn_conversation = {'input': data[i], 'output': data[i + 1]}
        conversation.append(single_turn_conversation)
    return {'conversation': conversation}
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

1. import Step 1 中实现的map function `oasst1_map_fn`
2. 用`oasst1_map_fn`替换`train_dataset`中的map_fn
3. 修改原始数据集路径，load_dataset相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)

```diff
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
+ from xtuner.datasets.map_fns import oasst1_map_fn
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
+   map_fn=oasst1_map_fn,
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

按照xTuner定义的[多轮对话数据格式](./dataset_format.md##多轮对话数据集格式)准备自定义数据：

```json
[
    {
        "conversation":[
            {
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
                "input": "xxx",
                "output": "xxx"
            },
            {
                "input": "xxx",
                "output": "xxx"
            }
        ]
    },
]
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

参考[插入对话模板文档](./single_turn_conversation.md##插入对话模板（可选）)
