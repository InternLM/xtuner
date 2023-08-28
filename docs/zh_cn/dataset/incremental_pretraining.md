# 增量预训练data pipeline

增量预训练旨在提升模型在特定领域或任务的能力，其数据处理流程可以分为以下两部分：

1. 扩充tokenizer词表大小（可选）
2. 按照相应数据集格式构建数据集

## 扩充tokenizer词表（可选，正在开发中···）

💡 为了适应词表扩展所带来的模型权重维度变化，xTuner会根据字表大小自动调整语言模型的token embedding layer和lm head的参数维度。由于这个操作会引入一些随机初始化的参数，因此通常需要在更大的语料库上进行预训练以获得更好的效果。

## 数据集构建

xTuner支持使用HuggingFace Hub数据集或自定义数据集进行SFT（Supervised FineTune）。二者的主要区别在于，使用HuggingFace Hub数据集时需要将原始数据映射为xTuner定义的[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)。而对于自定义数据集则推荐用户按照[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)构造数据集。

### 使用HuggingFace Hub数据集

#### Step 1 映射原始数据集为标准格式

由于不同数据集的格式各有不同，因此需要将原始数据映射为xTuner定义的[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)。xTuner支持通过map function来实现格式的映射。下面以[oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)数据集为例介绍如何实现数据映射。

oasst1数据集格式如下所示：

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='timdettmers/openassistant-guanaco')
>>> ds['train']
Dataset({
    features: ['text'],
    num_rows: 9846
})
```

由此可见，oasst1 train dataset有9846行，1列，列名为'text'，'text'这一列正是增量预训练需要用到的文本数据。[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)中介绍了增量预训练过程中，数据格式应该为：

```json
[{
    "conversation":[
        {
            "input": "",
            "output": "xxx"
        },
    ]
}]
```

因此，可以通过下面的map function将原始数据映射为标准格式：

```python
# 假设将该函数存放在./map_fn.py文件中
def oasst1_incremental_map_fn(example):
    """
    >>> train_ds = ds['train'].map(oasst1_map_fn)
    >>> train_ds
    Dataset({
        features: ['text', 'conversation'],
        num_rows: 9846
    })
    >>> train_ds[0]['conversation']
    [{'input': '', 'output': 'xxx'}]
    """
    return {'conversation': [{'input': '', 'output': example['text']}]}

```

#### Step 2 列出候选模型名字

xTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```bash
xtuner list-cfg -p internlm
```

`-p`为模糊查找，若想训练其他模型，可以修改`internlm`为xtuner支持的其他模型名称。

#### Step 3 导出config文件

如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

例如通过下列命令将名为`internlm_7b_qlora_oasst1_e3`的config导出至当前目录下：

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

#### Step 4 修改config文件

对step 3复制得到的config文件需要进行如下修改：

1. import Step 1 中实现的映射函数 `oasst1_incremental_map_fn`
2. 用`oasst1_incremental_map_fn`替换`train_dataset`中的`dataset_map_fn`
3. 将`train_dataset`中的`template_map_fn`置为None，因为不需要将对话模板加入至增量预训练数据集中
4. 调整原始数据集的路径，关于`load_dataset`的相关操作可以参考[用户文档](https://huggingface.co/docs/datasets/loading)
5. （可选）如果你希望利用 xTuner 提供的`EvaluateChatHook`在训练的各个阶段记录模型的对话结果，你还需要通过`prompt_template = PROMPT_TEMPLATE.openassistant`将对话模板调整为与`oasst1`数据集对应的对话模板。xTuner提供了一系列对话模板，你可以在`xtuner/utils/templates.py`中找到。其中，`INSTRUCTION_START`和`INSTRUCTION`分别代表第一轮对话和后续若干轮对话所使用的对话模板。在`EvaluateChatHook`中只会用到`INSTRUCTION_START`。（注意：由于增量预训练后的模型只具备续写功能，不具备对话功能，因此在 `EvaluateChatHook`打印的对话结果中，模型无法正常停止生成是正常现象。）

```diff
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
- from xtuner.datasets.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.datasets.map_fns import template_map_fn_factory
+ from map_fn import oasst1_incremental_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'path/to/your/data'

+ prompt_template = PROMPT_TEMPLATE.openassistant
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=data_path),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=oasst1_map_fn,
+   dataset_map_fn=oasst1_incremental_map_fn,
-   template_map_fn=dict(
-       type=template_map_fn_factory, template=prompt_template),
+   template_map_fn=None,
    remove_unused_columns=True,
    shuffle_before_pack=True,
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

在使用自定义数据集进行增量预训练时，我们推荐将数据集构造为xTuner定义的[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)。若自定义数据集格式为`oasst1`等其他格式，可参考[使用HuggingFace Hub数据集](#使用huggingface-hub数据集)一节。

#### Step 1 数据准备

按照xTuner定义的[增量预训练数据格式](./dataset_format.md#增量预训练数据集格式)准备自定义数据：

```json
[
    {
        "conversation":[
            {
                "input": "",
                "output": "xxx"
            },
        ]
    },
    {
        "conversation":[
            {
                "input": "",
                "output": "xxx"
            },
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
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

#### Step 4 修改config文件

对step 3复制得到的config文件需要进行如下修改：

1. 调整原始数据集的路径
2. 由于数据集格式已经是标准格式了，需要将`train_dataset`中的`dataset_map_fn`置为None
3. 将`train_dataset`中的`template_map_fn`置为None，因为不需要将对话模板加入至增量预训练数据集中
4. （可选）设置对话模板以调用`EvaluateChatHook`在训练的各个阶段记录模型的对话结果

```diff
from xtuner.datasets import process_hf_dataset
from datasets import load_dataset
- from xtuner.datasets.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.datasets.map_fns import template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'path/to/your/data'

+ prompt_template = PROMPT_TEMPLATE.openassistant
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
+   dataset_map_fn=oasst1_incremental_map_fn,
-   template_map_fn=dict(
-       type=template_map_fn_factory, template=prompt_template),
+   template_map_fn=None,
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
