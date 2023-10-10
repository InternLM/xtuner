# Multi-turn Dialogue Data Pipeline

- [Using Dataset in HuggingFace Hub](#using-dataset-in-huggingface-hub)
- [Using Custom Datasets](#using-custom-datasets)

The purpose of multi-turn dialogue command fine-tuning is to enhance the model's ability for multi-turn dialogues.

XTuner supports the use of HuggingFace Hub datasets or custom datasets for SFT (Supervised FineTune). The main difference between them is that when using the HuggingFace Hub dataset, the original data needs to be mapped to the [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format) defined by XTuner. For custom datasets, it is recommended that users construct the dataset according to the [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format).

## Using Dataset in HuggingFace Hub

### Step 1, Map Original Dataset to Standard Format

Since the formats of different datasets vary, the original data needs to be transformed into the [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format) defined by XTuner. XTuner supports the use of a map function to achieve format mapping. The following example uses the [oasst1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) to illustrate how to implement data mapping.

The oasst1 dataset format is as follows:

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

It's clear that the oasst1 dataset can not only be used as an incremental pre-training dataset for the model to learn some basic language knowledge, but also, after some processing, serve as a multi-turn dialogue dataset to cultivate the model's multi-turn conversation capabilities. The [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format) introduces that in the fine-tuning process of multi-turn dialogue instructions, the data format should be:

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

Therefore, the original data can be mapped to a standard format using the following map function:

```python
# Suppose the function is stored in ./map_fn.py
SYSTEM_OASST1 = ''  # oasst1 does not set the system text
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

### Step 2, List Candidate Model Names

XTuner provides several ready-to-use configuration files. Users can view them using the following command:

```bash
xtuner list-cfg -p internlm
```

`-p` is used for fuzzy search. If you want to train other models, you can replace `internlm` with other model names supported by XTuner.

### Step 3, Export the Config File

If the provided configuration file does not meet your needs, please export the offered configuration file and make appropriate changes:

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

For example, use the following command to export the config named `internlm_7b_qlora_oasst1_e3` to the current directory:

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, Modify Config Files

The config file copied in Step 3 needs to be modified as follows:

1. Import the map function `custom_map_fn` implemented in Step 1.
2. Replace `dataset_map_fn` in `train_dataset` with `custom_map_fn`.
3. Adjust the path of the original dataset. You can refer to the [user documentation](https://huggingface.co/docs/datasets/loading) for operations related to `load_dataset`.

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

### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.

## Using Custom Datasets

When using a custom multi-turn dialogue dataset for command fine-tuning, we recommend constructing the dataset in the [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format) as defined by XTuner. If the custom dataset format is oasst1 or other formats, you can refer to the section on [Using Datasets in HuggingFace Hub](#using-dataset-in-huggingface-hub).

### Step 1, Dataset Preparation

Prepare your custom data according to the [multi-turn dialogue data format](./dataset_format.md#multi-turn-dialogue-dataset-format) defined by XTuner:

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

### Step 2, List Candidate Model Names

```bash
xtuner list-cfg -p internlm
```

`-p` is for fuzzy search. If you want to train other models, you can replace `internlm` with other model names supported by XTuner.

### Step 3, Export the Config File

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, Modify Config File

The config file copied in Step 3 needs to be modified as follows:

1. Adjust the path of the original dataset
2. Since the dataset format is already in the standard format, set `dataset_map_fn` in `train_dataset` to `None`

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

### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.
