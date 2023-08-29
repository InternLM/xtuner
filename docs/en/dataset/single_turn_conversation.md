# Single-turn Dialogue Data Pipeline

Single-turn dialogue instruction fine-tuning aims to enhance the model's ability to respond to specific instructions. Its data processing flow can be divided into the following two parts:

1. Construct data according to the corresponding dataset format
2. Insert dialogue templates into the dataset (optional)

XTuner supports using HuggingFace Hub datasets or custom datasets for SFT (Supervised FineTune). The main difference between them is that when using the HuggingFace Hub dataset, the original data needs to be mapped to the XTuner-defined [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format), whereas for custom datasets, it is recommended that users construct the dataset according to the single-turn dialogue data format.

## Using Dataset in HuggingFace Hub

### Step 1, Map the Original Dataset to Standard Format

Since different datasets have different formats, it is necessary to map the original data to the XTuner-defined [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format). XTuner supports mapping of formats through a map function. Below we will use the [alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) as an example to show how to implement data mapping.

The alpaca dataset format is shown below:

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='tatsu-lab/alpaca')
>>> ds['train']
Dataset({
    features: ['instruction', 'input', 'output', 'text'],
    num_rows: 52002
})
```

The "Alpaca Train" dataset comprises 52,002 records, organized into four distinct columns denoted as 'instruction', 'input', 'output', and 'text'. In this dataset, 'instruction' and 'input' columns provide detailed descriptions of the presented problem, while the 'output' column contains the corresponding GroundTruth responses. This dataset adheres to the [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format) that was introduced during the process of fine-tuning using single round session instructions. The prescribed data format for this context is as follows:

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

Therefore, the original data can be mapped to a standard format using the following map function:

```python
# Suppose the function is stored in ./map_fn.py
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

For example, use the following command to export the config named `internlm_7b_qlora_alpaca_e3` to the current directory:

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

### Step 4, Set Conversation Templates (Optional)

Conversation templates refer to predefined patterns or structures used for generating dialogues. These templates may include questions, answers, or different roles' speeches in multi-turn dialogues. Adding conversation templates to the training dataset helps the model generate structured and logical dialogues and provide more accurate, consistent, and reasonable responses.

Different datasets and language models may correspond to different conversation templates. For instance, the conversation template of the [alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) is as follows:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
xxx

### Assistant:
xxx
```

XTuner provides a series of conversation templates, which you can find in `xtuner/utils/templates.py`. Among them, `INSTRUCTION_START` and `INSTRUCTION` represent the conversation templates used for the first round dialogue and subsequent rounds of dialogues, respectively. Only `INSTRUCTION_START` is used in a single-turn conversation dataset such as `alpaca`.

### Step 5, Modify Config Files

The config file copied in Step 3 needs to be modified as follows:

1. Import the map function `alpaca_map_fn` implemented in Step 1.
2. Replace `dataset_map_fn` in `train_dataset` with `alpaca_map_fn`.
3. (Optional) Set the conversation template corresponding to the `alpaca` dataset via `prompt_template = PROMPT_TEMPLATE.alpaca`.
4. Adjust the path of the original dataset. You can refer to the [user documentation](https://huggingface.co/docs/datasets/loading) for operations related to `load_dataset`.

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
+ from .map_fn import alpaca_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- alpaca_zh_path = 'silk-road/alpaca-data-gpt4-chinese'
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

## Using Custom Datasets

When using a custom single-turn dialogue dataset for command fine-tuning, we recommend constructing the dataset in the [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format) as defined by XTuner. If the custom dataset format is oasst1 or other formats, you can refer to the section on [Using Datasets in HuggingFace Hub](#using-dataset-in-huggingface-hub).

### Step 1, Dataset Preparation

Prepare your custom data according to the [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format) defined by XTuner:

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

### Step 2, List Candidate Model Names

```bash
xtuner list-cfg -p internlm
```

`-p` is for fuzzy search. If you want to train other models, you can replace `internlm` with other model names supported by XTuner.

### Step 3, Export the Config File

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

### Step 4, Setting Dialogue Template (Optional)

Refer to [Setting the Dialogue Template](#step-4-set-conversation-templates-optional).

### Step 5, Modify Config File

The config file copied in Step 3 needs to be modified as follows:

1. Adjust the path of the original dataset
2. Since the dataset format is already in the standard format, set `dataset_map_fn` in `train_dataset` to None
3. Set the dialogue template

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- alpaca_zh_path = 'silk-road/alpaca-data-gpt4-chinese'
- alpaca_en_path = 'tatsu-lab/alpaca'
+ data_path = 'path/to/your/data'

+ prompt_template = PROMPT_TEMPLATE.alpaca
#######################################################################
#                      STEP 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=data_path),
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
