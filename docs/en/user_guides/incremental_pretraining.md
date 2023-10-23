# Incremental Pre-training Data Pipeline

- [Using Dataset in HuggingFace Hub](#using-dataset-in-huggingface-hub)
- [Using Custom Datasets](#using-custom-datasets)

Incremental pre-training aims to enhance the model's capability in a specific domain or task.

XTuner supports using HuggingFace Hub datasets or custom datasets for SFT (Supervised FineTune). The main difference between them is that when using HuggingFace Hub datasets, it is necessary to map the original data to the [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format)defined by XTuner. For custom datasets, users are recommended to construct the dataset according to the [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format).

## Using Dataset in HuggingFace Hub

### Step 1, Map Original Dataset to Standard Format

Since different datasets have different formats, it is necessary to map the original data to the [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format) defined by XTuner. XTuner supports the implementation of format mapping through the map function. The following uses the [oasst1 dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) as an example to explain how to implement data mapping.

The format of the oasst1 dataset is shown below:

```python
>>> from datasets import load_dataset

>>> ds = load_dataset(path='timdettmers/openassistant-guanaco')
>>> ds['train']
Dataset({
    features: ['text'],
    num_rows: 9846
})
```

As you can see, the oasst1 train dataset has 9846 rows, 1 column, the column name is 'text'. This 'text' column is the text data needed for incremental pre-training. The [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format) describes that during the process of incremental pre-training, the data format should be:

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

Therefore, you can map the original data to the standard format using the following map function:

```python
# Suppose the function is stored in ./map_fn.py
def custom_map_fn(example):
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

### Step 2, List Candidate Model Names

XTuner provides several ready-to-use configuration files. Users can view them with the following command:

```bash
xtuner list-cfg -p internlm
```

`-p` is used for fuzzy search. If you want to train other models, you can replace internlm with other model names supported by XTuner.

### Step 3, Export the Config File

If the provided configuration file does not meet your needs, please export the provided configuration file and make corresponding changes:

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

For example, you can export the config named \`internlm_7b_qlora_oasst1_e3\`\` to the current directory using the following command:

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, Modify the Config File

The following modifications need to be made to the config file copied in Step 3:

1. Import the mapping function `oasst1_incremental_map_fn` implemented in Step 1.
2. Replace the `dataset_map_fn` in `train_dataset` with `custom_map_fn`.
3. Set the `template_map_fn` in `train_dataset` to \`None\`\` (because there is no need to add the dialogue template to the incremental pre-training dataset).
4. Adjust the path of the original dataset. For operations related to `load_dataset`, refer to the [user document](https://huggingface.co/docs/datasets/loading).
5. Close the `EvaluateChatHook`, since the model only has a continuation function during incremental pre-training and doesn't have the conversation function.

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from mmengine.config import read_base
+ with read_base():
+     from .map_fn import custom_map_fn
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
- prompt_template = PROMPT_TEMPLATE.internlm_chat
+ data_path = 'path/to/your/data'
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
-   template_map_fn=dict(
-       type=template_map_fn_factory, template=prompt_template),
+   template_map_fn=None,
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
...
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
-   dict(
-       type=EvaluateChatHook,
-       tokenizer=tokenizer,
-       every_n_iters=evaluation_freq,
-       evaluation_inputs=evaluation_inputs,
-       system=SYSTEM,
-       instruction=prompt_template.INSTRUCTION)
]
...
```

### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.

## Using Custom Datasets

When using custom datasets for incremental pre-training, we recommend constructing the dataset according to the [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format) defined by XTuner. If the custom dataset is in other formats such as oasst1, refer to the section on [Using Dataset in HuggingFace Hub](#using-dataset-in-huggingface-hub).

### Step 1, Data Preparation

Prepare custom data according to the [incremental pre-training data format](./dataset_format.md#incremental-pre-training-dataset-format) defined by XTuner:

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
    }
]
```

### Step 2, List Candidate Model Names

```bash
xtuner list-cfg -p internlm
```

The `-p` option is for fuzzy search. If you want to train other models, you can replace internlm with the name of any other model supported by XTuner.

### Step 3, Export the Config File

```bash
xtuner copy-cfg internlm_7b_qlora_oasst1_e3 .
```

### Step 4, Modify the config file

Modifications need to be made to the config file obtained in Step 3 as follows:

1. Adjust the path of the original dataset
2. Since the dataset format is already standardized, set `dataset_map_fn` in `train_dataset` to `None`
3. Set `template_map_fn` in `train_dataset` to `None`, because there is no need to add conversation templates to the incremental pre-training dataset
4. Close the `EvaluateChatHook`, since the model only has a continuation function during incremental pre-training and doesn't have the conversation function.

```diff
from xtuner.dataset import process_hf_dataset
from datasets import load_dataset
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
...
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
- data_path = 'timdettmers/openassistant-guanaco'
- prompt_template = PROMPT_TEMPLATE.internlm_chat
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
-   template_map_fn=dict(
-       type=template_map_fn_factory, template=prompt_template),
+   template_map_fn=None,
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
...
#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
-   dict(
-       type=EvaluateChatHook,
-       tokenizer=tokenizer,
-       every_n_iters=evaluation_freq,
-       evaluation_inputs=evaluation_inputs,
-       system=SYSTEM,
-       instruction=prompt_template.INSTRUCTION)
]
...
```

### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.
