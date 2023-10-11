# Single-turn Dialogue Data Pipeline

- [Using Dataset in HuggingFace Hub](#using-dataset-in-huggingface-hub)
- [Using Custom Datasets](#using-custom-datasets)
  - [Using Alpaca Format Custom Datasets](#using-alpaca-format-custom-datasets)
  - [Using Other Format Custom Datasets](#using-other-format-custom-datasets)

Single-turn dialogue instruction fine-tuning aims to enhance the model's ability to respond to specific instructions.

XTuner offers support for utilizing HuggingFace Hub datasets, Alpaca-Format custom datasets, or other format custom datasets for SFT (Supervised FineTune). The main differences between these options are as follows:

1. When using the HuggingFace Hub dataset for SFT, it is necessary to map the original data to the XTuner-defined [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format)
2. When utilizing Alpaca-Format custom datasets for SFT, it is crucial to ensure that the custom dataset includes a minimum of three columns: 'instruction', 'input', and 'output'.
3. When working with other custom datasets for SFT, it is recommended that users construct the dataset according to the single-turn dialogue data format. This is highly beneficial as it significantly reduces the time required for data preprocessing.

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

Therefore, the original data can be mapped to a standard format using the following map function:

```python
# Suppose the function is stored in ./map_fn.py
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

### Step 4, Modify Config Files

The config file copied in Step 3 needs to be modified as follows:

1. Import the map function `custom_map_fn` implemented in Step 1.
2. Replace `dataset_map_fn` in `train_dataset` with `custom_map_fn`.
3. Adjust the path of the original dataset. You can refer to the [user documentation](https://huggingface.co/docs/datasets/loading) for operations related to `load_dataset`.

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

### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.

## Using Custom Datasets

### Using Alpaca Format Custom Datasets

If the data format of the custom dataset meets the 'alpaca' format, you can refer to the following steps for SFT training.

#### Step 1, List Candidate Model Names

```bash
xtuner list-cfg -p internlm
```

`-p` is for fuzzy search. If you want to train other models, you can replace `internlm` with other model names supported by XTuner.

#### Step 2, Export the Config File

```bash
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_DIR}
```

As the custom dataset follows the Alpaca format, 'CONFIG_NAME' should select the ALPACA-related candidate model names listed in Step 1. For example, execute the following command to export the 'internlm_7b_qlora_alpaca_e3' config to the current directory:

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

#### Step 3, Modify Config File

The config copied in Step 2 needs to be modified as follows:

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

### Using Other Format Custom Datasets

#### Step 1, Dataset Preparation

Prepare your custom data according to the [single-turn dialogue data format](./dataset_format.md#single-turn-dialogue-dataset-format) defined by XTuner:

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

#### Step 2, List Candidate Model Names

```bash
xtuner list-cfg -p internlm
```

`-p` is for fuzzy search. If you want to train other models, you can replace `internlm` with other model names supported by XTuner.

#### Step 3, Export the Config File

```bash
xtuner copy-cfg internlm_7b_qlora_alpaca_e3 .
```

#### Step 4, Modify Config File

The config file copied in Step 3 needs to be modified as follows:

1. Adjust the path of the original dataset
2. Since the dataset format is already in the standard format, set `dataset_map_fn` in `train_dataset` to `None`

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

#### Step 5, Check custom Dataset (Optional)

After modifying the config file, you can execute the 'xtuner/tools/check_custom_dataset.py' script to verify the correct construction of the dataset.

```bash
xtuner check-custom-dataset $CONFIG
```

`$CONFIG` represents the file path of the modified configuration file in Step 4.
