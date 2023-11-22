**Note: The primary aim of this document is to provide detailed instructions on how to train models based on the data format provided by the InternLM repository, rather than to train the InternLM model itself.**

## Dataset Format

The training dataset of [InternLM](https://github.com/InternLM/InternLM) is pre-tokenized, and is formatted as follows:

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

Among them, tokens with negative values are not involved in the calculation of loss during the training process.

## Interface Introduction

To train with InternLM format data in XTuner, you need to convert the original data format into the XTuner standard dataset format. The core function for processing the InternLM format dataset is as follows:

```python
def process(dataset_folder,
            split='train',
            pack_to_max_length=False,
            max_length=2048,
            shuffle_before_pack=True,
            map_num_proc=32):
    ......
```

Where:

1. `dataset_folder`: Indicates the path where the training dataset is located. All files ending with `.bin` in the path will be used as training data.
2. `split`: The dataset read through hf datasets is usually a DatasetDict. The corresponding Dataset is obtained through the split variable. The default value 'train' can generally be used.
3. `pack_to_max_length`: Whether to concatenate multiple pieces of data into one piece of data for training.
4. `max_length`: Indicates that the data processing process will pack multiple training data into a piece of data with a length of max_length. Only effective when pack_to_max_length=True.
5. `shuffle_before_pack`: Whether to shuffle the dataset before packing, generally the default True can be used. Only effective when pack_to_max_length=True.
6. `map_num_proc`: Use multi-process for data processing. Depending on the situation, the value of map_num_proc can be increased.

## Tutorial

### Step 1, Export the Template Config File

you can export the config named \`internlm_7b_full_intern_repo_dataset_template\`\` to the current directory using the following command:

```bash
xtuner copy-cfg internlm_7b_full_intern_repo_dataset_template .
```

### Step 2, Modify the Template Config File

You only need to modify the corresponding part of the above interface in the Config file.

```diff
...

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'internlm/internlm-7b'

# Data
- dataset_folder = '/path/to/your/dataset'
+ dataset_folder = '/real/dataset/path'
max_length = 2048
pack_to_max_length = True
...
```

### Step 3, Start training

```
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template --launcher slurm --deepspeed deepspeed_zero3
```
