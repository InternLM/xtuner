**Note: The primary aim of this document is to provide detailed instructions on how to train models based on the data format provided by the InternLM repository, rather than to train the InternLM model itself.**

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
use_local_attn = True

# Data
- dataset_folder = '/path/to/your/dataset'
+ dataset_folder = '/real/dataset/path'
max_length = 8192
pack_to_max_length = True
...
```

### Step 3, Start training

```
srun ${SRUN_ARGS} xtuner train internlm_7b_full_intern_repo_dataset_template_copy --launcher slurm --deepspeed deepspeed_zero1
```

## Dataset Format

The training dataset of [InternLM](https://github.com/InternLM/InternLM) is pre-tokenized, and is formatted as follows:

```
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
{"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
```

Among them, tokens with negative values are not involved in the calculation of loss during the training process.
