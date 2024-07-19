## Quick Start with DPO

In this section, we will introduce how to use XTuner to train a 1.8B DPO (Direct Preference Optimization) model to help you get started quickly.

### Preparing Pretrained Model Weights

We use the model [InternLM2-chat-1.8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft), as the initial model for DPO training to align human preferences.

Set `pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'` in the training configuration file, and the model files will be automatically downloaded when training starts. If you need to download the model weights manually, please refer to the section [Preparing Pretrained Model Weights](https://xtuner.readthedocs.io/zh-cn/latest/preparation/pretrained_model.html), which provides detailed instructions on how to download model weights from Huggingface or Modelscope. Here are the links to the models on HuggingFace and ModelScope:

- HuggingFace link: https://huggingface.co/internlm/internlm2-chat-1_8b-sft
- ModelScope link: https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary

### Preparing Training Data

In this tutorial, we use the [mlabonne/orpo-dpo-mix-40k](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) dataset from Huggingface as an example.

```python
train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(
        type=load_dataset,
        path='mlabonne/orpo-dpo-mix-40k'),
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=True,
    is_reward=False,
)
```

Using the above configuration in the configuration file will automatically download and process this dataset. If you want to use other open-source datasets from Huggingface or custom datasets, please refer to the [Preference Dataset](../reward_model/preference_data.md) section.

### Preparing Configuration File

XTuner provides several ready-to-use configuration files, which can be viewed using `xtuner list-cfg`. Execute the following command to copy a configuration file to the current directory.

```bash
xtuner copy-cfg internlm2_chat_1_8b_dpo_full .
```

Open the copied configuration file. If you choose to download the model and dataset automatically, no modifications are needed. If you want to specify paths to your pre-downloaded model and dataset, modify the `pretrained_model_name_or_path` and the `path` parameter in `dataset` under `train_dataset`.

For more training parameter configurations, please refer to the section [Modifying DPO Training Configuration](./modify_settings.md) section.

### Starting the Training

After completing the above steps, you can start the training task using the following commands.

```bash
# Single machine, single GPU
xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py
# Single machine, multiple GPUs
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py
# Slurm cluster
srun ${SRUN_ARGS} xtuner train ./internlm2_chat_1_8b_dpo_full_copy.py --launcher slurm
```

### Model Conversion

XTuner provides integrated tools to convert models to HuggingFace format. Simply execute the following commands:

```bash
# Create a directory for HuggingFace format parameters
mkdir work_dirs/internlm2_chat_1_8b_dpo_full_copy/iter_15230_hf

# Convert format
xtuner convert pth_to_hf internlm2_chat_1_8b_dpo_full_copy.py \
                            work_dirs/internlm2_chat_1_8b_dpo_full_copy/iter_15230.pth \
                            work_dirs/internlm2_chat_1_8b_dpo_full_copy/iter_15230_hf
```

This will convert the XTuner's ckpt to the HuggingFace format.
