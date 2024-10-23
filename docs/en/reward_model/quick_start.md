## Quick Start Guide for Reward Model

In this section, we will introduce how to use XTuner to train a 1.8B Reward Model, helping you get started quickly.

### Preparing Pretrained Model Weights

According to the paper [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), we use a language model fine-tuned with SFT as the initialization model for the Reward Model. Here, we use [InternLM2-chat-1.8b-sft](https://huggingface.co/internlm/internlm2-chat-1_8b-sft) as the initialization model.

Set `pretrained_model_name_or_path = 'internlm/internlm2-chat-1_8b-sft'` in the training configuration file, and the model files will be automatically downloaded when training starts. If you need to download the model weights manually, please refer to the section [Preparing Pretrained Model Weights](https://xtuner.readthedocs.io/zh-cn/latest/preparation/pretrained_model.html), which provides detailed instructions on how to download model weights from Huggingface or Modelscope. Here are the links to the models on HuggingFace and ModelScope:

- HuggingFace link: https://huggingface.co/internlm/internlm2-chat-1_8b-sft
- ModelScope link: https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft/summary

### Preparing Training Data

In this tutorial, we use the [UltraFeedback](https://arxiv.org/abs/2310.01377) dataset as an example. For convenience, we use the preprocessed [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) dataset from Huggingface.

```python
train_dataset = dict(
    type=build_preference_dataset,
    dataset=dict(
        type=load_dataset,
        path='argilla/ultrafeedback-binarized-preferences-cleaned'),
    dataset_map_fn=orpo_dpo_mix_40k_map_fn,
    is_dpo=False,
    is_reward=True,
)
```

Using the above configuration in the configuration file will automatically download and process this dataset. If you want to use other open-source datasets from Huggingface or custom datasets, please refer to the [Preference Dataset](./preference_data.md) section.

### Preparing Configuration Files

XTuner provides several ready-to-use configuration files, which can be viewed using `xtuner list-cfg`. Execute the following command to copy a configuration file to the current directory.

```bash
xtuner copy-cfg internlm2_chat_1_8b_reward_full_ultrafeedback .
```

Open the copied configuration file. If you choose to download the model and dataset automatically, no modifications are needed. If you want to specify paths to your pre-downloaded model and dataset, modify the `pretrained_model_name_or_path` and the `path` parameter in `dataset` under `train_dataset`.

For more training parameter configurations, please refer to the section [Modifying Reward Training Configuration](./modify_settings.md).

### Starting the Training

After completing the above steps, you can start the training task using the following commands.

```bash
# Single node single GPU
xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py
# Single node multiple GPUs
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py
# Slurm cluster
srun ${SRUN_ARGS} xtuner train ./internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py --launcher slurm
```

The correct training log should look like the following (running on a single A800 GPU):

```
06/06 16:12:11 - mmengine - INFO - Iter(train) [   10/15230]  lr: 3.9580e-07  eta: 2:59:41  time: 0.7084  data_time: 0.0044  memory: 18021  loss: 0.6270  acc: 0.0000  chosen_score_mean: 0.0000  rejected_score_mean: 0.0000  num_samples: 4.0000  num_tokens: 969.0000
06/06 16:12:17 - mmengine - INFO - Iter(train) [   20/15230]  lr: 8.3536e-07  eta: 2:45:25  time: 0.5968  data_time: 0.0034  memory: 42180  loss: 0.6270  acc: 0.5000  chosen_score_mean: 0.0013  rejected_score_mean: 0.0010  num_samples: 4.0000  num_tokens: 1405.0000
06/06 16:12:22 - mmengine - INFO - Iter(train) [   30/15230]  lr: 1.2749e-06  eta: 2:37:18  time: 0.5578  data_time: 0.0024  memory: 32121  loss: 0.6270  acc: 0.7500  chosen_score_mean: 0.0016  rejected_score_mean: 0.0011  num_samples: 4.0000  num_tokens: 932.0000
06/06 16:12:28 - mmengine - INFO - Iter(train) [   40/15230]  lr: 1.7145e-06  eta: 2:36:05  time: 0.6033  data_time: 0.0025  memory: 42186  loss: 0.6270  acc: 0.7500  chosen_score_mean: 0.0027  rejected_score_mean: 0.0016  num_samples: 4.0000  num_tokens: 994.0000
06/06 16:12:35 - mmengine - INFO - Iter(train) [   50/15230]  lr: 2.1540e-06  eta: 2:41:03  time: 0.7166  data_time: 0.0027  memory: 42186  loss: 0.6278  acc: 0.5000  chosen_score_mean: 0.0031  rejected_score_mean: 0.0032  num_samples: 4.0000  num_tokens: 2049.0000
06/06 16:12:40 - mmengine - INFO - Iter(train) [   60/15230]  lr: 2.5936e-06  eta: 2:33:37  time: 0.4627  data_time: 0.0023  memory: 30238  loss: 0.6262  acc: 1.0000  chosen_score_mean: 0.0057  rejected_score_mean: 0.0030  num_samples: 4.0000  num_tokens: 992.0000
06/06 16:12:46 - mmengine - INFO - Iter(train) [   70/15230]  lr: 3.0331e-06  eta: 2:33:18  time: 0.6018  data_time: 0.0025  memory: 42186  loss: 0.6247  acc: 0.7500  chosen_score_mean: 0.0117  rejected_score_mean: 0.0055  num_samples: 4.0000  num_tokens: 815.0000
```

### Model Conversion

XTuner provides integrated tools to convert models to HuggingFace format. Simply execute the following commands:

```bash
# Create a directory to store HF format parameters
mkdir work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy/iter_15230_hf

# Convert the format
xtuner convert pth_to_hf internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py \
                            work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py/iter_15230.pth \
                            work_dirs/internlm2_chat_1_8b_reward_full_ultrafeedback_copy.py/iter_15230_hf
```

This will convert the XTuner's ckpt to the HuggingFace format.

Note: Since the Reward Model type is not integrated into the official transformers library, only the Reward Models trained with InternLM2 will be converted to the `InternLM2ForRewardModel` type. Other models will default to the `SequenceClassification` type (for example, LLaMa3 will be converted to the `LlamaForSequenceClassification` type).
