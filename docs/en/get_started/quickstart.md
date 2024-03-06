# Quickstart

In this section, we will show you how to use XTuner to fine-tune a model to help you get started quickly.

After installing XTuner successfully, we can start fine-tuning the model. In this section, we will demonstrate how to use XTuner to apply the QLoRA algorithm to fine-tune InternLM2-Chat-7B on the Colorist dataset.

The Colorist dataset ([HuggingFace link](https://huggingface.co/datasets/burkelibbey/colors); [ModelScope link](https://www.modelscope.cn/datasets/fanqiNO1/colors/summary)) is a dataset that provides color choices and suggestions based on color descriptions. A model fine-tuned on this dataset can be used to give a hexadecimal color code based on the user's description of the color. For example, when the user enters "a calming but fairly bright light sky blue, between sky blue and baby blue, with a hint of fluorescence due to its brightness", the model will output ![#66ccff](https://img.shields.io/badge/%2366ccff-66CCFF), which matches the user's description. There are a few sample data from this dataset:

| Enligsh Description                                                                                                                                                                                                              | Chinese Description                                                                                                              | Color                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Light Sky Blue: A calming, fairly bright color that falls between sky blue and baby blue, with a hint of slight fluorescence due to its brightness.                                                                              | 浅天蓝色：一种介于天蓝和婴儿蓝之间的平和、相当明亮的颜色，由于明亮而带有一丝轻微的荧光。                                         | #66ccff: ![#66ccff](https://img.shields.io/badge/%2366ccff-66CCFF) |
| Bright red: This is a very vibrant, saturated and vivid shade of red, resembling the color of ripe apples or fresh blood. It is as red as you can get on a standard RGB color palette, with no elements of either blue or green. | 鲜红色： 这是一种非常鲜艳、饱和、生动的红色，类似成熟苹果或新鲜血液的颜色。它是标准 RGB 调色板上的红色，不含任何蓝色或绿色元素。 | #ee0000: ![#ee0000](https://img.shields.io/badge/%23ee0000-EE0000) |
| Bright Turquoise: This color mixes the freshness of bright green with the tranquility of light blue, leading to a vibrant shade of turquoise. It is reminiscent of tropical waters.                                              | 明亮的绿松石色：这种颜色融合了鲜绿色的清新和淡蓝色的宁静，呈现出一种充满活力的绿松石色调。它让人联想到热带水域。                 | #00ffcc: ![#00ffcc](https://img.shields.io/badge/%2300ffcc-00FFCC) |

## Prepare the model weights

Before fine-tuning the model, we first need to prepare the weights of the model.

### Download from HuggingFace

```bash
pip install -U huggingface_hub

# Download the model weights to Shanghai_AI_Laboratory/internlm2-chat-7b
huggingface-cli download internlm/internlm2-chat-7b \
                            --local-dir Shanghai_AI_Laboratory/internlm2-chat-7b \
                            --local-dir-use-symlinks False \
                            --resume-download
```

### Download from ModelScope

Since pulling model weights from HuggingFace may lead to an unstable download process, slow download speed and other problems, we can choose to download the weights of InternLM2-Chat-7B from ModelScope when experiencing network issues.

```bash
pip install -U modelscope

# Download the model weights to the current directory
python -c "from modelscope import snapshot_download; snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='.')"
```

After completing the download, we can start to prepare the dataset for fine-tuning.

The HuggingFace link and ModelScope link are attached here:

- The HuggingFace link is located at: https://huggingface.co/internlm/internlm2-chat-7b
- The ModelScope link is located at: https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary

## Prepare the fine-tuning dataset

### Download from HuggingFace

```bash
git clone https://huggingface.co/datasets/burkelibbey/colors
```

### Download from ModelScope

Due to the same reason, we can choose to download the dataset from ModelScope.

```bash
git clone https://www.modelscope.cn/datasets/fanqiNO1/colors.git
```

The HuggingFace link and ModelScope link are attached here:

- The HuggingFace link is located at: https://huggingface.co/datasets/burkelibbey/colors
- The ModelScope link is located at: https://modelscope.cn/datasets/fanqiNO1/colors

## Prepare the config

XTuner provides several configs out-of-the-box, which can be viewed via `xtuner list-cfg`. We can use the following command to copy a config to the current directory.

```bash
xtuner copy-cfg internlm2_7b_qlora_colorist_e5 .
```

Explanation of the config name:

| Config Name | internlm2_7b_qlora_colorist_e5 |
| ----------- | ------------------------------ |
| Model Name  | internlm2_7b                   |
| Algorithm   | qlora                          |
| Dataset     | colorist                       |
| Epochs      | 5                              |

The directory structure at this point should look like this:

```bash
.
├── colors
│   ├── colors.json
│   ├── dataset_infos.json
│   ├── README.md
│   └── train.jsonl
├── internlm2_7b_qlora_colorist_e5_copy.py
└── Shanghai_AI_Laboratory
    └── internlm2-chat-7b
        ├── config.json
        ├── configuration_internlm2.py
        ├── configuration.json
        ├── generation_config.json
        ├── modeling_internlm2.py
        ├── pytorch_model-00001-of-00008.bin
        ├── pytorch_model-00002-of-00008.bin
        ├── pytorch_model-00003-of-00008.bin
        ├── pytorch_model-00004-of-00008.bin
        ├── pytorch_model-00005-of-00008.bin
        ├── pytorch_model-00006-of-00008.bin
        ├── pytorch_model-00007-of-00008.bin
        ├── pytorch_model-00008-of-00008.bin
        ├── pytorch_model.bin.index.json
        ├── README.md
        ├── special_tokens_map.json
        ├── tokenization_internlm2_fast.py
        ├── tokenization_internlm2.py
        ├── tokenizer_config.json
        └── tokenizer.model
```

## Modify the config

In this step, we need to modify the model path and dataset path to local paths and modify the dataset loading method.
In addition, since the copied config is based on the Base model, we also need to modify the `prompt_template` to adapt to the Chat model.

```diff
#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
- pretrained_model_name_or_path = 'internlm/internlm2-7b'
+ pretrained_model_name_or_path = './Shanghai_AI_Laboratory/internlm2-chat-7b'

# Data
- data_path = 'burkelibbey/colors'
+ data_path = './colors/train.jsonl'
- prompt_template = PROMPT_TEMPLATE.default
+ prompt_template = PROMPT_TEMPLATE.internlm2_chat

...
#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=colors_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
```

Therefore, `pretrained_model_name_or_path`, `data_path`, `prompt_template`, and the `dataset` fields in `train_dataset` are modified.

## Start fine-tuning

Once having done the above steps, we can start fine-tuning using the following command.

```bash
# Single GPU
xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py
# Multiple GPUs
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py
# Slurm
srun ${SRUN_ARGS} xtuner train ./internlm2_7b_qlora_colorist_e5_copy.py --launcher slurm
```

The correct training log may look similar to the one shown below:

```text
01/29 21:35:34 - mmengine - INFO - Iter(train) [ 10/720]  lr: 9.0001e-05  eta: 0:31:46  time: 2.6851  data_time: 0.0077  memory: 12762  loss: 2.6900
01/29 21:36:02 - mmengine - INFO - Iter(train) [ 20/720]  lr: 1.9000e-04  eta: 0:32:01  time: 2.8037  data_time: 0.0071  memory: 13969  loss: 2.6049  grad_norm: 0.9361
01/29 21:36:29 - mmengine - INFO - Iter(train) [ 30/720]  lr: 1.9994e-04  eta: 0:31:24  time: 2.7031  data_time: 0.0070  memory: 13969  loss: 2.5795  grad_norm: 0.9361
01/29 21:36:57 - mmengine - INFO - Iter(train) [ 40/720]  lr: 1.9969e-04  eta: 0:30:55  time: 2.7247  data_time: 0.0069  memory: 13969  loss: 2.3352  grad_norm: 0.8482
01/29 21:37:24 - mmengine - INFO - Iter(train) [ 50/720]  lr: 1.9925e-04  eta: 0:30:28  time: 2.7286  data_time: 0.0068  memory: 13969  loss: 2.2816  grad_norm: 0.8184
01/29 21:37:51 - mmengine - INFO - Iter(train) [ 60/720]  lr: 1.9863e-04  eta: 0:29:58  time: 2.7048  data_time: 0.0069  memory: 13969  loss: 2.2040  grad_norm: 0.8184
01/29 21:38:18 - mmengine - INFO - Iter(train) [ 70/720]  lr: 1.9781e-04  eta: 0:29:31  time: 2.7302  data_time: 0.0068  memory: 13969  loss: 2.1912  grad_norm: 0.8460
01/29 21:38:46 - mmengine - INFO - Iter(train) [ 80/720]  lr: 1.9681e-04  eta: 0:29:05  time: 2.7338  data_time: 0.0069  memory: 13969  loss: 2.1512  grad_norm: 0.8686
01/29 21:39:13 - mmengine - INFO - Iter(train) [ 90/720]  lr: 1.9563e-04  eta: 0:28:36  time: 2.7047  data_time: 0.0068  memory: 13969  loss: 2.0653  grad_norm: 0.8686
01/29 21:39:40 - mmengine - INFO - Iter(train) [100/720]  lr: 1.9426e-04  eta: 0:28:09  time: 2.7383  data_time: 0.0070  memory: 13969  loss: 1.9819  grad_norm: 0.9127
```

Before training begins, the output of the model is as shown below:

```text
2024/01/29 21:34:58 - mmengine - INFO - before_train in EvaluateChatHook.
2024/01/29 21:35:03 - mmengine - INFO - Sample output:
<s><|im_start|>system
You are a professional color designer. Please provide the corresponding colors based on the description of Human.
<|im_end|>
<|im_start|>user
请给我一个像天空一样清澈透明的蓝色。<|im_end|>
<|im_start|>assistant
为了匹配您所描述的“像天空一样清澈透明的蓝色”，我建议您选择一种名为“天蓝”（Cerulean）的颜色。这种颜色通常被用来代表天空、海洋和清澈的水域，它具有明亮

2024/01/29 21:35:07 - mmengine - INFO - Sample output:
<s><|im_start|>system
You are a professional color designer. Please provide the corresponding colors based on the description of Human.
<|im_end|>
<|im_start|>user
Please give me a clear blue like the sky.<|im_end|>
<|im_start|>assistant
The color you're describing is a clear, bright blue that resembles the color of the sky on a sunny day. This color is often referred to as "sky blue" or "cerulean blue." Here are some specific color codes and names that
```

After training, the output of the model is as shown below:

```text
2024/01/29 22:08:07 - mmengine - INFO - Sample output:
<s><|im_start|>system
You are a professional color designer. Please provide the corresponding colors based on the description of Human.
<|im_end|>
<|im_start|>user
请给我一个像天空一样清澈透明的蓝色。<|im_end|>
<|im_start|>assistant
#1099ee<|im_end|>

2024/01/29 22:08:08 - mmengine - INFO - Sample output:
<s><|im_start|>system
You are a professional color designer. Please provide the corresponding colors based on the description of Human.
<|im_end|>
<|im_start|>user
Please give me a clear blue like the sky.<|im_end|>
<|im_start|>assistant
#0066dd<|im_end|>
```

The color of the model output is shown below:

- 天空一样清澈透明的蓝色：![天空一样清澈透明的蓝色](https://img.shields.io/badge/天空一样清澈透明的蓝色-1099EE)
- A clear blue like the sky: ![A clear blue like the sky](https://img.shields.io/badge/A_clear_blue_like_the_sky-0066DD)

It is clear that the output of the model after training has been fully aligned with the content of the dataset.

# Model Convert + LoRA Merge

After training, we will get several `.pth` files that do **NOT** contain all the parameters of the model, but store the parameters updated by the training process of the QLoRA algorithm. Therefore, we need to convert these `.pth` files to HuggingFace format and merge them into the original LLM weights.

### Model Convert

XTuner has already integrated the tool of converting the model to HuggingFace format. We can use the following command to convert the model.

```bash
# Create the directory to store parameters in hf format
mkdir work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf

# Convert the model to hf format
xtuner convert pth_to_hf internlm2_7b_qlora_colorist_e5_copy.py \
                            work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720.pth \
                            work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf
```

This command will convert `work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720.pth` to hf format based on the contents of the config `internlm2_7b_qlora_colorist_e5_copy.py` and will save it in `work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf`.

### LoRA Merge

XTuner has also integrated the tool of merging LoRA weights, we just need to execute the following command:

```bash
# Create the directory to store the merged weights
mkdir work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged

# Merge the weights
xtuner convert merge Shanghai_AI_Laboratory/internlm2-chat-7b \
                        work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf \
                        work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged \
                        --max-shard-size 2GB
```

Similar to the command above, this command will read the original parameter path `Shanghai_AI_Laboratory/internlm2-chat-7b` and the path of parameter which has been converted to hf format `work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf` and merge the two parts of the parameters and save them in `work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged`, where the maximum file size for each parameter slice is 2GB.

## Chat with the model

To better appreciate the model's capabilities after merging the weights, we can chat with the model. XTuner also integrates the tool of chatting with models. We can start a simple demo to chat with the model with the following command:

```bash
xtuner chat work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged \
                --prompt-template internlm2_chat \
                --system-template colorist
```

Of course, we can also choose not to merge the weights and instead chat directly with the LLM + LoRA Adapter, we just need to execute the following command:

```bash
xtuner chat Shanghai_AI_Laboratory/internlm2-chat-7b
                --adapter work_dirs/internlm2_7b_qlora_colorist_e5_copy/iter_720_hf \
                --prompt-template internlm2_chat \
                --system-template colorist
```

where `work_dirs/internlm2_7b_qlora_colorist_e5_copy/merged` is the path to the merged weights, `--prompt-template internlm2_chat` specifies that the chat template is InternLM2-Chat, and `-- system-template colorist` specifies that the System Prompt for conversations with models is the template required by the Colorist dataset.

There is an example below:

```text
double enter to end input (EXIT: exit chat, RESET: reset history) >>> A calming but fairly bright light sky blue, between sky blue and baby blue, with a hint of fluorescence due to its brightness.

#66ccff<|im_end|>
```

The color of the model output is shown below:

A calming but fairly bright light sky blue, between sky blue and baby blue, with a hint of fluorescence due to its brightness: ![#66ccff](https://img.shields.io/badge/A_calming_but_fairly_bright_light_sky_blue_between_sky_blue_and_baby_blue_with_a_hint_of_fluorescence_due_to_its_brightness-66CCFF).
