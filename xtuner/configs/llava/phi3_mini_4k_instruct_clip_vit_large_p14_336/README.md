# LLaVA-Phi-3-mini

## Results

<div  align="center">
<img src="https://github.com/InternLM/xtuner/assets/36994684/78524f65-260d-4ae3-a687-03fc5a19dcbb" alt="Image" width=500" />
</div>

| Model                 | MMBench Test (EN) | MMMU  Val | SEED-IMG | AI2D Test | ScienceQA Test | HallusionBench aAcc | POPE | GQA  | TextVQA |   MME    | MMStar |                                                                                                                                                                                                                  Configs                                                                                                                                                                                                                   |
| :-------------------- | :---------------: | :-------: | :------: | :-------: | :------------: | :-----------------: | :--: | :--: | :-----: | :------: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B         |       66.5        |   35.3    |   60.5   |   54.8    |      70.4      |        44.9         | 85.9 | 62.0 |  58.2   | 1511/348 |  30.3  |                                                                                                                                                                                                                     -                                                                                                                                                                                                                      |
| LLaVA-Llama-3-8B      |       68.9        |   36.8    |   69.8   |   60.9    |      73.3      |        47.3         | 87.2 | 63.5 |  58.0   | 1506/295 |  38.2  |           [Pretrain](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/pretrain/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_pretrain.py) / [Fine-tune](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/finetune/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_finetune.py)           |
| LLaVA-Llama-3-8B-v1.1 |       72.3        |   37.1    |   70.1   |   70.0    |      72.9      |        47.7         | 86.4 | 62.6 |  59.0   | 1469/349 |  45.1  | [Pretrain](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/pretrain/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain.py) / [Fine-tune](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/finetune/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune.py) |
| **LLaVA-Phi-3-mini**  |       69.2        |   41.4    |   70.0   |   69.3    |      73.7      |        49.8         | 87.3 | 61.5 |  57.8   | 1477/313 |  43.7  |                                                                                                        [Pretrain](./pretrain/llava_phi3_mini_4k_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain.py) / [Fine-tune](./finetune/llava_phi3_mini_4k_instruct_full_clip_vit_large_p14_336_full_e2_gpu8_internvl_finetune.py)                                                                                                        |

## Resources

- Official LLaVA format model (`xtuner/llava-phi-3-mini`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-phi-3-mini) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-phi-3-mini)
- HuggingFace LLaVA format model (`xtuner/llava-phi-3-mini-hf`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-phi-3-mini-hf) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-phi-3-mini-hf)
- XTuner LLaVA format model (`xtuner/llava-phi-3-mini-xtuner`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-phi-3-mini-xtuner) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-phi-3-mini-xtuner)
- GGUF model (`xtuner/llava-phi-3-mini-gguf`): 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-phi-3-mini-gguf) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-phi-3-mini-gguf)
- Pretrained projector weights: 🤗 [HuggingFace](https://huggingface.co/xtuner/llava-phi-3-mini-pretrain) / 🤖 [ModelScope](https://modelscope.cn/models/xtuner/llava-phi-3-mini-pretrain)

## Data Preparation

Please refer to [here](https://github.com/InternLM/xtuner/tree/main/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336#data-preparation).

## Training

### LLaVA-Phi-3-mini

1. Pretrain

```bash
NPROC_PER_NODE=8 xtuner train llava_phi3_mini_4k_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

2. Fine-tune

```bash
NPROC_PER_NODE=8 xtuner train llava_phi3_mini_4k_instruct_full_clip_vit_large_p14_336_full_e2_gpu8_internvl_finetune --deepspeed deepspeed_zero2 --seed 1024
```

## Model Conversion

### Step 0. Convert `.pth` file to LLaVA model in xtuner format ([LLaVA-Phi-3-mini-xtuner](https://huggingface.co/xtuner/llava-phi-3-mini-xtuner))

After training, we will obtain a set of weights (*i.e.*, `iter_xxx.pth`), which are not in the universal HuggingFace format. We first need to convert them to the LLaVA model in xtuner format.

```bash
xtuner convert pth_to_hf $FINETUNE_CFG $PTH_PATH $SAVE_PATH
# e.g., xtuner convert pth_to_hf llava_phi3_mini_4k_instruct_full_clip_vit_large_p14_336_full_e2_gpu8_internvl_finetune ./iter_39620.pth ./iter_39620_xtuner
```

```
./iter_39620_xtuner
├── added_tokens.json
├── config.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── projector
│   ├── config.json
│   ├── configuration_projector.py
│   ├── modeling_projector.py
│   └── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── visual_encoder
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

At this time, the LLaVA model of xtuner-format can engage in conversation using xtuner chat, by

```bash
xtuner chat ./iter_39620_xtuner \
  --llava ./iter_39620_xtuner \
  --prompt-template phi3_chat \
  --image $IMAGE_PATH
```

and in MMBench evaluation, by

```bash
xtuner mmbench ./iter_39620_xtuner \
  --llava ./iter_39620_xtuner \
  --prompt-template phi3_chat \
  --data-path $DATA_PATH \
  --work-dir $RESULT_PATH
```

Here, `$DATA_PATH` refers to one of the mmbench datasets. You can download the expected data by

```bash
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_EN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_TEST_CN.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv
```

### Step 1. Convert LLaVA in xtuner format to official LLaVA format or HuggingFace LLaVA format

- The official LLaVA format is structured similarly to the architecture of the [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) model.
- The HuggingFace LLaVA format is structured similarly to the architecture of the [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model.

Since the official LLaVA format and the HuggingFace LLaVA format only support Llama architecture as the LLM, we need to first convert the phi-3 model to an equivalent Llama LLM.

```bash
python ./convert_phi_to_llama.py --phi_path ./iter_39620_xtuner --save_path ./iter_39620_xtuner_llama_llm
```

Here, `--phi_path` should specify the path to phi-3, which is the path obtained from Step.0 for the xtuner-format LLaVA model. `--save_path` should specify the save path for the converted Llama LLM.

#### To official LLaVA format ([LLaVA-Phi-3-mini](https://huggingface.co/xtuner/llava-phi-3-mini))

We can utilize the following command to obtain the LLaVA model in the official LLaVA format.

```bash
python ./convert_xtuner_weights_to_llava.py --text_model_id ./iter_39620_xtuner_llama_llm --vision_model_id ./iter_39620_xtuner/visual_encoder --projector_weight ./iter_39620_xtuner/projector/model.safetensors --save_path ./iter_39620_llava
```

Here, the converted LLaVA model in official LLaVA format is saved to `./iter_39620_llava`.

```
./iter_39620_llava
├── added_tokens.json
├── config.json
├── generation_config.json
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

#### To HuggingFace LLaVA format ([LLaVA-Phi-3-mini-hf](https://huggingface.co/xtuner/llava-phi-3-mini-hf))

We can utilize the following command to obtain the LLaVA model in the HuggingFace LLaVA format.

```bash
python ./convert_xtuner_weights_to_hf.py --text_model_id ./iter_39620_xtuner_llama_llm --vision_model_id ./iter_39620_xtuner/visual_encoder --projector_weight ./iter_39620_xtuner/projector/model.safetensors --save_path ./iter_39620_hf
```

Here, the converted LLaVA model in HuggingFace LLaVA format is saved to `./iter_39620_hf`.

```
./iter_39620_hf
├── added_tokens.json
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

## Chat

- XTuner LLaVA format [docs](https://huggingface.co/xtuner/llava-phi-3-mini-xtuner#quickstart)
- Official LLaVA format [docs](https://huggingface.co/xtuner/llava-phi-3-mini#quickstart)
- HuggingFace LLaVA format [docs](https://huggingface.co/xtuner/llava-phi-3-mini-hf#quickstart)
- GGUF format [docs](https://huggingface.co/xtuner/llava-phi-3-mini-gguf#quickstart)
