<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/xtuner.svg)](https://pypi.org/project/xtuner/)
[![Generic badge](https://img.shields.io/badge/🤗%20Huggingface-xtuner-yellow.svg)](https://huggingface.co/xtuner)

English | [简体中文](README_zh-CN.md)

👋 join us on <a href="https://twitter.com/intern_lm" target="_blank">Twitter</a>, <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/internlm/xtuner.jpg" target="_blank">WeChat</a>

</div>

## 🎉 News

- **\[2023.09.20\]** Support [InternLM-20B](https://huggingface.co/internlm) models!
- **\[2023.09.06\]** Support [Baichuan2](https://huggingface.co/baichuan-inc) models!
- **\[2023.08.30\]** XTuner is released, with multiple fine-tuned adapters on [HuggingFace](https://huggingface.co/xtuner).

## 📖 Introduction

XTuner is a toolkit for efficiently fine-tuning LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams.

- **Efficiency**: Support LLM fine-tuning on consumer-grade GPUs. The minimum GPU memory required for 7B LLM fine-tuning is only **8GB**, indicating that users can use nearly any GPU (even the free resource, *e.g.*, Colab) to fine-tune custom LLMs.
- **Versatile**: Support various **LLMs** ([InternLM](https://huggingface.co/internlm), [Llama2](https://huggingface.co/meta-llama), [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b), [Qwen](https://huggingface.co/Qwen), [Baichuan2](https://huggingface.co/baichuan-inc), ...), **datasets** ([MOSS_003_SFT](https://huggingface.co/datasets/fnlp/moss-003-sft-data), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [WizardLM](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k), [oasst1](https://huggingface.co/datasets/timdettmers/openassistant-guanaco), [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus), [Code Alpaca](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K), [Colorist](https://huggingface.co/datasets/burkelibbey/colors), ...) and **algorithms** ([QLoRA](http://arxiv.org/abs/2305.14314), [LoRA](http://arxiv.org/abs/2106.09685)), allowing users to choose the most suitable solution for their requirements.
- **Compatibility**: Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 and [HuggingFace](https://huggingface.co) 🤗 training pipeline, enabling effortless integration and utilization.

## 🌟 Demos

- Ready-to-use models and datasets from XTuner API [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eBI9yiOkX-t7P-0-t9vS8y1x5KmWrkoU?usp=sharing)

- QLoRA Fine-tune [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing)

- Plugin-based Chat [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144OuTVyT_GvFyDMtlSlTzcxYIfnRsklq?usp=sharing)

  <table>
  <tr>
    <th colspan="3" align="center">Examples of Plugin-based Chat 🔥🔥🔥</th>
  </tr>
  <tr>
  <td>
  <a><img src="https://github.com/InternLM/lmdeploy/assets/36994684/7c429d98-7630-4539-8aff-c89094826f8c"></a>
  </td>
  <td>
  <a><img src="https://github.com/InternLM/lmdeploy/assets/36994684/05d02906-5a82-45bc-b4e3-2cc32d473b2c"></a>
  </td>
  <td>
  <a><img src="https://github.com/InternLM/lmdeploy/assets/36994684/80395303-997a-47f2-b7d2-d585034df683"></a>
  </td>
  </tr>
  </table>

## 🔥 Supports

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>Models</b>
</td>
<td>
  <b>SFT Datasets</b>
</td>
<td>
  <b>Data Pipelines</b>
</td>
 <td>
  <b>Algorithms</b>
</td>
</tr>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li><a href="https://huggingface.co/internlm/internlm-7b">InternLM</a></li>
  <li><a href="https://huggingface.co/meta-llama">Llama</a></li>
  <li><a href="https://huggingface.co/meta-llama">Llama2</a></li>
  <li><a href="https://huggingface.co/THUDM/chatglm2-6b">ChatGLM2</a></li>
  <li><a href="https://huggingface.co/Qwen/Qwen-7B">Qwen</a></li>
  <li><a href="https://huggingface.co/baichuan-inc/Baichuan-7B">Baichuan</a></li>
  <li><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base">Baichuan2</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="https://huggingface.co/datasets/fnlp/moss-003-sft-data">MOSS-003-SFT</a> 🔧</li>
  <li><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">Alpaca en</a> / <a href="https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese">zh</a></li>
  <li><a href="https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k">WizardLM</a></li>
  <li><a href="https://huggingface.co/datasets/timdettmers/openassistant-guanaco">oasst1</a></li>
  <li><a href="https://huggingface.co/datasets/garage-bAInd/Open-Platypus">Open-Platypus</a></li>
  <li><a href="https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K">Code Alpaca</a></li>
  <li><a href="https://huggingface.co/datasets/burkelibbey/colors">Colorist</a> 🎨</li>
  <li><a href="https://github.com/WangRongsheng/ChatGenTitle">Arxiv GenTitle</a></li>
  <li><a href="https://github.com/LiuHC0428/LAW-GPT">Chinese Law</a></li>
  <li><a href="https://huggingface.co/datasets/Open-Orca/OpenOrca">OpenOrca</a></li>
  <li><a href="https://huggingface.co/datasets/shibing624/medical">Medical Dialogue</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="docs/zh_cn/user_guides/incremental_pretraining.md">Incremental Pre-training</a> </li>
  <li><a href="docs/zh_cn/user_guides/single_turn_conversation.md">Single-turn Conversation SFT</a> </li>
  <li><a href="docs/zh_cn/user_guides/multi_turn_conversation.md">Multi-turn Conversation SFT</a> </li>
</ul>
</td>
<td>
<ul>
  <li><a href="http://arxiv.org/abs/2305.14314">QLoRA</a></li>
  <li><a href="http://arxiv.org/abs/2106.09685">LoRA</a></li>
  <li>Full parameter fine-tune</li>
</ul>
</td>
</tr>
</tbody>
</table>

## 🛠️ Quick Start

### Installation

- Install XTuner via pip

  ```shell
  pip install xtuner
  ```

  or with DeepSpeed integration

  ```shell
  pip install 'xtuner[deepspeed]'
  ```

- Install XTuner from source

  ```shell
  git clone https://github.com/InternLM/xtuner.git
  cd xtuner
  pip install -e '.[all]'
  ```

### Fine-tune [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing)

XTuner supports the efficient fine-tune (*e.g.*, QLoRA) for LLMs. Dataset prepare guides can be found on [dataset_prepare.md](./docs/en/user_guides/dataset_prepare.md).

- **Step 0**, prepare the config. XTuner provides many ready-to-use configs and we can view all configs by

  ```shell
  xtuner list-cfg
  ```

  Or, if the provided configs cannot meet the requirements, please copy the provided config to the specified directory and make specific modifications by

  ```shell
  xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
  ```

- **Step 1**, start fine-tuning.

  ```shell
  xtuner train ${CONFIG_NAME_OR_PATH}
  ```

  For example, we can start the QLoRA fine-tuning of InternLM-7B with oasst1 dataset by

  ```shell
  # On a single GPU
  xtuner train internlm_7b_qlora_oasst1_e3
  # On multiple GPUs
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_7b_qlora_oasst1_e3
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm_7b_qlora_oasst1_e3 --launcher slurm
  ```

  For more examples, please see [finetune.md](./docs/en/user_guides/finetune.md).

- **Step 2**, convert the saved PTH model (if using DeepSpeed, it will be a directory) to HuggingFace model, by

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```

### Chat [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144OuTVyT_GvFyDMtlSlTzcxYIfnRsklq?usp=sharing)

XTuner provides tools to chat with pretrained / fine-tuned LLMs.

```shell
xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} [optional arguments]
```

For example, we can start the chat with Llama2-7b with adapter trained from MOSS-003-SFT by

```shell
xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --with-plugins calculate solve search --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
```

For more examples, please see [chat.md](./docs/en/user_guides/chat.md).

### Deployment

- **Step 0**, merge the HuggingFace adapter to pretrained LLM, by

  ```shell
  xtuner convert merge \
      ${NAME_OR_PATH_TO_LLM} \
      ${NAME_OR_PATH_TO_ADAPTER} \
      ${SAVE_PATH} \
      --max-shard-size 2GB
  ```

- **Step 1**, deploy fine-tuned LLM with any other framework, such as [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀.

  ```shell
  pip install lmdeploy
  python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_LLM} \
      --max_new_tokens 256 \
      --temperture 0.8 \
      --top_p 0.95 \
      --seed 0
  ```

  🔥 Seeking efficient inference with less GPU memory? Try 4-bit quantization from [LMDeploy](https://github.com/InternLM/lmdeploy)! For more details, see [here](https://github.com/InternLM/lmdeploy/tree/main#quantization).

  🎯 We are woking closely with [LMDeploy](https://github.com/InternLM/lmdeploy), to implement the deployment of **plugin-based chat**!

### Evaluation

- We recommend using [OpenCompass](https://github.com/InternLM/opencompass), a comprehensive and systematic LLM evaluation library, which currently supports 50+ datasets with about 300,000 questions.

## 🤝 Contributing

We appreciate all contributions to XTuner. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## 🎖️ Acknowledgement

- [Llama 2](https://github.com/facebookresearch/llama)
- [QLoRA](https://github.com/artidoro/qlora)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
