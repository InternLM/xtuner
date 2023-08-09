# MMChat

<div align="center">

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/en)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/)

[📘 Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🤔 Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose) |
[⚙️ Model Zoo](<>)

English | [简体中文](README_zh-CN.md)

</div>

## 📣 News

- **\[2023.08.xx\]** We release XXX, with multiple fine-tuned adapters.

## 🧭 Introduction

MMChat is a toolkit for efficiently fine-tuning LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams.

- **Efficiency**: Support the LLM fine-tuning on consumer-grade GPUs.
- **Versatile**: Support various LLMs, datasets and algorithms, allowing users to choose the most suitable solution for their requirements.
- **Compatibility**: Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed) and the [HuggingFace](https://huggingface.co) training pipeline, enabling effortless integration and utilization.

> 💥 [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams have also collaborated in developing [LMDeploy](https://github.com/InternLM/lmdeploy), a toolkit for for compressing, deploying, and serving LLM. Welcome to subscribe to stay updated with our latest developments.

## 🌟 Demos

- QLoRA fine-tune for InternLM-7B [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yzGeYXayLomNQjLD4vC6wgUHvei3ezt4?usp=sharing)
- Chat with Llama2-7B-Plugins [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<>)
- Use MMChat in HuggingFace training pipeline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eBI9yiOkX-t7P-0-t9vS8y1x5KmWrkoU?usp=sharing)

## 🔥 Supports

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>Models</b>
</td>
<td>
  <b>Datasets</b>
</td>
<td>
  <b>Strategies</b>
</td>
 <td>
  <b>Algorithms</b>
</td>
</tr>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li><a href="configs/internlm/internlm_7b">InternLM</a></li>
  <li><a href="configs/internlm/internlm_chat_7b">InternLM-Chat</a></li>
  <li><a href="configs/llama/llama_7b">Llama</a></li>
  <li><a href="configs/llama/llama2_7b">Llama2</a></li>
  <li><a href="configs/llama/llama2_7b_chat">Llama2-Chat</a></li>
  <li><a href="configs/qwen/qwen_7b">Qwen</a></li>
  <li><a href="configs/qwen/qwen_7b_chat">Qwen-Chat</a></li>
  <li><a href="configs/baichuan/baichuan_7b">Baichuan-7B</a></li>
  <li><a href="configs/baichuan/baichuan_13b_base">Baichuan-13B-Base</a></li>
  <li><a href="configs/baichuan/baichuan_13b_chat">Baichuan-13B-Chat</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="configs/_base_/datasets/moss_003_sft_all.py">MOSS-003-SFT</a></li>
  <li><a href="configs/_base_/datasets/arxiv.py">Arxiv GenTitle</a></li>
  <li><a href="configs/_base_/datasets/open_orca.py">OpenOrca</a></li>
  <li><a href="configs/_base_/datasets/alpaca.py">Alpaca en</a> / <a href="configs/_base_/datasets/alpaca_zh.py">zh</a></li>
  <li><a href="configs/_base_/datasets/oasst1.py">oasst1</a></li>
  <li><a href="configs/_base_/datasets/cmd.py">Chinese Medical Dialogue</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li>(Distributed) Data Parallel</li>
  <li><a href="examples">DeepSpeed</a> 🚀</li>
</ul>
</td>
<td>
<ul>
  <li>Full parameter fine-tune</li>
  <li><a href="http://arxiv.org/abs/2106.09685">LoRA</a></li>
  <li><a href="http://arxiv.org/abs/2305.14314">QLoRA</a></li>
  <li>...</li>
</ul>
</td>
</tr>
</tbody>
</table>

## 🛠️ Quick Start

### Installation

Below are quick steps for installation:

```shell
conda create -n mmchat python=3.10
conda activate mmchat
git clone XXX
cd MMChat
pip install -v -e .
```

### Chat [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<>)

<table>
<tr>
  <th colspan="3" align="center">Examples of Plugins-based Chat 🔥🔥🔥</th>
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

MMChat provides the tools to chat with pretrained / fine-tuned LLMs.

- For example, we can start the chat with Llama2-7B-Plugins by

  ```shell
  python ./tools/chat_hf.py meta-llama/Llama-2-7b --adapter XXX --bot-name Llama2 --prompt plugins --with-plugins --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
  ```

For more usages, please see [chat.md](./docs/en/chat.md).

### Fine-tune [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yzGeYXayLomNQjLD4vC6wgUHvei3ezt4?usp=sharing)

MMChat supports the efficient fine-tune (*e.g.*, QLoRA) for Large Language Models (LLM).

Taking the QLoRA fine-tuning  as an example, we can start it by

- For example, we can start the QLoRA fine-tuning of InternLM-7B with oasst1 dataset by

  ```shell
  # On a single GPU
  python ./tools/train.py ./configs/internlm/internlm_7b/internlm_7b_qlora_oasst1.py
  # On multiple GPUs
  bash ./tools/dist_train.sh ./configs/internlm/internlm_7b/internlm_7b_qlora_oasst1.py ${GPU_NUM}
  ```

For more usages, please see [finetune.md](./docs/en/finetune.md).

### Deploy

- **Step 0**, convert the pth adapter to HuggingFace adapter, by

  ```shell
  python ./tools/model_converters/adapter_pth2hf.py \
  		${CONFIG_FILE} \
  		${PATH_TO_PTH_ADAPTER} \
  		${SAVE_PATH_TO_HF_ADAPTER}
  ```

- **Step 1**, merge the HuggingFace adapter to the pretrained LLM, by

  ```shell
  python ./tools/model_converters/merge_lora_hf.py \
      ${NAME_OR_PATH_TO_HF_MODEL} \
      ${NAME_OR_PATH_TO_HF_ADAPTER} \
      ${SAVE_PATH}
  ```

- **Step 2**, deploy the merged LLM with any other framework, such as [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀.

  ```shell
  pip install lmdeploy
  python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_HF_MODEL} \
      --max_new_tokens 256 \
      --temperture 0.8 \
      --top_p 0.95 \
      --seed 0
  ```

  🎯 We are woking closely with [LMDeploy](https://github.com/InternLM/lmdeploy), to implement the deployment of **dialogues with plugins**!

### Evaluation

- We recommend using [OpenCompass](https://github.com/InternLM/opencompass),  a comprehensive and systematic LLM evaluation library, which currently supports 50+ datasets with about 300,000 questions.

## 🔜 Roadmap

## 🤝 Contributing

We appreciate all contributions to XXX. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## 🎖️ Acknowledgement

- [Llama 2](https://github.com/facebookresearch/llama)
- [QLoRA](https://github.com/artidoro/qlora)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
