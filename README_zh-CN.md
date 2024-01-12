<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 加入我们：[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=微信)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=推特)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 探索我们的模型：
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)

[English](README.md) | 简体中文

</div>

## 🎉 更新

- **\[2024/01\]** 支持 [DeepSeek-MoE](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) 模型！20GB 显存即可实现 QLoRA 微调，4x80GB 即可实现全参数微调。快速开始请查阅相关[配置文件](xtuner/configs/deepseek/)！
- **\[2023/12\]** 🔥 支持多模态模型 VLM（[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA)）预训练和指令微调！快速开始请查阅此[文档](xtuner/configs/llava/README_zh.md)！
- **\[2023/12\]** 🔥 支持 [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 模型！快速开始请查阅此[文档](xtuner/configs/mixtral/README.md)！
- **\[2023/11\]** 支持 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) 模型！
- **\[2023/10\]** 支持 [MSAgent-Bench](https://modelscope.cn/datasets/damo/MSAgent-Bench) 数据集，并且微调所得大语言模型可应用至 [Lagent](https://github.com/InternLM/lagent) 框架！
- **\[2023/10\]** 优化数据处理逻辑以兼容 `system` 字段，相关细节请查阅[文档](docs/zh_cn/user_guides/dataset_format.md)！
- **\[2023/09\]** 支持 [InternLM-20B](https://huggingface.co/internlm) 系列模型！
- **\[2023/09\]** 支持 [Baichuan2](https://huggingface.co/baichuan-inc) 系列模型！
- **\[2023/08\]** XTuner 正式发布！众多微调模型已上传至 [HuggingFace](https://huggingface.co/xtuner)！

## 📖 介绍

XTuner 是一个轻量级微调大语言模型的工具库，由 [MMRazor](https://github.com/open-mmlab/mmrazor) 和 [MMDeploy](https://github.com/open-mmlab/mmdeploy) 团队联合开发。

- **轻量级**: 支持在消费级显卡上微调大语言模型。对于 7B 参数量，微调所需的最小显存仅为 **8GB**，这使得用户可以使用几乎任何显卡（甚至免费资源，例如Colab）来微调获得自定义大语言模型助手。
- **多样性**: 支持多种**大语言模型**（[InternLM](https://huggingface.co/internlm)、[Llama2](https://huggingface.co/meta-llama)、[ChatGLM](https://huggingface.co/THUDM)、[Qwen](https://huggingface.co/Qwen)、[Baichuan2](https://huggingface.co/baichuan-inc), ...），**数据集**（[MOSS_003_SFT](https://huggingface.co/datasets/fnlp/moss-003-sft-data), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [WizardLM](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k), [oasst1](https://huggingface.co/datasets/timdettmers/openassistant-guanaco), [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus), [Code Alpaca](https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K), [Colorist](https://huggingface.co/datasets/burkelibbey/colors), ...）和**微调算法**（[QLoRA](http://arxiv.org/abs/2305.14314)、[LoRA](http://arxiv.org/abs/2106.09685)），支撑用户根据自身具体需求选择合适的解决方案。
- **兼容性**: 兼容 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 和 [HuggingFace](https://huggingface.co) 🤗 的训练流程，支撑用户无感式集成与使用。

## 🌟 示例

- XTuner APIs所提供的开箱即用的模型与数据集 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17CSO7T8q6KePuvu684IiHl6_id-CjPjh?usp=sharing)

- QLoRA 微调 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing)

- 基于插件的对话 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144OuTVyT_GvFyDMtlSlTzcxYIfnRsklq?usp=sharing)

  <table>
  <tr>
    <th colspan="3" align="center">基于插件的对话 🔥🔥🔥</th>
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

## 🔥 支持列表

<table>
<tbody>
<tr align="center" valign="middle">
<td>
  <b>模型</b>
</td>
<td>
  <b>数据集</b>
</td>
<td>
  <b>数据格式</b>
</td>
 <td>
  <b>微调算法</b>
</td>
</tr>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li><a href="https://huggingface.co/internlm/internlm-7b">InternLM</a></li>
  <li><a href="https://huggingface.co/meta-llama">Llama</a></li>
  <li><a href="https://huggingface.co/meta-llama">Llama2</a></li>
  <li><a href="https://huggingface.co/THUDM/chatglm2-6b">ChatGLM2</a></li>
  <li><a href="https://huggingface.co/THUDM/chatglm3-6b">ChatGLM3</a></li>
  <li><a href="https://huggingface.co/Qwen/Qwen-7B">Qwen</a></li>
  <li><a href="https://huggingface.co/baichuan-inc/Baichuan-7B">Baichuan</a></li>
  <li><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base">Baichuan2</a></li>
  <li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1">Mixtral 8x7B</a></li>
  <li><a href="https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat">DeepSeek MoE</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="https://modelscope.cn/datasets/damo/MSAgent-Bench">MSAgent-Bench</a></li>
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
  <li>全量参数微调</li>
</ul>
</td>
</tr>
</tbody>
</table>

## 🛠️ 快速上手

### 安装

- 推荐使用 conda 先构建一个 Python-3.10 的虚拟环境

  ```bash
  conda create --name xtuner-env python=3.10 -y
  conda activate xtuner-env
  ```

- 通过 pip 安装 XTuner：

  ```shell
  pip install -U xtuner
  ```

  亦可集成 DeepSpeed 安装：

  ```shell
  pip install -U 'xtuner[deepspeed]'
  ```

- 从源码安装 XTuner：

  ```shell
  git clone https://github.com/InternLM/xtuner.git
  cd xtuner
  pip install -e '.[all]'
  ```

### 微调 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QAEZVBfQ7LZURkMUtaq0b-5nEQII9G9Z?usp=sharing)

XTuner 支持微调大语言模型。数据集预处理指南请查阅[文档](./docs/zh_cn/user_guides/dataset_prepare.md)。

- **步骤 0**，准备配置文件。XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

  ```shell
  xtuner list-cfg
  ```

  或者，如果所提供的配置文件不能满足使用需求，请导出所提供的配置文件并进行相应更改：

  ```shell
  xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
  ```

- **步骤 1**，开始微调。

  ```shell
  xtuner train ${CONFIG_NAME_OR_PATH}
  ```

  例如，我们可以利用 QLoRA 算法在 oasst1 数据集上微调 InternLM-7B：

  ```shell
  # 单卡
  xtuner train internlm_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  # 多卡
  (DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_7b_qlora_oasst1_e3 --deepspeed deepspeed_zero2
  (SLURM) srun ${SRUN_ARGS} xtuner train internlm_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2
  ```

  - `--deepspeed` 表示使用 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 🚀 来优化训练过程。XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3 等。如果用户期望关闭此功能，请直接移除此参数。

  - 更多示例，请查阅[文档](./docs/zh_cn/user_guides/finetune.md)。

- **步骤 2**，将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型：

  ```shell
  xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
  ```

### 对话 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144OuTVyT_GvFyDMtlSlTzcxYIfnRsklq?usp=sharing)

XTuner 提供与大语言模型对话的工具。

```shell
xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} [optional arguments]
```

例如：

与 InternLM-7B + Alpaca-enzh adapter 对话：

```shell
xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
```

与 Llama2-7b + MOSS-003-SFT adapter 对话：

```shell
xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --system-template moss_sft --with-plugins calculate solve search --no-streamer
```

更多示例，请查阅[文档](./docs/zh_cn/user_guides/chat.md)。

### 部署

- **步骤 0**，将 HuggingFace adapter 合并到大语言模型：

  ```shell
  xtuner convert merge \
      ${NAME_OR_PATH_TO_LLM} \
      ${NAME_OR_PATH_TO_ADAPTER} \
      ${SAVE_PATH} \
      --max-shard-size 2GB
  ```

- **步骤 1**，使用任意推理框架部署微调后的大语言模型，例如 [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀：

  ```shell
  pip install lmdeploy
  python -m lmdeploy.pytorch.chat ${NAME_OR_PATH_TO_LLM} \
      --max_new_tokens 256 \
      --temperture 0.8 \
      --top_p 0.95 \
      --seed 0
  ```

  🔥 追求速度更快、显存占用更低的推理？欢迎体验 [LMDeploy](https://github.com/InternLM/lmdeploy) 提供的 4-bit 量化！使用指南请见[文档](https://github.com/InternLM/lmdeploy/tree/main#quantization)。

### 评测

- 推荐使用一站式平台 [OpenCompass](https://github.com/InternLM/opencompass) 来评测大语言模型，其目前已涵盖 50+ 数据集的约 30 万条题目。

## 🤝 贡献指南

我们感谢所有的贡献者为改进和提升 XTuner 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 🎖️ 致谢

- [Llama 2](https://github.com/facebookresearch/llama)
- [QLoRA](https://github.com/artidoro/qlora)
- [LMDeploy](https://github.com/InternLM/lmdeploy)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

## 🖊️ 引用

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## 开源许可证

该项目采用 [Apache License 2.0 开源许可证](LICENSE)。同时，请遵守所使用的模型与数据集的许可证。
