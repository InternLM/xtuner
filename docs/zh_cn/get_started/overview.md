# 概述

本节将向您介绍 XTuner 的整体框架和工作流程，并提供详细的教程链接。

## 什么是XTuner？

- XTuner 是由 InternLM 团队开发的一个高效、灵活、全能的轻量化大模型微调工具库。
- 主要用于多种大模型的高效微调，包括 InternLM 和多模态图文模型 LLaVa。
- 支持 QLoRA、全量微调等多种微调训练方法，并可与 DeepSpeed 集成优化训练。
- 提供模型、数据集、数据管道和算法支持，配备配置文件和快速入门指南。
- 训练所得模型可无缝接入部署工具库 LMDeploy、大规模评测工具库 OpenCompass 及 VLMEvalKit。为大型语言模型微调提供一个全面且用户友好的解决方案。

## XTuner 的工作流程
我们可以通过以下这张图，简单的了解一下XTuner的整体运作流程。
![image](https://github.com/Jianfeng777/xtuner/assets/108343727/1e5b91ca-e19c-4bbf-9659-d9ace33465c3)

整个工作流程分为以下四个步骤：
1. **数据采集及格式转换**：
   - 首先，根据任务的不同，我们需要明确微调目标，进行数据采集，并将数据转换为 XTuner 所支持的格式类型。
   - 然后我们还需要各种自己的硬件条件选择合适的微调方法和合适的基座模型。

2. **配置文件的创建**：
   - 我们可以通过执行 `xtuner list-cfg` 命令列出所有配置文件。
   - 通过上面选择的微调方法和基座模型找到合适的配置文件，并使用 `xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}` 命令复制到本地端。
   - 复制完成后还需要根据自己的需求修改配置文件以更新模型路径和数据集路径。
   - 特定时候还需要调整模型参数和配置，更改 `load_dataset` 函数和 `dataset_map_fn` 函数。并根据模型选择合适的 `prompt_template`。

3. **模型训练**：
   - 修改配置文件后，我就可以使用 `xtuner train` 命令启动训练。
   - 除此之外我们还可以设置特定参数优化训练，如启用 deepspeed，以及设置训练文件的保存路径。
   - 假如意外的中断了训练，还可以通过加上`--resume {checkpoint_path}`的方式进行模型续训。具体可看下面的指令详解。

4. **模型转换、测试及部署**：
   - 在完成训练后，找到对应的训练文件并执行 `xtuner convert pth_to_hf` 命令，就可以将转换模型格式为 huggingface 格式。
   - 转换完成后，我们就可以以转换后的文件路径并使用 `xtuner chat` 命令启动模型进行性能测试。
   - 除此之外，我们还可以在安装 LMDeploy 后通过 `python -m lmdeploy.pytorch.chat` 命令进行模型部署。
  
以下是每一步具体的指令应用：
![image](https://github.com/Jianfeng777/xtuner/assets/108343727/8b3dcf6b-0dfe-43b0-a67c-a0850e895f1f)


## XTuner的核心模块

### apis
- 提供了部分给第三方连接的接口（目前正在开发）

### Configs
- 存放着不同模型、不同数据集以及微调方法的配置文件
- 可以自行从 huggingface 上下载模型和数据集后进行一键启动

### Dataset
- 在`map_fns`下存放了支持的数据集的映射规则
- 在`collate_fns`下存放了关于数据整理函数部分的内容
- 提供了用于存放和加载不同来源数据集的函数和类

### Engine
- `hooks`中展示了哪些信息将会哪个阶段下在终端被打印出来。

### Evaluation
- `metric`中展示了XTuer所支持的一种评测数据集MMLU的格式，我们可以通过`XTuner test`进行调用

### Model
- `modules`中存放了对部分模型的优化策略，可以非侵入式地对模型进行修改。

### Tools
- 这里面是XTuner中的核心工具箱，里面存放了我们常用的打印config文件（`list_cfg`）、复制config文件(`copy_cfg`)、训练(`train`)以及对话(`chat`)等等。
- 在`model_converters`中也提供了模型转换以及切分的脚本
- 在`plugin`中提供了部分工具调用的函数


## XTuner当前支持的模型、数据集及微调方法

### 支持的大语言模型
XTuner目前支持以下大语言模型，可支持所有huggingface格式的大语言模型：
- `baichuan`
- `chatglm`
- `internlm`
- `llama`
- `llava`
- `mistral`
- `mixtral`
- `qwen`
- `yi`
- `starcoder`
- `zephyr`
- ...

### 支持的数据集格式
XTuner目前支持以下数据集格式：
- `alpaca`
- `alpaca_zh`
- `code_alpaca`
- `arxiv`
- `colors`
- `crime_kg_assistant`
- `law_reference`
- `llava`
- `medical`
- `msagent`
- `oasst1`
- `openai`
- `openorca`
- `pretrain`
- `sql`
- `stack_exchange`
- `tiny_codes`
- `wizardlm`
- ...

### 支持的微调方法
XTuner目前支持以下微调方法：
- `QLora`
- `Lora`
- `Full`
- ...
