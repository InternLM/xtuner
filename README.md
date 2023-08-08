# MMChat

## Introduction

XXX is a toolkit for quickly fine-tuning LLM, developed by the [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams. It has the following core features:

- Embrace [HuggingFace](https://huggingface.co) and provide fast support for new models, datasets, and algorithms 🤗
- Provide a comprehensive solution and related models for [MOSS plugins datasets](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data) 🤖️
- Support arbitrary combinations of multiple datasets during fine-tuning ➕
- Compatible with [DeepSpeed](https://github.com/microsoft/DeepSpeed), enabling the efficient fine-tuning of LLM on multiple GPUs 🚀
- Support [QLoRA](http://arxiv.org/abs/2305.14314), enabling the efficient fine-tuning of LLM using free resources on Colab ⚡️

> 💥 [MMRazor](https://github.com/open-mmlab/mmrazor) and [MMDeploy](https://github.com/open-mmlab/mmdeploy) teams have also collaborated in developing [LMDeploy](https://github.com/InternLM/lmdeploy), a toolkit for for compressing, deploying, and serving LLM. Welcome to subscribe to stay updated with our latest developments.

## Highlights

### 🔥 Supported Models, Datasets, Strategies, and Algorithms

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
  <li><a href="configs/_base_/datasets/moss_003_sft_all.py">MOSS-003-SFT</a> 🔧</li>
  <li><a href="configs/_base_/datasets/arxiv.py">Arxiv GenTitle</a> 👨‍🎓</li>
  <li><a href="configs/_base_/datasets/open_orca.py">OpenOrca</a> 🐋</li>
  <li><a href="configs/_base_/datasets/alpaca.py">Alpaca en</a> / <a href="configs/_base_/datasets/alpaca_zh.py">zh</a> 🦙</li>
  <li><a href="configs/_base_/datasets/oasst1.py">oasst1</a> 🤖️</li>
  <li><a href="configs/_base_/datasets/cmd.py">Chinese Medical Dialogue</a> 🧑‍⚕️</li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li>(Distributed) Data Parallel</li>
  <li><a href="https://github.com/microsoft/DeepSpeed">Deepspeed 🚀</a></li>
  <li>...</li>
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

### 🔧 LLM with Plugins

- Calculate, Equations Solve, Web Search, ...

<img src="https://github.com/InternLM/lmdeploy/assets/36994684/20159556-7711-4b0d-9568-6884998ad66a">

### 🌟 Colab Demos

- InternLM-7B, QLoRA Fine-tune. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yzGeYXayLomNQjLD4vC6wgUHvei3ezt4?usp=sharing)
- Llama2-7B-Plugins, Chat. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<>)

## Quick Start

### Installation

1. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), *e.g.*,

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. (optional) Install mpi4py and deepspeed.

```shell
conda install mpi4py
pip install deepspeed
```

3. Install dependencies and XXX.

```shell
git clone XXX
cd XXX
pip install -v -e .
```

### Chat

- With the pretrained HuggingFace LLM, and the corresponding HuggingFace adapter fine-tuned from XXX.

  ```shell
  python ./tools/chat_hf.py ${MODEL_NAME_OR_PATH} --adapter ${ADAPTER_NAME_OR_PATH} [other optional arguments]
  ```

  <details>
  <summary>Examples</summary>

  - Llama-2-7B, plugins adapter,

    ```shell
    python ./tools/chat_hf.py meta-llama/Llama-2-7b --adapter XXX --bot-name Llama2 --prompt plugins --with-plugins --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
    ```

  - InternLM-7B, arxiv GenTitle adapter,

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt title
    ```

  - InternLM-7B, alpaca adapter,

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt alpaca
    ```

  - InternLM-7B, oasst1 adapter,

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

  </details>

- With XXX config, and the corresponding PTH adapter fine-tuned from XXX.

  ```shell
  python ./tools/chat.py ${CONFIG_FILE} --adapter ${PTH_ADAPTER_PATH} [other optional arguments]
  ```

  <details>
  <summary>Examples</summary>

  - Llama-2-7B, plugins adapter,

    ```shell
    python ./tools/chat.py ./configs/llama/llama2_7b/llama2_7b_qlora_moss_sft_all.py --adapter XXX --prompt plugins --with-plugins --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
    ```

  - InternLM-7B, arxiv GenTitle adapter,

    ```shell
    python ./tools/chat.py ./configs/internlm/internlm_7b/internlm_7b_qlora_arxiv.py --adapter XXX --prompt title
    ```

  - InternLM-7B, alpaca adapter,

    ```shell
    python ./tools/chat.py ./configs/internlm/internlm_7b/internlm_7b_qlora_alpaca.py --adapter XXX --prompt alpaca
    ```

  - InternLM-7B, oasst1 adapter,

    ```shell
    python ./tools/chat.py ./configs/internlm/internlm_7b/internlm_7b_qlora_oasst1.py --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

  </details>

### Fine-tune

- On a single GPU

  ```shell
  python ./tools/train.py ${CONFIG_FILE} [optional arguments]
  ```

  Taking the QLoRA fine-tuning of InternLM-7B with Alpaca dataset as an example, we can start it by

  ```shell
  python ./tools/train.py ./configs/internlm/internlm_7b/internlm_7b_qlora_alpaca.py
  ```

- On multiple GPUs

  ```shell
  bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
  ```

### Evaluation

- If a comprehensive and systematic evaluation of the LLM is required, we recommend using the [OpenCompass](https://github.com/InternLM/opencompass), which currently supports evaluation scheme of 50+ datasets with about 300,000 questions.

- In XXX, we support the MMLU evaluation for LLM, by

  ```
  python ./tools/test.py ${CONFIG_FILE} --checkpoint ${PTH_ADAPTER_PATH} [other optional arguments]
  ```

  Notably, all provided configs disable the evaluation since it may introduce potential biases when evaluated by only one dataset and it is widely believed that fine-tune stage introduces little additional knowledge to LLM.

  <details>
  <summary>How to enable it?</summary>

  If the evaluation is needed, user can add below lines to the original config to enable it.

  ```python
  from mmengine.config import read_base

  with read_base():
      from ..._base_.datasets.evaluation.mmlu_fs import *  # noqa: F401,F403

  test_dataloader.dataset.tokenizer = tokenizer  # noqa: F405
  test_evaluator.tokenizer = tokenizer  # noqa: F405
  ```

  </details>

### Deploy

- **Step 0**, convert the pth adapter to HuggingFace adapter, by

  ```shell
  python ./tools/model_converters/adapter_pth2hf.py ${CONFIG_FILE} ${PTH_ADAPTER_PATH} ${SAVE_DIR}
  ```

- **Step 1**, merge the HuggingFace adapter to the pretrained LLM, by

  ```shell
  python ./tools/model_converters/merge_lora_hf.py ${MODEL_NAME_OR_PATH} ${ADAPTER_NAME_OR_PATH} ${SAVE_DIR}
  ```

- **Step 2**, deploy the merged LLM with any other framework, such as [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀.

  - We are woking closely with LMDeploy team, to implement the deployment of **dialogues with plugins**!

## Performance

### Objective evaluation

The project has conducted testing on various relevant models on the objective evaluation set for the "Natural Language Understanding (NLU)" category. Given that such evaluations are strictly reliant on provided label outputs, the results are devoid of any subjective elements. This allows for a degree of reflection on the performance and numerous practical capabilities of large-scale models. We have empirically tested the performance of a series of related models on the newly released MMLU dataset. Below are the average evaluation results of some models on the validation and test sets.

| Model                  | Valid (zero-shot) | Valid (five-shot) | Test (zero-shot) | Test (five-shot) |
| :--------------------- | ----------------: | ----------------: | ---------------: | ---------------: |
| Llama2-7b              |              42.6 |              46.5 |             42.4 |             46.8 |
| Llama2-7b oasst1 QLoRA |              Data |              Data |             Data |             Data |

### Instant generation

Nonetheless, experimental findings indicate that MMLU does not fully capture the performance of large-scale models. Hence we've leveraged [SampleGenerateHook](<>) to illustrate the impact of instruction fine-tuning more vividly. By engaging in single-turn dialogues with the model, using user-defined commands at regular intervals throughout the training process, we're able to offer a more lucid portrayal of the model's conversational abality.

## Roadmap

## Acknowledgement

- [Llama 2](https://github.com/facebookresearch/llama)
- [QLoRA](http://arxiv.org/abs/2305.14314)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

## License
