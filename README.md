# MMChat

## Highlights

### 🔥 Supported Models, Datasets, and Strategies

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
</tr>
<tr valign="top">
<td align="left" valign="top">
<ul>
  <li><a href="https://github.com/InternLM/InternLM">InternLM</a></li>
  <li><a href="https://github.com/InternLM/InternLM">InternLM-Chat</a></li>
  <li><a href="https://github.com/facebookresearch/llama">Llama</a></li>
  <li><a href="https://github.com/facebookresearch/llama">Llama2</a></li>
  <li><a href="https://github.com/facebookresearch/llama">Llama2-Chat</a></li>
  <li><a href="https://github.com/QwenLM/Qwen-7B">Qwen</a></li>
  <li><a href="https://github.com/QwenLM/Qwen-7B">Qwen-Chat</a></li>
  <li><a href="https://github.com/baichuan-inc/Baichuan-7B">Baichuan-7B</a></li>
  <li><a href="https://github.com/baichuan-inc/Baichuan-13B">Baichuan-13B-Base</a></li>
  <li><a href="https://github.com/baichuan-inc/Baichuan-13B">Baichuan-13B-Chat</a></li>
  <li>...</li>
</ul>
</td>
<td>
<ul>
  <li><a href="https://github.com/OpenLMLab/MOSS/tree/main/SFT_data">MOSS-003-SFT</a> 🔧</li>
  <li><a href="https://github.com/WangRongsheng/ChatGenTitle">Arxiv GenTitle</a> 👨‍🎓</li>
  <li><a href="https://huggingface.co/datasets/Open-Orca/OpenOrca">OpenOrca</a> 🐋</li>
  <li><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">Alpaca en</a> / <a href="https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese">zh</a> 🦙</li>
  <li><a href="https://huggingface.co/datasets/timdettmers/openassistant-guanaco">oasst1</a> 🤖️</li>
  <li><a href="https://github.com/Toyhom/Chinese-medical-dialogue-data">Chinese Medical Dialogue</a> 🧑‍⚕️</li>
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
</tr>
</tbody>
</table>

### 🔧 LLMs with Plugins

- Calculate, Equations Solve, Web Search, ...

<img src="https://github.com/InternLM/lmdeploy/assets/36994684/43a87e81-a726-4ef1-a251-c698186b4938">

### 🌟 Colab Demos

- InternLM-7B, QLoRA Fine-tune. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yzGeYXayLomNQjLD4vC6wgUHvei3ezt4?usp=sharing)
- Llama2-7B-Plugins, Chat. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<>)

### 🖥️ Minimum System Requirements

## Quick Start

### Installation

1. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), *e.g.*,

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. Install dependencies and XXX.

```shell
git clone XXX
cd XXX
pip install -v -e .
```

### Fine-tune

We support the efficient fine-tune (*e.g.*, QLoRA) for LLMs.

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
  bash ./tools/dist_train.sh \
      ${CONFIG_FILE} \
      ${GPU_NUM} \
      [optional arguments]
  ```

### Chat

We support to chat with pretrained / fine-tuned LLMs.

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

### Evaluation

- If a comprehensive and systematic evaluation of the LLM is required, we recommend using the [OpenCompass](https://github.com/InternLM/opencompass), which currently supports evaluation scheme of 50+ datasets with about 300,000 questions.

- In XXX, we support the MMLU evaluation for LLMs, by

  ```
  python ./tools/test.py ${CONFIG_FILE} --checkpoint ${PTH_ADAPTER_PATH} [other optional arguments]
  ```

  Notably, all provided configs disable the evaluation since it may introduce potential biases when evaluated by only one dataset and it is widely believed that fine-tune stage introduces little additional knowledge to LLMs.

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
| ---------------------- | ----------------- | ----------------- | ---------------- | ---------------- |
| Llama2-7b              | 42.6              | 46.5              | 42.4             | 46.8             |
| Llama2-7b oasst1 qlora | Data              | Data              | Data             | Data             |

### Instant generation

Nonetheless, experimental findings indicate that MMLU does not fully capture the performance of large-scale models. Hence we've leveraged [SampleGenerateHook](<>) to illustrate the impact of instruction fine-tuning more vividly. By engaging in single-turn dialogues with the model, using user-defined commands at regular intervals throughout the training process, we're able to offer a more lucid portrayal of the model's conversational abality.

## Roadmap

## Acknowledgement

- [Llama 2](https://github.com/facebookresearch/llama-recipes)
- [QLoRA](https://github.com/artidoro/qlora)
- [LMDeploy](https://github.com/InternLM/lmdeploy)

## License
