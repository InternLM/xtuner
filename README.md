# MMChat

## Highlights

### 🔥 Supported Models, Datasets, and Strategies

<table>
<tbody>
<tr align="left" valign="middle">
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
  <li><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">Alpaca en</a> / <a href="https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese">zh</a></li>
  <li><a href="https://huggingface.co/datasets/timdettmers/openassistant-guanaco">oasst1</a></li>
  <li><a href="https://github.com/WangRongsheng/ChatGenTitle">Arxiv GenTitle</a> 👨‍🎓</li>
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






### 🔧 Chat with Plugins

Calculate, Equations Solve, Web Search, ...

### 🌟 Colab Demos 

### 🖥️ Minimum System Requirements





## Quick Start

### Installation

1. Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

  ```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  ```

2. Install dependencies and XXX

  ```shell
git clone XXX
cd XXX
pip install -v -e .
  ```

### Fine-tune

We support the efficient fine-tune (*e.g.*, QLoRA) for Large Language Models (LLM). 

Taking the QLoRA fine-tuning of InternLM-7B with Alpaca dataset as an example, we can start it by：

```shell
python tools/train.py configs/internlm/internlm_7b/internlm_7b_qlora_alpaca.py
```

### Chat

We support to chat with pretrained / fine-tuned LLMs.

- With the pretrained HuggingFace LLM, and the corresponding HuggingFace adapter fine-tuned from XXX

  ```shell
  python tools/chat_hf.py [MODEL_NAME_OR_PATH] --adapter [ADAPTER_NAME_OR_PATH] ...
  ```

  *e.g.*,

  - Llama-2-7B, plugins adapter,

    ```shell
    python tools/chat_hf.py meta-llama/Llama-2-7b --adapter XXX --prompt plugins --with-plugins --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
    ```

  - InternLM-7B, arxiv GenTitle adapter,

    ```shell
    python tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt title
    ```

  - InternLM-7B, alpaca adapter,

    ```shell
    python tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt alpaca
    ```

  - InternLM-7B, oasst1 adapter,

    ```shell
    python tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

- With XXX config, and the corresponding PTH adapter fine-tuned from XXX

  ```shell
  python tools/chat.py [CONFIG] --adapter [PTH_ADAPTER_PATH] ...
  ```

### Deploy

- **Step 0**, convert the pth adapter to HuggingFace adapter, by

  ```shell
  python tools/model_converters/adapter_pth2hf.py [CONFIG] [PTH_ADAPTER_PATH] [SAVE_DIR]
  ```

- **Step 1**, merge the HuggingFace adapter to the pretrained LLM, by

  ```shell
  python tools/model_converters/merge_lora_hf.py [MODEL_NAME_OR_PATH] [ADAPTER_NAME_OR_PATH] [SAVE_DIR]
  ```

- **Step 2**, deploy the merged LLM with any other framework, such as [LMDeploy](https://github.com/InternLM/lmdeploy) 🚀.

  - Note: We are woking closely with LMDeploy team, to implement the deployment of **dialogues with plugins**.

## Performance

## Roadmap

## Acknowledgement

## License
