# 与微调后的大语言模型 LLMs 对话

## 与微调后的 [InternLM](https://github.com/InternLM/InternLM) 对话

### InternLM-7B

- InternLM-7B, oasst1

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-oasst1 --prompt-template internlm_chat
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-arxiv-gentitle --prompt-template internlm_chat --system-prompt arxiv_gentile
  ```

- InternLM-7B, Colorist

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-colorist --prompt-template internlm_chat --system-prompt colorist
  ```

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

- InternLM-7B, MSAgent **（支持 Lagent ReAct）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得 API_KEY，以此支持谷歌搜索！
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-msagent-react --lagent
  ```

### InternLM-Chat-7B

- InternLM-Chat-7B, oasst1

  ```shell
  xtuner chat internlm/internlm-chat-7b --adapter xtuner/internlm-chat-7b-qlora-oasst1 --prompt-template internlm_chat
  ```

- InternLM-Chat-7B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-chat-7b --adapter xtuner/internlm-chat-7b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

### InternLM-20B

- InternLM-20B, oasst1

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-oasst1 --prompt-template internlm_chat
  ```

- InternLM-20B, Arxiv Gentitle

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-arxiv-gentitle --prompt-template internlm_chat --system-prompt arxiv_gentile
  ```

- InternLM-20B, Colorist

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-colorist --prompt-template internlm_chat --system-prompt colorist
  ```

- InternLM-20B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

- InternLM-20B, MSAgent **（支持 Lagent ReAct）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得 API_KEY，以此支持谷歌搜索！
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-msagent-react --lagent
  ```

### InternLM-Chat-20B

- InternLM-Chat-20B, oasst1

  ```shell
  xtuner chat internlm/internlm-chat-20b --adapter xtuner/internlm-chat-20b-qlora-oasst1 --prompt-template internlm_chat
  ```

- InternLM-Chat-20B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-chat-20b --adapter xtuner/internlm-chat-20b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

## 与微调后的 [Llama-2](https://github.com/facebookresearch/llama) 对话

> 在使用 Llama-2 之前，请先使用 `huggingface-cli login` 输入你的访问令牌（access token）！点击[这里](https://huggingface.co/docs/hub/security-tokens#user-access-tokens)了解如何获取访问令牌。

### Llama-2-7B

- Llama-2-7B, MOSS-003-SFT **（支持调用插件）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得 API_KEY，以此支持谷歌搜索！
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --system-template moss_sft --with-plugins calculate solve search --no-streamer
  ```

- Llama-2-7B, MSAgent **（支持 Lagent ReAct）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得 API_KEY，以此支持谷歌搜索！
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-msagent-react --lagent
  ```

## 与微调后的 [Qwen](https://github.com/QwenLM) 对话

### Qwen-7B

- Qwen-7B, MOSS-003-SFT **（支持调用插件）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得API_KEY，以此支持谷歌搜索！
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-moss-003-sft --bot-name Qwen --prompt-template moss_sft --system-template moss_sft --with-plugins calculate solve search
  ```
