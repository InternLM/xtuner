# Chat with fine-tuned LLMs

## Chat with [InternLM](https://github.com/InternLM/InternLM)

### InternLM-7B

- InternLM-7B, oasst1

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-oasst1 --prompt-template internlm_chat
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-arxiv-gentitle --prompt-template internlm_chat --system-template arxiv_gentile
  ```

- InternLM-7B, Colorist

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-colorist --prompt-template internlm_chat --system-template colorist
  ```

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

- InternLM-7B, MSAgent **(Lagent ReAct!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
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
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-arxiv-gentitle --prompt-template internlm_chat --system-template arxiv_gentile
  ```

- InternLM-20B, Colorist

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-colorist --prompt-template internlm_chat --system-template colorist
  ```

- InternLM-20B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-20b --adapter xtuner/internlm-20b-qlora-alpaca-enzh --prompt-template internlm_chat --system-template alpaca
  ```

- InternLM-20B, MSAgent **(Lagent ReAct!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
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

## Chat with [Llama2](https://github.com/facebookresearch/llama)

> Don't forget to use `huggingface-cli login` and input your access token first to access Llama2! See [here](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) to learn how to obtain your access token.

### Llama-2-7B

- Llama-2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --system-template moss_sft --with-plugins calculate solve search --no-streamer
  ```

- Llama-2-7B, MSAgent **(Lagent ReAct!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-msagent-react --lagent
  ```

## Chat with [Qwen](https://github.com/QwenLM)

### Qwen-7B

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-moss-003-sft --bot-name Qwen --prompt-template moss_sft --system-template moss_sft --with-plugins calculate solve search
  ```
