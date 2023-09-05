# 与微调后的大语言模型 LLMs 对话

## 与微调后的 [InternLM](https://github.com/InternLM/InternLM) 对话

- InternLM-7B, oasst1

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-oasst1 --prompt-template openassistant
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-arxiv-gentitle --prompt-template title
  ```

- InternLM-7B, Colorist

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-colorist --prompt-template colorist
  ```

- InternLM-7B, Coder

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-coder --prompt-template code
  ```

- InternLM-7B, SQL

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-sql --prompt-template sql
  ```

- InternLM-7B, Lawyer

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-lawyer --prompt-template lawyer
  ```

- InternLM-7B, Open-Platypus

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-open-platypus --prompt-template alpaca
  ```

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner chat internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-alpaca-enzh --prompt-template alpaca
  ```

## 与微调后的 [Llama2](https://github.com/facebookresearch/llama) 对话

> 在使用 Llama2 之前，请先使用 `huggingface-cli login` 输入你的访问令牌（access token）！点击[这里](https://huggingface.co/docs/hub/security-tokens#user-access-tokens)了解如何获取访问令牌。

- Llama2-7B, MOSS-003-SFT **（支持调用插件）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得API_KEY，以此支持谷歌搜索！
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --with-plugins calculate solve search --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-arxiv-gentitle --prompt-template title
  ```

- Llama2-7B, Colorist

  ```shell
  xtuner chat meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-colorist --prompt-template colorist
  ```

## 与微调后的 [Qwen](https://github.com/QwenLM) 对话

- Qwen-7B, MOSS-003-SFT **（支持调用插件）**

  ```shell
  export SERPER_API_KEY="xxx"  # 请从 https://serper.dev 获得API_KEY，以此支持谷歌搜索！
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-moss-003-sft --bot-name Qwen --prompt-template moss_sft --with-plugins calculate solve search --command-stop-word "<eoc>" --answer-stop-word "<eom>"
  ```

- Qwen-7B, oasst1

  ```shell
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-oasst1 --prompt-template openassistant --answer-stop-word '<|endoftext|>'
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-arxiv-gentitle --prompt-template title --answer-stop-word '<|endoftext|>'
  ```

- Qwen-7B, Alpaca-enzh

  ```shell
  xtuner chat Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-alpaca-enzh --prompt-template alpaca --answer-stop-word '<|endoftext|>'
  ```

## 与微调后的 [Baichuan](https://github.com/baichuan-inc) 对话

- Baichuan-7B, oasst1

  ```shell
  xtuner chat baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-oasst1 --prompt-template openassistant
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  xtuner chat baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-arxiv-gentitle --prompt-template title --no-streamer
  ```

- Baichuan-7B, Alpaca-enzh

  ```shell
  xtuner chat baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-alpaca-enzh --prompt-template alpaca
  ```

  ## 与 [CodeLlama](https://github.com/facebookresearch/codellama) 对话

- CodeLlama-7B, Instruct

  ```shell
  xtuner chat codellama/CodeLlama-7b-Instruct-hf --prompt-template code_llama_chat
  ```
