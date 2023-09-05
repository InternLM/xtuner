# Chat with fine-tuned LLMs

## Chat with [InternLM](https://github.com/InternLM/InternLM)

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

## Chat with [Llama2](https://github.com/facebookresearch/llama)

> Don't forget to use `huggingface-cli login` and input your access token first to access Llama2! See [here](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) to learn how to obtain your access token.

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
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

## Chat with [Qwen](https://github.com/QwenLM)

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  export SERPER_API_KEY="xxx"  # Please get the key from https://serper.dev to support google search!
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

## Chat with [Baichuan](https://github.com/baichuan-inc)

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

  ## Chat with [CodeLlama](https://github.com/facebookresearch/codellama)

- CodeLlama-7B, Instruct

  ```shell
  xtuner chat codellama/CodeLlama-7b-Instruct-hf --prompt-template code_llama_chat
  ```
