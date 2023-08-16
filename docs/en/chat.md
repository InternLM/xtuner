# Chat with fine-tuned LLMs

## Chat with [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner chat hf internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-alpaca --prompt-template alpaca
  ```

- InternLM-7B, oasst1

  ```shell
  xtuner chat hf internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-oasst1 --prompt-template openassistant --answer-stop-word "###"
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner chat hf internlm/internlm-7b --adapter xtuner/internlm-7b-qlora-arxiv-gentitle --prompt-template title
  ```

## Chat with [Llama2](https://github.com/facebookresearch/llama)

> Don't forget to use `huggingface-cli login` and input your access token first to access Llama2! See [here](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) to learn how to obtain your access token.

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner chat hf meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-moss-003-sft --bot-name Llama2 --prompt-template moss_sft --with-plugins calculate solve search --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  xtuner chat hf meta-llama/Llama-2-7b-hf --adapter xtuner/Llama-2-7b-qlora-arxiv-gentitle --prompt-template title --no-streamer
  ```

## Chat with [Qwen](https://github.com/QwenLM)

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner chat hf Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-moss-003-sft --bot-name Qwen --prompt-template moss_sft --with-plugins calculate solve search --command-stop-word "<eoc>" --answer-stop-word "<eom>"
  ```

- Qwen-7B, Alpaca-enzh

  ```shell
  xtuner chat hf Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-alpaca --prompt-template alpaca --answer-stop-word '<|endoftext|>'
  ```

- Qwen-7B, oasst1

  ```shell
  xtuner chat hf Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-oasst1 --prompt-template openassistant --answer-stop-word "###"
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  xtuner chat hf Qwen/Qwen-7B --adapter xtuner/Qwen-7B-qlora-arxiv-gentitle --prompt-template title --answer-stop-word '<|endoftext|>'
  ```

## Chat with [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, Alpaca-enzh

  ```shell
  xtuner chat hf baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-alpaca --prompt-template alpaca
  ```

- Baichuan-7B, oasst1

  ```shell
  xtuner chat hf baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-oasst1 --prompt-template openassistant --answer-stop-word "###"
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  xtuner chat hf baichuan-inc/Baichuan-7B --adapter xtuner/Baichuan-7B-qlora-arxiv-gentitle --prompt-template title
  ```
