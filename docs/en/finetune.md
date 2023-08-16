# Fine-tune the pretrained LLMs

## Fine-tune [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner train internlm_7b_qlora_alpaca
  ```

- InternLM-7B, oasst1

  ```shell
  xtuner train internlm_7b_qlora_oasst1
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner train internlm_7b_qlora_arxiv
  ```

## Fine-tune [Llama2](https://github.com/facebookresearch/llama)

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner train llama2_7b_qlora_moss_sft_all
  xtuner dist_train llama2_7b_qlora_moss_sft_all_gpu8 8  # Recommended!
  ```

- Llama2-7B, MOSS-003-SFT-Plugins **(plugins!)**

  ```shell
  xtuner train llama2_7b_qlora_moss_sft_plugins
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  xtuner train llama2_7b_qlora_arxiv
  ```

## Fine-tune [Qwen](https://github.com/QwenLM)

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner train qwen_7b_qlora_moss_sft_all
  xtuner dist_train qwen_7b_qlora_moss_sft_all_gpu8 8  # Recommended!
  ```

- Qwen-7B, Alpaca-enzh

  ```shell
  xtuner train qwen_7b_qlora_alpaca
  ```

- Qwen-7B, oasst1

  ```shell
  xtuner train qwen_7b_qlora_oasst1
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  xtuner train qwen_7b_qlora_arxiv
  ```

## Fine-tune [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, Alpaca-enzh

  ```shell
  xtuner train baichuan_7b_qlora_alpaca
  ```

- Baichuan-7B, oasst1

  ```shell
  xtuner train baichuan_7b_qlora_oasst1
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  xtuner train baichuan_7b_qlora_arxiv
  ```
