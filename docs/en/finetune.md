# Fine-tune the pretrained LLMs

## Fine-tune [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner train internlm_7b_qlora_alpaca_e3
  ```

- InternLM-7B, oasst1

  ```shell
  xtuner train internlm_7b_qlora_oasst1_e3
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner train internlm_7b_qlora_arxiv_e3
  ```

## Fine-tune [Llama2](https://github.com/facebookresearch/llama)

> Don't forget to use `huggingface-cli login` and input your access token first to access Llama2! See [here](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) to learn how to obtain your access token.

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner train llama2_7b_qlora_moss_sft_all_e1
  xtuner dist_train llama2_7b_qlora_moss_sft_all_gpu8_e2 8  # Recommended!
  ```

- Llama2-7B, MOSS-003-SFT-Plugins **(plugins!)**

  ```shell
  xtuner train llama2_7b_qlora_moss_sft_plugins_e1
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  xtuner train llama2_7b_qlora_arxiv_e3
  ```

## Fine-tune [Qwen](https://github.com/QwenLM)

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  xtuner train qwen_7b_qlora_moss_sft_all_e1
  xtuner dist_train qwen_7b_qlora_moss_sft_all_gpu8_e2 8  # Recommended!
  ```

- Qwen-7B, Alpaca-enzh

  ```shell
  xtuner train qwen_7b_qlora_alpaca_e1
  ```

- Qwen-7B, oasst1

  ```shell
  xtuner train qwen_7b_qlora_oasst1_e3
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  xtuner train qwen_7b_qlora_arxiv_e3
  ```

## Fine-tune [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, Alpaca-enzh

  ```shell
  xtuner train baichuan_7b_qlora_alpaca_e3
  ```

- Baichuan-7B, oasst1

  ```shell
  xtuner train baichuan_7b_qlora_oasst1_e3
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  xtuner train baichuan_7b_qlora_arxiv_e3
  ```
