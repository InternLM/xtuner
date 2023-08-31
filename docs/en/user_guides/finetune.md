# Fine-tune the pretrained LLMs

## QLoRA Fine-tune [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, oasst1

  ```shell
  xtuner train internlm_7b_qlora_oasst1_e3
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  xtuner train internlm_7b_qlora_arxiv_gentitle_e3
  ```

- InternLM-7B, Colorist

  ```shell
  xtuner train internlm_7b_qlora_colorist_e5
  ```

- InternLM-7B, Coder

  ```shell
  xtuner train internlm_7b_qlora_code_alpaca_e3
  ```

- InternLM-7B, SQL

  ```shell
  xtuner train internlm_7b_qlora_sql_e3
  ```

- InternLM-7B, Lawyer

  ```shell
  xtuner train internlm_7b_qlora_lawyer_e3
  ```

- InternLM-7B, Open-Platypus

  ```shell
  xtuner train internlm_7b_qlora_open_platypus_e3
  ```

- InternLM-7B, Alpaca-enzh

  ```shell
  xtuner train internlm_7b_qlora_alpaca_enzh_e3
  ```

## QLoRA Fine-tune [Llama2](https://github.com/facebookresearch/llama)

> Don't forget to use `huggingface-cli login` and input your access token first to access Llama2! See [here](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) to learn how to obtain your access token.

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  NPROC_PER_NODE=8 xtuner train llama2_7b_qlora_moss_sft_all_e2_gpu8  # Recommended!
  xtuner train llama2_7b_qlora_moss_sft_all_e1
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  xtuner train llama2_7b_qlora_arxiv_gentitle_e3
  ```

- Llama2-7B, Colorist

  ```shell
  xtuner train llama2_7b_qlora_colorist_e5
  ```

## QLoRA Fine-tune [Qwen](https://github.com/QwenLM)

- Qwen-7B, MOSS-003-SFT **(plugins!)**

  ```shell
  NPROC_PER_NODE=8 xtuner train qwen_7b_qlora_moss_sft_all_e2_gpu8  # Recommended!
  xtuner train qwen_7b_qlora_moss_sft_all_e1
  ```

- Qwen-7B, oasst1

  ```shell
  xtuner train qwen_7b_qlora_oasst1_e3
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  xtuner train qwen_7b_qlora_arxiv_gentitle_e3
  ```

- Qwen-7B, Alpaca-enzh

  ```shell
  xtuner train qwen_7b_qlora_alpaca_enzh_e3
  ```

## QLoRA Fine-tune [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, oasst1

  ```shell
  xtuner train baichuan_7b_qlora_oasst1_e3
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  xtuner train baichuan_7b_qlora_arxiv_gentitle_e3
  ```

- Baichuan-7B, Alpaca-enzh

  ```shell
  xtuner train baichuan_7b_qlora_alpaca_enzh_e3
  ```
