# Fine-tune the pretrained LLMs

## Fine-tune [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, Alpaca-enzh

  ```shell
  python ./tools/train.py ./configs/internlm/internlm_7b/internlm_7b_qlora_alpaca.py
  ```

- InternLM-7B, oasst1

  ```shell
  python ./tools/train.py ./configs/internlm/internlm_7b/internlm_7b_qlora_oasst1.py
  ```

- InternLM-7B, Arxiv Gentitle

  ```shell
  python ./tools/train.py ./configs/internlm/internlm_7b/internlm_7b_qlora_arxiv.py
  ```

## Fine-tune [Llama2](https://github.com/facebookresearch/llama)

- Llama2-7B, MOSS-003-SFT **(plugins!)**

  > Recommended to use multiple GPUs (*e.g.*, 8), with larger batch-size (*e.g.*, 64) and longer training epochs (*e.g.*, 2).

  ```shell
  python ./tools/train.py ./configs/llama/llama2_7b/llama2_7b_qlora_moss_sft_all.py
  ```

- Llama2-7B, MOSS-003-SFT-Plugins **(plugins!)**

  ```shell
  python ./tools/train.py ./configs/llama/llama2_7b/llama2_7b_qlora_moss_sft_plugins.py
  ```

- Llama2-7B, Arxiv Gentitle

  ```shell
  python ./tools/train.py ./configs/llama/llama2_7b/llama2_7b_qlora_arxiv.py
  ```

## Fine-tune [Qwen](https://github.com/QwenLM)

- Qwen-7B, Alpaca-enzh

  ```shell
  python ./tools/train.py ./configs/qwen/qwen_7b/qwen_7b_qlora_alpaca.py
  ```

- Qwen-7B, oasst1

  ```shell
  python ./tools/train.py ./configs/qwen/qwen_7b/qwen_7b_qlora_oasst1.py
  ```

- Qwen-7B, Arxiv Gentitle

  ```shell
  python ./tools/train.py ./configs/qwen/qwen_7b/qwen_7b_qlora_arxiv.py
  ```

## Fine-tune [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, Alpaca-enzh

  ```shell
  python ./tools/train.py ./configs/baichuan/baichuan_7b/baichuan_7b_qlora_alpaca.py
  ```

- Baichuan-7B, oasst1

  ```shell
  python ./tools/train.py ./configs/baichuan/baichuan_7b/baichuan_7b_qlora_oasst1.py
  ```

- Baichuan-7B, Arxiv Gentitle

  ```shell
  python ./tools/train.py ./configs/baichuan/baichuan_7b/baichuan_7b_qlora_arxiv.py
  ```
