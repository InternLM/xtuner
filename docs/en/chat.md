# Chat with fine-tuned LLMs

## Chat with [InternLM](https://github.com/InternLM/InternLM)

- InternLM-7B, Alpaca-enzh

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt alpaca
    ```

- InternLM-7B, oasst1

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

- InternLM-7B, Arxiv Gentitle

    ```shell
    python ./tools/chat_hf.py internlm/internlm-7b --adapter XXX --prompt title
    ```

## Chat with [Llama2](https://github.com/facebookresearch/llama)

- Llama2-7B, MOSS-003-SFT (plugins!)

    ```shell
    python ././tools/chat_hf.py meta-llama/Llama-2-7b --adapter XXX --bot-name Llama2 --prompt plugins --with-plugins --command-stop-word "<eoc>" --answer-stop-word "<eom>" --no-streamer
    ```

- Llama2-7B, Arxiv Gentitle

    ```shell
    python ././tools/chat_hf.py meta-llama/Llama-2-7b --adapter XXX --prompt title --no-streamer
    ```

## Chat with [Qwen](https://github.com/QwenLM)

- Qwen-7B, Alpaca-enzh

    ```shell
    python ./tools/chat_hf.py Qwen/Qwen-7B --adapter XXX --prompt alpaca --answer-stop-word '<|endoftext|>'
    ```

- Qwen-7B, oasst1

    ```shell
    python ./tools/chat_hf.py Qwen/Qwen-7B --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

- Qwen-7B, Arxiv Gentitle

    ```shell
    python ./tools/chat_hf.py Qwen/Qwen-7B --adapter XXX --prompt title --answer-stop-word '<|endoftext|>'
    ```

## Chat with [Baichuan](https://github.com/baichuan-inc)

- Baichuan-7B, Alpaca-enzh

    ```shell
    python ./tools/chat_hf.py baichuan-inc/Baichuan-7B --adapter XXX --prompt alpaca
    ```

- Baichuan-7B, oasst1

    ```shell
    python ./tools/chat_hf.py baichuan-inc/Baichuan-7B --adapter XXX --prompt openassistant --answer-stop-word "###"
    ```

- Baichuan-7B, Arxiv Gentitle

    ```shell
    python ./tools/chat_hf.py baichuan-inc/Baichuan-7B --adapter XXX --prompt title
    ```
