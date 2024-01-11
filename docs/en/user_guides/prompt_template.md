# Prompt Template

The prompt template of XTuner ensures consistency with the LLMs' official templates. Below, we will elaborate on its logic using the example of InternLM-Chat model (`internlm_chat`).

## Structure

```python
internlm_chat=dict(
    SYSTEM='<|System|>:{system}\n',
    INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
    SUFFIX='<eoa>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<eoa>'])
```

- `SYSTEM`: The template for the "system" field during Q&A, where `{system}` represents the "system" text. It's worth noting that this field only appears once in multi-turn dialogues, specifically in the first turn.

- `INSTRUCTION`: The template for the "instruction" field during Q&A, where `{input}` represents the user instruction text.

- `SUFFIX`: The suffix for the "instruction" field, which will be appended to the "response" of each Q&A turn. Typically, this also serves as a special ending symbol (*i.e.*, `eos`). Defaults to `''`.

- `SUFFIX_AS_EOS`: Represents whether the aforementioned suffix acts as an ending symbol. If set to `True`, it will replace the `eos_token` of the `tokenizer`. Otherwise, the `eos_token` of the `tokenizer` will still be used to denote the end of sequence. Defaults to `False`.

- `SEP`: Used to separate multi-turn dialogues, it will be appended after the `INSTRUCTION` and `SUFFIX`. Defaults to `''`.

- `STOP_WORDS`: Used to specify the stop words, this information will be utilized during the text generation stage.  It's worth noting that the `eos_token` of the `tokenizer` is automatically added to `STOP_WORDS`, without the need for manual setting.

## Results

**Single-turn**

```
<|System|>:{system}
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
```

**Multi-turn**

```
<|System|>:{system}
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
```

## Choosing the prompt template

| Model                                    | Prompt Template |
| ---------------------------------------- | --------------- |
| baichuan-inc/Baichuan-7B                 | default\*       |
| baichuan-inc/Baichuan-13B-Base           | default\*       |
| baichuan-inc/Baichuan-13B-Chat           | baichuan_chat   |
| baichuan-inc/Baichuan2-7B-Base           | default\*       |
| baichuan-inc/Baichuan2-7B-Chat           | baichuan2_chat  |
| baichuan-inc/Baichuan2-13B-Base          | default\*       |
| baichuan-inc/Baichuan2-13B-Chat          | baichuan2_chat  |
| THUDM/chatglm2-6b                        | chatglm2        |
| THUDM/chatglm3-6b                        | chatglm3        |
| THUDM/chatglm3-6b-base                   | chatglm3        |
| deepseek-ai/deepseek-coder-6.7b-base     | deepseek_coder  |
| deepseek-ai/deepseek-coder-6.7b-instruct | deepseek_coder  |
| internlm/internlm-7b                     | default\*       |
| internlm/internlm-20b                    | default\*       |
| internlm/internlm-chat-7b                | internlm_chat   |
| internlm/internlm-chat-20b               | internlm_chat   |
| huggyllama/llama-7b                      | default         |
| meta-llama/Llama-2-7b-hf                 | llama2_chat     |
| meta-llama/Llama-2-7b-chat-hf            | llama2_chat     |
| meta-llama/Llama-2-70b-hf                | llama2_chat     |
| lmsys/vicuna-7b-v1.5                     | vicuna          |
| lmsys/vicuna-13b-v1.5                    | vicuna          |
| mistralai/Mistral-7B-v0.1                | mistral         |
| mistralai/Mixtral-8x7B-v0.1              | mixtral         |
| mistralai/Mixtral-8x7B-Instruct-v0.1     | mixtral         |
| Qwen/Qwen-1_8B                           | default\*       |
| Qwen/Qwen-1_8B-Chat                      | qwen_chat       |
| Qwen/Qwen-7B                             | default\*       |
| Qwen/Qwen-7B-Chat                        | qwen_chat       |
| Qwen/Qwen-72B                            | default\*       |
| Qwen/Qwen-72B-Chat                       | qwen_chat       |
| bigcode/starcoder                        | default         |
| 01-ai/Yi-6B                              | default         |
| 01-ai/Yi-34B                             | default         |
| HuggingFaceH4/zephyr-7b-beta             | zephyr          |
| deepseek-ai/deepseek-moe-16b-base        | deepseek_moe    |
| deepseek-ai/deepseek-moe-16b-chat        | deepseek_moe    |

\*: The official template has special tokens (like `<|im_start|>`, `<|im_end|>`) that were not trained during the pre-training phase. Therefore, these models utilize the `default` template.
