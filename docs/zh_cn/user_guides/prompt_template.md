# 对话模版（prompt template）

XTuner 提供一系列对话模版（prompt template），其与众多 LLM 的官方模版完全对齐。本文档将以 InternLM-Chat 的模版 `internlm_chat` 为例，详细介绍对话模版的代码结构及执行逻辑。

## 代码结构

```python
internlm_chat=dict(
    SYSTEM='<|System|>:{system}\n',
    INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
    SUFFIX='<eoa>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<eoa>'])
```

- `SYSTEM`：表示问答时“系统”字段的模版，其中 `{system}` 指代“系统”文本。值得注意的是，该字段在多轮对话中只会出现一次，即在第一轮。
- `INSTRUCTION`：表示问答时“指令”字段的模版，其中 `{input}` 指代用户指令文本。
- `SUFFIX`：表示“指令”字段的后缀，将会追加在每一轮问答的“回答”后面。通常，这也是一个特殊的结束符号。默认是空串`''`。
- `SUFFIX_AS_EOS`：表示上述后缀是否作为结束符号。如果为 `True`，则会取代 `tokenizer` 的 `eos_token`，否则，仍会使用 `tokenizer` 的 `eos_token` 表示结束符号。默认是 `False`。
- `SEP`：用于间隔多轮对话，将会追加在 `INSTRUCTION` 和 `SUFFIX` 后面。默认是空串`''`。
- `STOP_WORDS`：用于指明结束词，该信息将被用在文本生成阶段。值得注意的是，`tokenizer` 的 `eos_token` 会被自动添加到 `STOP_WORDS`，而无需手动配置。

## 结果

**单轮对话**

```
<|System|>:{system}
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
```

**多轮对话**

```
<|System|>:{system}
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
<|User|>:{input}<eoh>
<|Bot|>:{output}<eoa>
```

## 模版的选择

| 模型                                     | 对话模版       |
| ---------------------------------------- | -------------- |
| baichuan-inc/Baichuan-7B                 | default\*      |
| baichuan-inc/Baichuan-13B-Base           | default\*      |
| baichuan-inc/Baichuan-13B-Chat           | baichuan_chat  |
| baichuan-inc/Baichuan2-7B-Base           | default\*      |
| baichuan-inc/Baichuan2-7B-Chat           | baichuan2_chat |
| baichuan-inc/Baichuan2-13B-Base          | default\*      |
| baichuan-inc/Baichuan2-13B-Chat          | baichuan2_chat |
| THUDM/chatglm2-6b                        | chatglm2       |
| THUDM/chatglm3-6b                        | chatglm3       |
| THUDM/chatglm3-6b-base                   | chatglm3       |
| deepseek-ai/deepseek-coder-6.7b-base     | deepseek_coder |
| deepseek-ai/deepseek-coder-6.7b-instruct | deepseek_coder |
| internlm/internlm-7b                     | default\*      |
| internlm/internlm-20b                    | default\*      |
| internlm/internlm-chat-7b                | internlm_chat  |
| internlm/internlm-chat-20b               | internlm_chat  |
| huggyllama/llama-7b                      | default        |
| meta-llama/Llama-2-7b-hf                 | llama2_chat    |
| meta-llama/Llama-2-7b-chat-hf            | llama2_chat    |
| meta-llama/Llama-2-70b-hf                | llama2_chat    |
| lmsys/vicuna-7b-v1.5                     | vicuna         |
| lmsys/vicuna-13b-v1.5                    | vicuna         |
| mistralai/Mistral-7B-v0.1                | mistral        |
| mistralai/Mixtral-8x7B-v0.1              | mixtral        |
| mistralai/Mixtral-8x7B-Instruct-v0.1     | mixtral        |
| Qwen/Qwen-1_8B                           | default\*      |
| Qwen/Qwen-1_8B-Chat                      | qwen_chat      |
| Qwen/Qwen-7B                             | default\*      |
| Qwen/Qwen-7B-Chat                        | qwen_chat      |
| Qwen/Qwen-72B                            | default\*      |
| Qwen/Qwen-72B-Chat                       | qwen_chat      |
| bigcode/starcoder                        | default        |
| 01-ai/Yi-6B                              | default        |
| 01-ai/Yi-34B                             | default        |
| HuggingFaceH4/zephyr-7b-beta             | zephyr         |
| deepseek-ai/deepseek-moe-16b-base        | deepseek_moe   |
| deepseek-ai/deepseek-moe-16b-chat        | deepseek_moe   |
| internlm/internlm2-1_8b                  | default\*      |
| internlm/internlm2-7b                    | default\*      |
| internlm/internlm2-20b                   | default\*      |
| internlm/internlm2-chat-1_8b             | internlm2_chat |
| internlm/internlm2-chat-7b               | internlm2_chat |
| internlm/internlm2-chat-20b              | internlm2_chat |
| Qwen/Qwen1.5-0.5B                        | default\*      |
| Qwen/Qwen1.5-0.5B-Chat                   | qwen_chat      |
| Qwen/Qwen1.5-1.8B                        | default\*      |
| Qwen/Qwen1.5-1.8B-Chat                   | qwen_chat      |
| Qwen/Qwen1.5-4B                          | default\*      |
| Qwen/Qwen1.5-4B-Chat                     | qwen_chat      |
| Qwen/Qwen1.5-7B                          | default\*      |
| Qwen/Qwen1.5-7B-Chat                     | qwen_chat      |
| Qwen/Qwen1.5-14B                         | default\*      |
| Qwen/Qwen1.5-14B-Chat                    | qwen_chat      |
| Qwen/Qwen1.5-72B                         | default\*      |
| Qwen/Qwen1.5-72B-Chat                    | qwen_chat      |
| google/gemma-2b                          | default\*      |
| google/gemma-2b-it                       | gemma\*        |
| google/gemma-7b                          | default\*      |
| google/gemma-7b-it                       | gemma\*        |

\*: 官方对话模版中存在特殊 token（比如 `<|im_start|>`、`<|im_end|>`），这类特殊 token 在预训练阶段并未得到训练。故，使用 `default` 模版。
