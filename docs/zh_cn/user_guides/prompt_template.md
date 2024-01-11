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
- `STOP_WORDS`：用于指明结束词，该信息将被用在文本生成阶段。值得注意的是，`tokenizer` 的 `eos_token` 会被自动添加到 `STOP_WORDS`，而无序手动配置。

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
