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

In configs provided by XTuner, the chat model uses the same template as the original dialogue, while the base models all utilize the `default` template.

The base model uses the default template rather than the original dialogue template. This decision was made because in certain models (*e.g.*, Qwen), there may exist special tokens (like `<|im_start|>`, `<|im_end|>`) within the dialogue template that were not trained during the pre-training phase. In such cases, if LoRA fine-tuning is employed, both the `embed_tokens` and the `lm_head` layers are frozen by default, preventing the learning of these special tokens.
Therefore, the base models all utilize the `default` template.
