# Chat Template Description

## GPT-OSS

OpenAI's open-source [GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) model has built-in Chain-of-Thought and adjustable inference intensity, and proposes the [Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony), and
it is trained in inference mode. Compared with previous conventional model dialogue templates and training modes, there are some differences, so special explanation is needed.

Before reading this article, it is recommended that you read:

- [OpenAI Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony)
- [HF OpenAI GPT-OSS](https://huggingface.co/blog/welcome-openai-gpt-oss)

Compared with the conventional `system/user/assistant` three-role dialogue template, GPT-OSS newly introduces the `developer` role. The definitions of system and developer are:

- system: Used to specify inference intensity, meta-information (such as knowledge cutoff time) and built-in tools
- developer: Used to provide information about model instructions (usually considered "system prompts") and available functional tools.

Simply put, developer is the system role we commonly used before, while system is used for meta-information and built-in tools of the inference model.

If your dataset contains inference data, that is, the assistant field also includes a thinking field, as shown below:

```{code-block} python
messages = [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi! How can I help you today?"},
      {"role": "user", "content": "Can you tell me a joke?"},
      {"role": "assistant", "thinking": "Thinking real hard...", "content": "Okay!"},
      {"role": "user", "content": "Please!"},
      {"role": "assistant", "thinking": "Thinking real hard...Thinking real hard...", "content": "Sure!"},
]
```

Then during training, all thinking fields will be removed except for the last thinking that is also assistant's content, becoming the following form:

```{code-block} python
messages = [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi! How can I help you today?"},
      {"role": "user", "content": "Can you tell me a joke?"},
      {"role": "assistant", "content": "Okay!"}, # del thinking
      {"role": "user", "content": "Please!"},
      {"role": "assistant", "thinking": "Thinking real hard...Thinking real hard...", "content": "Sure!"},
]
```

At the same time, special attention should be paid to: **It will mask labels in all rounds, only keeping the label of the last assistant message, that is, if it is a multi-round dialogue dataset, only the last assistant will calculate Loss.**
If you don't want to keep the original where each round of assistant calculates Loss, you can consider modifying `only_last_assistant_loss` to False in GPT-OSS's `HybridChatTemplate`.