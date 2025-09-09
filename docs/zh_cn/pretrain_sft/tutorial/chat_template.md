# 对话模板说明

## GPT-OSS 

OpenAI 开源的 [GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) 模型内置链式思维（Chain‑of‑Thought）并可调节推理强度，并提出了 [Harmony 响应格式](https://cookbook.openai.com/articles/openai-harmony), 并且
其采用推理模式进行训练。相比于之前常规模型对话模板和训练模式有些许区别，因此需要特殊说明。

在阅读本文前，建议您先阅读：

- [OpenAI Harmony Response Format](https://cookbook.openai.com/articles/openai-harmony)
- [HF OpenAI GPT-OSS](https://huggingface.co/blog/welcome-openai-gpt-oss)

相比于常规的 `system/user/assistant` 三角色对话模板，GPT-OSS 新引入了 `developer` 角色。其中 system 和 developer 的定义为：

- system： 用于指定推理力度、元信息（如知识截止时间）以及内置工具
- developer： 用于提供有关模型指令（通常被认为是“系统提示”）和可用功能工具的信息。

简单来说 developer 是我们之前常用的 system 角色，而 system 则用于推理模型的元信息和内置工具等。

如果你的数据集中存在推理数据即 assistant 字段中还包括了 thinking 字段，如下所示：

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

则在训练时候会把除了最后一个 thinking 同时为 assistant 的 content 外，所有的 thinking 字段都去掉，变成如下形式：

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

同时需要特别注意的是： **其会在所有回合中对标签进行掩码（mask），仅保留最后一条 assistant 消息的标签，也就是说如果多轮对话数据集，则只有最后一个 assistant 会计算 Loss。**
如果不想保持原来的，每个回合的 assistant 都计算 Loss，则可以考虑将 GPT-OSS 的 `HybridChatTemplate` 中 `only_last_assistant_loss` 修改为 False。

