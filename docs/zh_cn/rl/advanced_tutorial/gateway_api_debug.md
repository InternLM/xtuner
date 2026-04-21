# Gateway 兼容接口联调

本文记录如何使用真实的 Agent 客户端和 OpenAI SDK 联调 XTuner Gateway，验证 Gateway 对 Anthropic Messages、OpenAI Responses 和 OpenAI Chat Completions 接口的兼容情况。

## 适用场景

当你修改 Gateway、Rollout Controller、Agent Loop 或协议适配层后，可以按本文流程做一次端到端验证，确认：

- Gateway 能够接收 `/v1/messages`、`/v1/responses` 和 `/v1/chat/completions` 请求。
- Claude Code、Codex 等真实 Agent 客户端能够连接到本地 Gateway。
- 普通对话和工具调用链路都能正常返回。
- Gateway 的请求捕获日志能够记录调试所需的协议转换信息。

## 前置条件

1. 已安装 XTuner 运行环境，并能启动 Rollout Controller 和 Gateway。
2. Gateway 服务默认监听 `http://127.0.0.1:8091`。
3. Gateway 模型名配置为 `local-test`。
4. 鉴权 token 使用本地调试值 `dummy`。
5. 启动 Gateway 时建议打开 `capture_folder`，便于回看请求、协议适配结果和模型输出。

```{note}
真实 Agent 客户端会携带较长的系统提示词和工具定义。联调 Claude Code 时建议将上下文长度设置到 32K；联调 Codex 时建议至少设置到 16K。
```

## 启动 Gateway

先启动 Rollout Controller 和 Gateway。以下命令是本地调试脚本示例：

```bash
python .dev_scripts/debug_gateway.py \
  --model-path </path/to/model> \
  --model-name local-test \
  --context-length 32768
```

启动时需要确认：

- Gateway 端口为 `8091`。
- 模型名为 `local-test`。
- 上下文长度满足当前客户端需求。
- 已配置 `capture_folder`。

## 验证 Anthropic Messages 接口

Claude Code 通过 Anthropic Messages API 访问 Gateway，可用于验证 `/v1/messages` 的协议适配和工具调用链路。

### 安装 Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### 配置环境变量

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8091
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=local-test
export API_TIMEOUT_MS=600000
```

### 验证普通对话

启动 Claude Code 后发送：

```text
Reply with exactly: OK
```

如果客户端能够收到模型回复，说明 `/v1/messages` 的基础请求链路可用。

### 验证工具调用

继续发送以下 prompt：

```text
Use your tools to find the gateway route definitions, then add a single log line for every incoming request to /v1/messages. Show me the exact file you changed and the patch you would apply.
```

如果 Claude Code 能够正常调用工具、读取仓库文件，并返回拟修改的文件和 patch，说明工具调用链路可用。

## 验证 OpenAI Responses 接口

Codex 通过 OpenAI Responses API 访问 Gateway，可用于验证 `/v1/responses` 的协议适配和工具调用链路。

### 安装 Codex

按 Codex 官方安装方式完成安装后，配置本地模型提供方。

### 配置 Codex

在 Codex 的 `config.toml` 中添加本地 Gateway provider：

```toml
model = "local-test"
model_provider = "xtuner"

[model_providers.xtuner]
name = "xtuner gateway"
base_url = "http://127.0.0.1:8091/v1"
env_key = "XTUNER_GATEWAY_KEY"
```

配置访问 token：

```bash
export XTUNER_GATEWAY_KEY=dummy
```

### 先用 curl 验证接口

启动 Codex 前，先确认 `/v1/responses` 能直接返回：

```bash
curl http://127.0.0.1:8091/v1/responses \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer dummy' \
  -d '{
    "model": "local-test",
    "input": "Reply with exactly OK"
  }'
```

如果返回状态为 `completed`，且 `output` 中包含模型回复，说明 Responses 接口基础链路可用。

### 验证普通对话

启动 Codex 后发送：

```text
你好
```

如果 Codex 能收到中文回复，说明客户端能够通过本地 Gateway 完成基础对话。

### 验证工具调用

继续发送以下 prompt：

```text
Use your tools to list the top-level files and directories in the current repository.
Do not explain your plan.
Do not answer from memory.
If you cannot access tools, reply exactly: NO_TOOLS
```

如果 Codex 返回了仓库顶层文件和目录，而不是 `NO_TOOLS`，说明 Responses 接口下的工具调用链路可用。

## 验证 OpenAI Chat Completions 接口

除了真实 Agent 客户端，也可以使用 OpenAI Python SDK 验证 `/v1/chat/completions`。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8091/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="local-test",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Reply with exactly: OK"},
    ],
    max_tokens=32,
    temperature=0,
)

print(resp.choices[0].message.content)
```

如果输出包含 `OK`，说明 Chat Completions 接口基础链路可用。

## 检查 capture 日志

联调过程中建议同步检查 Gateway 的 `capture_folder` 输出。重点确认每条记录中是否包含：

- `source_protocol`：请求来源协议，例如 `anthropic_messages` 或 `openai_responses`。
- `internal_messages`：Gateway 转换后发送给 Rollout 的内部消息。
- `output_messages` 或 `output_text`：模型输出转换回客户端协议后的结果。
- `rollout_tools` 和 `rollout_tool_choice`：工具定义和工具选择策略。
- `request_id`：用于串联客户端请求、Gateway 记录和 Rollout 结果。

这些字段能帮助定位问题出在客户端请求、协议适配、Rollout 生成还是响应转换阶段。

## 常见问题

### 客户端请求超时

先检查 Gateway 是否仍在运行，并适当增大客户端超时时间。Claude Code 可设置：

```bash
export API_TIMEOUT_MS=600000
```

同时检查 Rollout Controller 是否收到请求，以及推理服务是否有可用并发。

### 客户端上下文过长

真实 Agent 客户端会注入系统提示词、工具 schema 和历史消息。如果请求被截断或报 context length 相关错误，需要增大 Gateway 和推理后端的上下文长度。

### 工具调用没有触发

先使用本文中的工具调用 prompt 做最小复现，再检查 `capture_folder` 中是否记录了工具定义。如果 `rollout_tools` 为空，问题通常出在客户端请求到 Gateway 的协议适配阶段；如果工具定义存在但没有工具调用结果，需要继续检查模型输出和 Agent 客户端的工具执行日志。
