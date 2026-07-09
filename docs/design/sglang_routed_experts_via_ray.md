# Support SGLang routed experts via Ray

## 1. 背景

XTuner 的 MoE RL 训练需要拿到 rollout 阶段的 routed experts 信息，用于后续 trainer
侧构造 `rollout_routed_experts`。在 SGLang rollout 后端中，`routed_experts` 由
SGLang server 写入 HTTP response 的 `meta_info["routed_experts"]`。

SGLang 0.5.13 的原始实现会把整个 routed experts tensor 转成 bytes，再通过 base64
字符串放进 HTTP JSON response。这个方案实现简单，但在长序列、MoE 层数较多、每 token
top-k experts 较多时会带来明显问题：

- HTTP response 体积被 base64 放大。
- JSON 编解码和 bytes 拷贝成本高。
- routed experts 本来只需要在 Ray 集群内传递，不需要真正走 HTTP 传输大数组。

本次改动把 SGLang 侧返回内容从“大数组 base64 字符串”改成“Ray shared store key”，再由
XTuner rollout worker 通过 Ray 取回真实 ndarray，并重新放入 Ray object store 交给 trainer
worker 消费。

## 2. 目标

- 避免通过 HTTP JSON 传输完整 routed experts 数组。
- 保持 XTuner 现有 `RolloutState.routed_experts` 和 trainer 消费路径不变。
- 只对当前 XTuner 默认的 SGLang 0.5.13 增加一个小 patch，不 fork SGLang 源码。
- 在 Ray shared store 不可用时保留 base64 fallback，兼容未打 patch 或异常场景。
- 让启用方式继续由 `enable_return_routed_experts` 和样本级
  `return_routed_experts` 共同控制。

## 3. 非目标

- 不改变 trainer 侧 routed experts loss 或 sequence packing 逻辑。
- 不改变 `RolloutState` / `SequenceContext` 的数据结构定义。
- 不支持所有 SGLang 版本。当前 patch 明确基于干净的 `sglang==0.5.13` wheel 生成。
- 不改变非 SGLang rollout 后端，例如 LMDeploy。

## 4. 总体设计

整体路径分成三段：

```text
SGLang scheduler / tokenizer manager
  -> routed_experts tensor
  -> SGLang _RoutedExpertsSharedStore.put()
  -> Ray object store
  -> HTTP meta_info["routed_experts"] 返回 shared-store key
  -> XTuner SGLangWorker._decode_routed_experts()
  -> shared_store.get(key)
  -> XTuner ray.put(np.asarray(...))
  -> RolloutState.routed_experts 保存 Ray ObjectRef
  -> trainer worker ray.get(...) 消费
```

关键设计点是：HTTP 仍然负责传递控制信息和小的 metadata，真实 routed experts 数据只在 Ray
object store 中传递。

## 5. SGLang patch 设计

Patch 文件位于：

```text
patch/sglang-0.5.13-routed-experts-ray.patch
```

Patch 修改 SGLang 的：

```text
sglang/srt/managers/tokenizer_manager.py
```

### 5.1 Shared store actor

SGLang patch 新增 `_RoutedExpertsSharedStore`：

- actor 名称：`shared_store`
- Ray namespace：`sglang`
- lifetime：`detached`
- actor 内部维护 `key -> ObjectRef` 字典

`put(data)` 的行为：

1. 调用 `ray.put(data)`，把 ndarray 放进 Ray object store。
2. 使用 `ObjectRef.hex()` 作为 key。
3. 在 actor 内保存 `key -> ref`。
4. 返回 key 给 tokenizer manager。

`get(key)` 的行为：

1. 从 actor 字典中 `pop(key)`。
2. 对 ref 调用 `ray.get(ref)`。
3. 返回真实数据给 XTuner rollout worker。

这里使用 `pop` 是为了让 shared store 中的临时引用在成功消费后立即移除，避免 routed experts
长期挂在 detached actor 上。

### 5.2 懒加载 actor

Patch 新增 `_lazy_get_routed_experts_shared_store()`：

- 如果当前进程还没有 Ray client，则调用 `ray.init(address="auto", ignore_reinit_error=True)`。
- 优先通过 `ray.get_actor("shared_store", namespace="sglang")` 复用已存在 actor。
- 如果 actor 不存在，则创建 detached actor。
- 如果并发创建触发 `ActorAlreadyExistsError`，再回退到 `get_actor`。

这样可以支持多个 SGLang tokenizer manager 进程并发启动，最终共享同一个 Ray named actor。

### 5.3 编码逻辑

SGLang 原逻辑：

```text
routed_experts tensor -> numpy bytes -> base64 string -> HTTP response
```

Patch 后逻辑：

```text
routed_experts tensor -> numpy ndarray -> Ray shared store -> key string -> HTTP response
```

如果 Ray actor 写入失败，patch 会记录异常日志，并回退到原始 base64 编码。这保证 XTuner
仍可通过 fallback 路径解析旧格式。

## 6. XTuner 侧设计

XTuner 改动位于：

```text
xtuner/v1/rl/rollout/sglang.py
```

### 6.1 启用请求参数

`SGLangWorker` 只有在下面条件同时满足时，才向 SGLang 请求 routed experts：

- `RolloutConfig.enable_return_routed_experts == True`
- `sample_params.return_routed_experts == True`
- 当前 `RolloutState.extra_fields` 没有设置 `disable_routed_experts`

请求会携带：

```python
return_routed_experts = True
```

SGLang server 启动参数也会在 `enable_return_routed_experts` 打开时传入：

```python
init_kwargs["enable_return_routed_experts"] = True
```

这保持了全局开关和样本级开关的双层控制。

### 6.2 解码返回值

`SGLangWorker._decode_routed_experts()` 处理三种情况：

1. `routed_experts` 是字符串，且可以作为 Ray shared store key 解析。
2. `routed_experts` 是字符串，但 shared store 解析失败，按 base64 fallback 解析。
3. `routed_experts` 已经是数组类对象，直接转成 ndarray。

主路径：

```text
key string
  -> ray.get_actor("shared_store", namespace="sglang")
  -> actor.get.remote(key)
  -> np.asarray(...)
  -> ray.put(...)
  -> ObjectRef
```

fallback 路径：

```text
base64 string
  -> base64.b64decode(...)
  -> np.frombuffer(..., dtype=np.int32)
  -> reshape(-1, num_hidden_layers, num_experts_per_tok)
  -> copy()
```

`RolloutWorker._safe_handle_response()` 会保证最终写入 `RolloutState.routed_experts` 的是
`ray.ObjectRef`，除非返回值为空或请求失败。

### 6.3 Trainer 消费

Trainer 侧已有 `_add_rollout_routed_experts()` 逻辑：

- 对 `ray.ObjectRef` 调用 `ray.get()` 拿到 ndarray。
- 转成 `torch.long`。
- reshape 为 `[-1, num_hidden_layers, num_experts_per_tok]`。
- 拼接到 `SequenceContext.rollout_routed_experts`。
- 在合适时机通过 `ray.internal.free(..., local_only=False)` 释放 object refs。

因此本次改动不需要修改 trainer 数据结构，也不需要新增 train worker 配置。

## 7. 配置与使用

运行时需要满足：

- 使用 SGLang rollout backend。
- SGLang 版本为 `0.5.13`，并应用 `patch/sglang-0.5.13-routed-experts-ray.patch`。
- 使用能返回 routed experts 的 MoE 模型。
- 配置中启用 `enable_return_routed_experts`。

示例配置会从环境变量读取：

```bash
export ENABLE_RETURN_ROUTED_EXPERTS=1
```

然后在 rollout config 中设置：

```python
enable_return_routed_experts=(enable_return_routed_experts == "1")
```

Patch 应用方式见：

```text
patch/README.md
```

## 8. 兼容性与失败模式

### 8.1 SGLang 未打 patch

如果 SGLang 未打 patch，但仍返回旧的 base64 字符串，XTuner 会尝试从 Ray shared store 取
key，失败后走 base64 fallback。

这能兼容旧返回格式，但不能解决 HTTP 传大数组的开销问题。

### 8.2 SGLang 版本不匹配

Patch 基于 `sglang==0.5.13` 生成。对其他版本执行 `git apply --check` 可能失败。

如果失败，应先确认：

```bash
python -c "import sglang; print(sglang.__version__)"
```

如果版本不是 `0.5.13`，需要重新基于对应版本生成 patch。

### 8.3 Ray actor 不可用

SGLang patch 写 shared store 失败时会回退到 base64。XTuner 读取 shared store 失败时也会回退到
base64 解析。

如果 SGLang 已经返回 key，但 actor 在 XTuner 读取前异常退出，则 fallback 不会成功，因为 key
不是 base64。这种情况通常表示 Ray 集群或 SGLang tokenizer manager 异常，应从 Ray actor 状态和
SGLang 日志排查。

### 8.4 Key 只能消费一次

`_RoutedExpertsSharedStore.get()` 使用 `pop(key)`，同一个 key 被成功读取后不能再次读取。

当前 XTuner 设计中，每个 response 的 routed experts 只由对应 rollout worker 读取一次，然后
rollout worker 再通过 `ray.put` 生成 trainer 侧共享的 `ObjectRef`。因此单次消费符合当前数据流。

如果未来引入多个 XTuner consumer 直接读取 SGLang shared store，需要把 shared store 的释放策略
改成引用计数或显式清理。

## 9. 验证方式

### 9.1 Patch apply 验证

在 XTuner checkout 根目录执行：

```bash
# Optional: activate the environment that contains sglang==0.5.13.
source /path/to/venv/bin/activate
SITE_PACKAGES=$(python -c "import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parents[1])")
PATCH_FILE=$(python -c "import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" patch/sglang-0.5.13-routed-experts-ray.patch "$SITE_PACKAGES")
cd "$SITE_PACKAGES"
git apply --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
git apply -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```

如果当前环境已经应用过 patch，正向 `--check` 会失败，反向检查应成功：

```bash
git apply -R --check -p2 --include='sglang/**' --exclude='*' "$PATCH_FILE"
```

### 9.2 静态验证

应用 patch 后，SGLang 的 `tokenizer_manager.py` 中应能找到：

```text
_RoutedExpertsSharedStore
_lazy_get_routed_experts_shared_store
_encode_routed_experts
```

XTuner 的 `xtuner/v1/rl/rollout/sglang.py` 中应能找到：

```text
SHARED_STORE = "shared_store"
SHARED_STORE_NAMESPACE = "sglang"
ray.get_actor(SHARED_STORE, namespace=SHARED_STORE_NAMESPACE)
```

### 9.3 运行时验证

运行 RL 任务后检查：

- SGLang server 启动参数中包含 `enable_return_routed_experts=True`。
- Ray actor 列表中存在 `shared_store`，namespace 为 `sglang`。
- XTuner rollout response 中 `meta_info["routed_experts"]` 是 key 字符串，而不是大 base64 字符串。
- Trainer worker 能成功执行 `torch.as_tensor(..., dtype=torch.long)` 并构造
  `seq_ctx.rollout_routed_experts`。
- 训练日志没有 `missing routed_experts` 或 shared store 解析失败后的异常。