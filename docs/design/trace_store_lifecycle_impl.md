# Trace Store Lifecycle Implementation Design

## 1. 目标和边界

本文是 `trace_store_lifecycle_simple.md` 的实现讨论稿，用于把 Rollout Trace Store 的生命周期设计落到
`xtuner/v1/rl/rollout/trace_store.py`。

本文件记录当前实现决策。后续每次讨论后，直接更新本文，而不是立即修改代码。

第一版实现边界：

1. 只设计 Trace Store actor 内部的 session 级生命周期管理。
2. 核心只围绕 `state`。
3. 不引入 `created_at`；`updated_at` 保留，但从 `RolloutTraceStore.updated_at` 下沉为每个 session / `Trie` 自己的属性。
4. 不引入 trainer rank / materialize 计数。trainer 自己汇总消费状态，并在确认不再依赖 Store 后调用 Trace Store 状态转换 API。
5. 不引入新的 staged / committed object registry。
6. routed experts Ray `ObjectRef` 由 `RolloutTraceStore.objects: dict[str, ray.ObjectRef]` 统一持有；`TokenizedSegment.expert_key` 中只保留对应 object key。
7. 一个 session 只记录一个 routed experts object key；释放 session 时删除这个 key 对应的 object ref。

非目标：

1. 不修改 ReplayBuffer / producer 的采样和重试策略。
2. 不实现 failed prefix resume。
3. 不支持失败重试复用旧 `session_id`。
4. 不做节点级 partial release。
5. 不设计 trainer 内部多 rank ack 协议。

## 2. 当前实现理解

旧实现里 `RolloutTraceStore` 主要有三组 actor 级状态：

```python
self.sessions: Dict[str, Trie]
self.objects: Dict[str, ray.ObjectRef]
self.updated_at: Dict[str, float]
```

其中 `self.objects` 已有查询接口，但没有完整写入和释放链路；`self.updated_at` 与 `sessions` 分离。新实现保留 `self.objects`，但把它收敛为正式的 routed experts object registry；`updated_at` 下沉到 `Trie`。

当前实际使用路径里，`sessions` 是核心：

1. `SessionServer.on_request` 根据 `session_id` 在 store 中做 prefix `search`。
2. prompt delta 会写入对应 session 的 `Trie`。
3. `SessionServer.on_response` 会把 assistant output 写入同一棵 `Trie`。
4. routed experts 通过 `ray.put(...)` 后得到 `ObjectRef`。
5. 旧实现存在把 `ObjectRef` 直接放进 `TokenizedSegment.expert_key` 的路径；新实现禁止这种写法。

新的实现方向：

1. `RolloutTraceStore.sessions` 继续保持 `dict[str, Trie]`。
2. `Trie` 不改名，作为单条 session 的数据和生命周期承载对象。
3. actor 级 `self.objects` 保留，作为唯一持有 routed experts `ObjectRef` 的 registry。
4. actor 级 `self.updated_at` 下沉为 `Trie.updated_at`。
5. routed experts 的实际 `ray.ObjectRef` 存到 `RolloutTraceStore.objects[expert_key]`。
6. `TokenizedSegment.expert_key` 保存 object key，而不是直接保存 routed experts 的 `ObjectRef`。
7. session 删除时，必须按该 session 记录的 object key 从 `self.objects` 中删除并释放 ref。

这样保持原有 actor 级 object lookup 形态，同时要求所有 object 写入、替换和释放都经过 `RolloutTraceStore` 的 helper，避免 500M 级别 routed experts object 因漏删 key 而常驻。

## 3. 核心状态模型

新增 `TraceState`，与上游设计保持一致：

```python
class TraceState(str, Enum):
    ROLLOUT_RUNNING = "RolloutRunning"
    ROLLOUT_FINISHED = "RolloutFinished"
    TRAIN_RUNNING = "TrainRunning"
    TRAIN_FINISHED = "TrainFinished"
    TO_BE_RELEASED = "ToBeReleased"
    RELEASED = "Released"
```

状态含义：

| 状态 | 含义 | 能否物理释放 |
| --- | --- | --- |
| `RolloutRunning` | rollout 还在运行、继续写入，或 `ABORTED + enable_partial_rollout=True` 等待续跑 | 不能 |
| `RolloutFinished` | rollout 已完成且未被过滤，等待训练侧 export | 不能 |
| `TrainRunning` | 训练侧已 export，仍可能依赖 Store 中的 refs | 不能 |
| `TrainFinished` | trainer 已确认不再依赖 Store | 不能；下一步进入 `ToBeReleased` |
| `ToBeReleased` | 已不再服务 rollout 或 training，等待统一释放 | 可以 |
| `Released` | 概念终态；实现上表现为 session metadata 已删除 | 终态 |

## 4. Trie 作为 session 数据结构

不新增 `TraceSession` 包装类。`Trie` 名字保持不变，在 `Trie` 内新增 session 级属性：

```python
class Trie:
    root: TreeNode
    state: TraceState = TraceState.ROLLOUT_RUNNING
    expert_key: str | None = None
    updated_at: float
```

对应地，`RolloutTraceStore.sessions` 保持：

```python
self.sessions: Dict[str, Trie]
```

不调整为 `Dict[str, TraceSession]`。

字段含义：

1. `root`：原有 prefix tree 根节点。
2. `state`：该 session 当前生命周期状态。
3. `expert_key`：该 session 唯一 routed experts object key。
4. `updated_at`：该 session 最近一次写入、状态转换或 routed experts object key 变更时间。

实现约束：

1. `get_or_create(session_id)` 仍返回 `Trie`。
2. 现有 `keys` / `insert` / `search` 的外部接口尽量保持兼容。
3. `Trie` 可以新增内部 helper，例如 `touch()`。

### 4.1 routed experts object 存储

`TokenizedSegment.expert_key` 继续保留这个字段名，但语义调整为 object key。

推荐实现方式：

```python
def make_expert_key(session_id: str) -> str:
    return f"{session_id}:routed_experts"

def insert(
    session_id: str,
    key: str,
    value: TokenizedSegment,
    routed_experts: ray.ObjectRef | None = None,
) -> None:
    if routed_experts is not None:
        expert_key = make_expert_key(session_id)
        self.objects[expert_key] = routed_experts
        value.expert_key = expert_key
        trie.expert_key = expert_key
    trie.insert(key, value)
```

写入 `TokenizedSegment` 前后需要满足：

1. routed experts 的实际 `ObjectRef` 在 `RolloutTraceStore.objects` 中。
2. `TokenizedSegment.expert_key` 是能找回该 `ObjectRef` 的 key。
3. `TokenizedSegment.expert_key` 只允许 `str | None`，禁止写入 `ray.ObjectRef`。
4. object key 是 `TokenizedSegment` 里保存的轻量索引，用来从 session 找回实际 `ray.ObjectRef`。它不是新的业务协议，只是避免把 `ObjectRef` 直接塞进 trie node value。
5. object key 从 `session_id` 派生，一个 session 只有一个 routed experts object key。
6. `TokenizedSegment.expert_key` 类型语义为 `str | None`。`None` 表示该 segment 没有 routed experts object。
7. 一个 session 只有一个 object key。
8. 调用方通过 `insert(..., routed_experts=obj_ref)` 写入真实 ref；`insert` 负责生成 `expert_key` 并写回 `TokenizedSegment`。
9. 第一版假设同一个 expert key 不会重复写入；`insert` 覆盖 dict 时不负责释放旧 ref。

## 5. 状态转换 API 草案

### 5.1 查询状态

```python
def get_state(session_id: str) -> dict | None:
    ...
```

返回示例：

```python
{
    "session_id": session_id,
    "state": "RolloutRunning",
    "updated_at": 1234567890.0,
    "has_object_ref": True,
}
```

session 不存在时返回 `None`。

### 5.2 rollout 状态上报

```python
def mark_rollout_status(
    session_id: str,
    status: Status,
    *,
    enable_partial_rollout: bool = False,
) -> str:
    ...
```

状态映射：

| rollout status | 条件 | Trace Store 状态 |
| --- | --- | --- |
| `COMPLETED` | - | `RolloutFinished` |
| `ABORTED` | `enable_partial_rollout=True` | 保持 `RolloutRunning` |
| `ABORTED` | `enable_partial_rollout=False` | `ToBeReleased` |
| `FAILED` | - | `ToBeReleased` |
| `FILTERED` | - | `ToBeReleased` |
| `EXPIRED` | - | `ToBeReleased` |
| `INIT` | - | 保持 `RolloutRunning` |
| `ARCHIVED` | - | 第一版暂不处理，传入 `mark_rollout_status` 时抛错 |

进入 `ToBeReleased` 后立即调用 `_maybe_release(session_id)`。

状态约束：

1. `mark_rollout_status` 不创建 session。
2. `COMPLETED` 和 `ABORTED + enable_partial_rollout=True` 必须要求 session 已存在且当前为 `RolloutRunning`；否则抛 `KeyError` 或 `RuntimeError`。
3. `FAILED` / `FILTERED` / `EXPIRED` / `ABORTED + enable_partial_rollout=False` 是 release-like 事件；session 不存在时返回 `Released`，不创建空 session。
4. 如果 session 已经不是 `RolloutRunning`，新的 rollout status 不能覆盖后续状态，必须抛 `RuntimeError`。

### 5.3 rollout 放弃和 commit 失败

```python
def mark_commit_failed(session_id: str) -> str:
    ...

def mark_rollout_discarded(session_id: str) -> str:
    ...
```

两者都进入 `ToBeReleased`，然后调用 `_maybe_release(session_id)`。

`mark_rollout_discarded` 用于表达 skipped、timeout、final cancelled、旧 session 被新 session 替换等语义事件。

状态约束：

1. `mark_commit_failed` 只允许从 `RolloutRunning` 进入 `ToBeReleased`。
2. `mark_rollout_discarded` 允许从 `RolloutRunning` 或 `RolloutFinished` 进入 `ToBeReleased`。
3. 两者都是 release-like 事件；session 不存在时返回 `Released`，不创建空 session。

### 5.4 training export

```python
def export_training_trace(session_id: str, prompt_text: str) -> dict:
    ...
```

成功条件：

1. session 存在。
2. session 处于 `RolloutFinished`。
3. `prompt_text` 能完整命中 session trie。
4. token-level 字段可以组成训练 trace。

返回的 `routed_experts` 是 session 级 `expert_key` 或 `None`。第一版不再返回每个 segment 的 routed experts key
列表，因为 routed experts 在 Store 中按 session 级别维护。

成功后：

```text
RolloutFinished -> TrainRunning
```

失败后：

```text
RolloutFinished -> ToBeReleased -> Released
```

失败时进入 `ToBeReleased` 并继续抛出 `ValueError`。

约束：`export_training_trace` 必须要求 session 已经是 `RolloutFinished`。不能为了兼容当前代码从 `RolloutRunning`
直接导出；调用侧必须先通过 rollout 完成事件把 session 推进到 `RolloutFinished`。

失败行为：

1. session 不存在时抛 `KeyError`，不能自动创建 session。
2. session 不是 `RolloutFinished` 时抛 `RuntimeError`，不能导出训练 trace。
3. session 是 `RolloutFinished` 但 trace 不完整时，先进入 `ToBeReleased`，再抛 `ValueError`。

### 5.5 trainer 完成或放弃

```python
def mark_train_finished(session_id: str) -> str:
    ...

def mark_train_abandoned(session_id: str) -> str:
    ...
```

`mark_train_finished` 由 trainer 在确认所有训练消费者都不再依赖 Store 后调用：

```text
TrainRunning -> TrainFinished -> ToBeReleased -> Released
```

`mark_train_abandoned` 由 trainer 在取消、不可恢复 materialize 失败、batch 被替换等情况下调用。调用前提仍然是
trainer 已经确认不会再访问 Store：

```text
TrainRunning -> ToBeReleased -> Released
```

Trace Store 不维护 rank 级 materialize 状态。

状态约束：

1. `mark_train_finished` 只允许从 `TrainRunning` 调用。
2. `mark_train_abandoned` 只允许从 `TrainRunning` 调用。
3. 两者都是训练侧消费结束事件；session 不存在时返回 `Released`，不创建空 session，用于兼容重复上报或释放后的迟到事件。

### 5.6 不提供外部 release API

外部模块不能直接调用 `release(session_id)`。物理释放只能由 Trace Store 内部在状态进入 `ToBeReleased` 后触发。

当前已有的 actor 方法：

```python
def release(session_id: str):
    ...
```

实现时删除 public actor method `release(session_id)`，改成内部私有方法 `_release_session(session_id)`。对外暴露的 API
只能是语义事件，例如 `mark_rollout_status`、`mark_commit_failed`、`mark_rollout_discarded`、`mark_train_finished`、
`mark_train_abandoned`。

## 6. 状态转换和释放触发

所有状态写入必须通过统一 helper 完成，避免某条路径只改状态但漏掉 release。第一版暂不在 `_set_state`
里实现模块级目标状态转换表；具体语义 API 在调用 `_set_state` 前用自己的入口状态校验表达约束。后续如果状态转换路径继续变多，
可以再引入 `next_state -> allowed_previous_states` 的全局表。

```python
def _set_state(
    self,
    session_id: str,
    next_state: TraceState,
) -> TraceState:
    trie = self.sessions.get(session_id)
    if trie is None:
        raise KeyError(f"Trace session {session_id!r} does not exist.")
    trie.state = next_state
    trie.touch()
    self._maybe_release(session_id)
    return next_state
```

约束：

1. 任何进入 `ToBeReleased` 的 API 都必须走 `_set_state(...)` 或等价 helper。
2. `_set_state` 每次状态更新后都调用 `_maybe_release(session_id)`。
3. `_maybe_release` 内部只在状态为 `ToBeReleased` 时释放，所以可以在每次状态变更后安全调用。
4. `_set_state` 不能调用 `get_or_create`，状态事件不能创建空 session。
5. 第一版不在 `_set_state` 内实现模块级目标状态转换表；每个语义 API 自己校验允许的入口状态。
6. release-like 事件如果允许 missing session no-op，必须在外层语义 API 中处理，不进入 `_set_state`。
7. 正常路径 `mark_train_finished` 需要先记录 `TrainFinished`，再进入 `ToBeReleased`：

```python
def mark_train_finished(self, session_id: str) -> str:
    if session_id not in self.sessions:
        return TraceState.RELEASED.value
    self._set_state(
        session_id,
        TraceState.TRAIN_FINISHED,
    )
    return self._set_state(
        session_id,
        TraceState.TO_BE_RELEASED,
    ).value
```

8. 异常放弃路径直接进入 `ToBeReleased`：

```python
def mark_train_abandoned(self, session_id: str) -> str:
    if session_id not in self.sessions:
        return TraceState.RELEASED.value
    return self._set_state(
        session_id,
        TraceState.TO_BE_RELEASED,
    ).value
```

### 6.1 合法状态转换表

下表是设计约束，不表示当前代码中有模块级全局转换表。第一版由各语义 API 在入口处校验来源状态。

| 事件/API | 允许来源状态 | 目标状态 |
| --- | --- | --- |
| 首次 `insert` / `search` / `keys` | session 不存在 | `RolloutRunning` |
| `insert` | `RolloutRunning` | `RolloutRunning` |
| `mark_rollout_status(COMPLETED)` | `RolloutRunning` | `RolloutFinished` |
| `mark_rollout_status(ABORTED, enable_partial_rollout=True)` | `RolloutRunning` | `RolloutRunning` |
| `mark_rollout_status(ABORTED, enable_partial_rollout=False)` | `RolloutRunning` | `ToBeReleased` |
| `mark_rollout_status(FAILED/FILTERED/EXPIRED)` | `RolloutRunning` | `ToBeReleased` |
| `mark_commit_failed` | `RolloutRunning` | `ToBeReleased` |
| `mark_rollout_discarded` | `RolloutRunning` / `RolloutFinished` | `ToBeReleased` |
| `export_training_trace` 成功 | `RolloutFinished` | `TrainRunning` |
| `export_training_trace` trace 不完整 | `RolloutFinished` | `ToBeReleased` |
| `mark_train_finished` | `TrainRunning` | `TrainFinished` -> `ToBeReleased` |
| `mark_train_abandoned` | `TrainRunning` | `ToBeReleased` |
| `_maybe_release` | `ToBeReleased` | `Released`，实现上删除 session |

## 7. 释放语义

唯一物理释放入口：

```python
def _maybe_release(session_id: str) -> None:
    trie = self.sessions.get(session_id)
    if trie is None:
        return
    if trie.state != TraceState.TO_BE_RELEASED:
        return
    self._release_session(session_id, trie)
```

`_release_session(session_id, trie)` 是内部物理释放方法，负责：

1. 从 `trie.expert_key` 获取该 session 唯一 routed experts object key。
2. 删除该 key 在 `self.objects` 中对应的 ref，并调用 `_free_ray_refs`。
3. 调用 `trie.release()` 释放 session tree 中其他可能残留的 `ObjectRef`。
4. 从 `self.sessions` 删除 `session_id`。

约束：

1. `_maybe_release` 必须幂等。
2. session 不存在时直接返回。
3. 非 `ToBeReleased` 不释放。
4. 释放粒度是整个 session tree。
5. routed experts object 的释放由 `RolloutTraceStore` 负责，`Trie.release()` 不直接访问 `self.objects`。
6. `Trie.release()` 仍保留对 tree node value 的递归 `_free_ray_refs`，作为防御性清理，避免非 routed experts 字段里仍残留 `ObjectRef`。
7. 删除 session metadata 前必须保证该 session 记录的 object key 已经从 `self.objects` 中删除。
8. `Released` 不需要持久保存；删除 session metadata 就代表已释放。
9. lifecycle 释放只允许 full-session release。现有 `Trie.release(key=None)` 的 `key` 参数需要删除，或保留为非 lifecycle 私有能力；生命周期路径不能做 subtree release。

## 8. 现有方法兼容策略

### 8.1 `insert`

`insert(session_id, key, value, routed_experts=None)` 当前会自动创建 session。

第一版保留该行为：

1. session 不存在时创建 `RolloutRunning` session。
2. session 为 `RolloutRunning` 时允许写入。
3. 如果 `value` 是 `TokenizedSegment`，`value.expert_key` 必须已经是 `str | None`，不能是 `ray.ObjectRef`。
4. routed experts `ObjectRef` 的写入只能走 `insert(..., routed_experts=obj_ref)`。
5. `insert` 不处理旧 `TokenizedSegment` 覆盖后的 object ref 释放；第一版先假设写入路径不会产生需要清理的旧 ref。
6. session 已进入 `RolloutFinished` / `TrainRunning` / `ToBeReleased` 后，`insert` 必须记录 error 日志并跳过写入，避免已完成或待释放 session 被继续污染，同时不打断上层训练/rollout 主流程。

### 8.2 `search` / `keys`

当前 `search` / `keys` 也会通过 `get_or_create` 自动创建 session。第一版继续保留这个兼容行为。

实现约束：

1. session 不存在时创建空 `RolloutRunning` session。
2. session 处于 `RolloutRunning` / `RolloutFinished` / `TrainRunning` 时允许读取。
3. session 处于 `ToBeReleased` 时不应继续读取；实现上 `_maybe_release` 在进入该状态后立即触发释放，故此情形窗口极窄。防御路径记录 error 日志，`keys` 返回 `[]`，`search` 返回 `("", [])`。

### 8.3 `get_objects`

`get_objects(keys)` 的实现从 actor 级 `self.objects` 取对象。

object key 的含义：

1. `self.objects[object_key]` 保存实际 `ray.ObjectRef`。
2. `TokenizedSegment.expert_key` 保存 object key 字符串。
3. `export_training_trace` 返回 session 级 routed experts object key 或 `None`。
4. 训练侧如果需要实际 ref，再调用 `get_objects([key])`。
5. `None` 是合法的 routed experts 占位，表示该 session 没有 routed experts object；训练侧不能把 `None` 传给 `get_objects`。

`get_objects` 行为：

1. 只接受非空字符串 object key。
2. object key 必须存在于 `self.objects`。
3. 缺失 key 必须抛 `KeyError`，不能静默跳过。

## 9. 孤儿 Session TTL 清理

### 9.1 问题背景

rollout worker crash 后，若未调用任何生命周期 API，session 会永远卡在 `RolloutRunning`，`self.objects` 中该 session 引用的 `ray.ObjectRef` 持续占用内存。这类 session 称为**孤儿 session（orphan session）**。

### 9.2 TTL 机制设计

在 `Trie` 中使用 `updated_at` 字段记录最近一次写入或状态变更时间。`RolloutTraceStore` 暴露一个 `gc_stale_sessions(ttl_seconds: float)` 方法，供外部按需调用（例如由 train controller 或 rollout coordinator 周期性触发）：

```python
def gc_stale_sessions(self, ttl_seconds: float) -> list[str]:
    """释放超过 ttl_seconds 未更新且仍处于 RolloutRunning 的孤儿 session。

    Returns:
        list[str]: 被释放的 session_id 列表，用于日志和告警。
    """
    now = time.time()
    stale = [
        sid
        for sid, trie in self.sessions.items()
        if trie.state == TraceState.ROLLOUT_RUNNING
        and (now - trie.updated_at) > ttl_seconds
    ]
    for sid in stale:
        self._set_state(
            sid,
            TraceState.TO_BE_RELEASED,
        )
    return stale
```

### 9.3 约束

1. `gc_stale_sessions` 只清理 `RolloutRunning` 状态的 session，不触碰 `RolloutFinished` / `TrainRunning`（这些状态可能正在被训练侧使用）。
2. TTL 值建议由调用方配置（例如 `ttl_seconds = 600`），Trace Store 不内置 TTL 常量。
3. `gc_stale_sessions` 本身不启动后台线程，不依赖 Ray actor 内部 timer；调用方负责周期性调用。
4. 返回被释放的 session_id 列表，调用方应记录 WARNING 日志，以便排查 worker crash。
5. `gc_stale_sessions` 另提供诊断能力：被观测到频繁有孤儿 session 说明 rollout worker crash 率异常，需告警。

### 9.4 可观测性补充

新增 `list_sessions(state: str | None = None) -> list[dict]` 诊断 API，返回当前所有 session 的状态快照（`session_id`、`state`、`updated_at`、`has_object_ref`）。
`state` 参数可过滤特定状态，例如 `list_sessions(state="RolloutRunning")` 快速定位孤儿 session。用于外部监控和调试，不触发任何状态变更。

## 10. Trainer 接入点

trainer 在 train controller 的训练结束位置调用 Trace Store 状态转换 API：

1. 正常训练消费结束，确认后续不会再访问 Store 后，调用 `mark_train_finished(session_id)`。
2. 训练取消、batch 放弃、不可恢复 materialize 失败，并确认后续不会再访问 Store 后，调用 `mark_train_abandoned(session_id)`。
3. Trace Store 不在 trainer 内部分 rank 维度做判断，只接收 train controller 汇总后的事件。

### 10.1 主流程接入顺序

Trace Store actor 内部 API 完成后，主流程接入按以下顺序推进。

1. Rollout 状态接入：在 `ProduceContext.put_generated_group` 中等待业务过滤、ReplayBuffer 过期处理完成后，按每个 `RolloutState` 的最终 `status` 调用 `mark_rollout_status`。这里是 rollout 结果进入系统状态机的统一入口。
2. Training trace materialize 接入：在 `BaseRLTrainer._prepare_train_data` 中默认对每个 sample 调用 `export_training_trace`。如果 trace 返回 routed experts object key，trainer 只保留 `get_objects([key])` 这个 Ray ref，不在 trainer 里 `ray.get` 实体；真正的 routed experts tensor 只在 `TrainingWorker` 内部消费时 `ray.get`。ReplayBuffer 中的轻量 `RolloutState` 可以不携带 token-heavy 字段，进入 `_prepare_train_data` 后由 Trace Store 回填 `prompt_ids` / `response_ids` / `response_mask` / `logprobs` / `routed_experts`，再复用原 batch 构造逻辑。
3. 训练完成释放接入：在 `BaseRLTrainer._train_one_batch` 调用 `self.train_controller.fit(...)` 成功返回后，对本 batch 中已 export 的 session 调用 `mark_train_finished`。
4. 训练放弃释放接入：如果 export 成功后 `_prepare_train_data`、packing 或 `train_controller.fit` 出现不可恢复异常，并且确认后续不会再访问 Store，则对已 export 的 session 调用 `mark_train_abandoned`。
5. 孤儿 session GC 接入：trainer 主循环按配置频率调用 `gc_stale_sessions(ttl_seconds)`，并对返回的 stale session_id 记录 warning 日志。

### 10.2 接入约束

1. Trace Store 状态必须以最终进入 ReplayBuffer 或最终被 trainer 消费的状态为准，不能在 rollout worker 单次 generate 内提前标最终状态。
2. `export_training_trace` 只能由 trainer/data prepare 层调用一次，成功后 session 进入 `TrainRunning`。
3. TrainingWorker rank 内部不直接调用 Trace Store lifecycle API，也不分别 release session。
4. `mark_train_finished` 必须在所有 TrainingWorker 完成消费后调用。
5. `mark_train_abandoned` 只用于确认后续不会再访问 Store 的失败路径。
6. `gc_stale_sessions` 只清理 `RolloutRunning`，不能作为训练完成释放的替代机制。

### 10.3 Trace Store 训练样本的轻量字段契约

第一版先不新增独立训练样本结构，仍复用 `RolloutState` 进入 ReplayBuffer 和 trainer。ReplayBuffer 中的 `RolloutState` 保持轻量，token 级训练数据留在 `RolloutTraceStore`：

1. `RolloutState.session_uid` 是 Trace Store 的 `session_id`。
2. trainer 默认从 Trace Store materialize 训练数据，不再用 `extra_fields["use_trace_store"]` 在 trainer 内部分支。
3. `RolloutState.extra_fields["trace_store_prompt_text"]` 是可选字段；如果调用方能拿到与 Trace Store key 完全一致的最终 prompt text，应写入这里，trainer 优先使用它。
4. trainer 中的 group 是 advantage group，不是 Trace Store session group；group 内每个 rollout sample 都是一次独立 request，必须有独立 `session_uid`。
5. `RolloutState.response_ids` / `logprobs` / `routed_experts` 在 ReplayBuffer 中可以保持 `None`，不能写成 `session_id` 或 object key。
6. `_prepare_train_data` 从 Trace Store 导出后，把 token 字段回填到当前 `RolloutState`，让后续 batch 构造继续按 `RolloutState` 字段读取。
7. trainer 不 `ray.get` routed experts 实体，只把 routed experts 的 Ray ref 放进 `SequenceContext.rollout_routed_experts`。
8. `RolloutState.response`、`message`、`tools`、`tool_calls` 仍可保留，供 reward、日志、prompt_text 重建或调试使用。
9. 如果 `trace_store_prompt_text` 不存在，trainer 可以用 `message + response/tool_calls + tools` 重新渲染完整 prompt text；这要求调用方保证这些轻量字段和 SessionServer 写入 Trace Store 时使用的 chat template 一致。

这只是过渡方案。后续如果 Trace Store 成为唯一训练数据来源，应新增专门的轻量训练 envelope，替代在 `RolloutState.extra_fields` 中携带协议字段。

## 11. 第一版测试计划

设计稳定并进入代码实现后，新增 `tests/rl/test_trace_store_lifecycle.py`。

测试覆盖：

1. first insert 创建 `RolloutRunning` session。
2. `COMPLETED` 进入 `RolloutFinished`。
3. `FAILED` / `FILTERED` / `EXPIRED` 进入 `ToBeReleased` 并释放。
4. `ABORTED + enable_partial_rollout=True` 保持 `RolloutRunning`，已有 trie 内容仍可 search。
5. `ABORTED + enable_partial_rollout=False` 释放 session。
6. `export_training_trace` 成功后进入 `TrainRunning`。
7. `export_training_trace` prefix 不完整时释放。
9. `mark_train_finished` 触发 `TrainFinished -> ToBeReleased -> Released`。
10. `mark_train_abandoned` 触发释放。
11. routed experts `ObjectRef` 写入后进入 `RolloutTraceStore.objects`，`TokenizedSegment.expert_key` 保存 object key。
12. `get_objects` 能从 object key 定位真实 `ObjectRef`。
13. `routed_experts` 允许 `None`，但 `get_objects([None])` 或缺失 key 必须报错。
14. session lifecycle release 会删除该 session 记录的 object key，并释放对应 ref。
15. 非法状态转换报错，例如 `RolloutRunning -> TrainFinished`。
16. `_maybe_release` 对不存在 session 幂等 no-op；release-like 语义事件对已释放 session 返回 `Released`。
17. `search` / `keys` 对不存在 session 自动创建空 `RolloutRunning` session。
18. session 处于 `ToBeReleased` 时调用 `search` / `keys` 记录 error 日志并返回空结果。
19. `gc_stale_sessions(ttl)` 释放超过 TTL 的 `RolloutRunning` session，不影响 `RolloutFinished` / `TrainRunning` session。
20. `gc_stale_sessions` 返回被释放的 session_id 列表。
21. `list_sessions()` 返回所有 session 快照；`list_sessions(state="RolloutRunning")` 过滤指定状态。

建议验证命令：

```bash
python -m unittest tests.rl.test_trace_store_lifecycle
python -m compileall -q xtuner/v1/rl/rollout/trace_store.py tests/rl/test_trace_store_lifecycle.py
```
