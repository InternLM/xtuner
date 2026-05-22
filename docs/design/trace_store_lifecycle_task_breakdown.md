# Trace Store Lifecycle Task Breakdown

本文记录 `trace_store_lifecycle_impl.md` 对应的功能拆分和开发流程。后续实现必须按小功能逐个推进，每个小功能都经过代码 review、测试 review 和测试执行后，再进入下一个功能。

## 1. 固定开发流程

每个小功能按以下顺序执行：

1. 开发该功能的最小代码改动。
2. 提交给用户 review 代码。
3. 代码 review 通过后，编写该功能对应测试。
4. 提交给用户 review 测试。
5. 测试 review 通过后，执行测试并汇报结果。

约束：

1. 每轮只开发一个小功能。
2. 不提前实现后续功能。
3. 不提前写后续功能测试。
4. review 未完成前不进入下一阶段。
5. 每个小功能的测试优先放在 `tests/rl/test_trace_store_lifecycle.py`，除非该功能必须覆盖现有调用方。

## 2. 功能拆分顺序

### 2.1 Trie session metadata 基础

开发内容：

1. 新增 `TraceState`。
2. 给 `Trie` 增加 `state`、`updated_at`。
3. 增加 `Trie.touch()`。
4. 增加 `RolloutTraceStore.get_state(session_id)`。

不做：

1. 不实现状态转换 API。
2. 不实现 release 行为变化。
3. 不改 `insert` / `search` / `keys` 语义。
4. 不实现 routed experts object registry 写入链路。

代码 review 关注点：

1. `Trie` 名字和 `self.sessions: dict[str, Trie]` 不变。
2. `state` 默认是 `RolloutRunning`。
3. `updated_at` 只用于观测。

测试范围：

1. 新 session 默认 `RolloutRunning`。
2. `updated_at` 初始化存在，`touch()` 后更新。
3. `get_state` 对存在 session 返回状态快照。
4. `get_state` 对不存在 session 返回 `None`。

### 2.2 actor 级 routed experts object registry

开发内容：

1. 恢复并正式接通 `RolloutTraceStore.objects: dict[str, ray.ObjectRef]`。
2. 实现 object key 生成 helper，object key 从 `session_id` 派生。
3. 调整 `insert` 入参为 `insert(session_id, key: str, value: TokenizedSegment, routed_experts: ray.ObjectRef | None = None)`。
4. `insert` 负责在 `routed_experts` 存在时生成 `expert_key`、写入 `self.objects`，并把字符串 key 写回 `TokenizedSegment.expert_key`。
5. `insert` 不再接收或兼容 `TokenizedSegment.expert_key` 为 `ray.ObjectRef` 的情况。
6. `insert` 不处理旧 `TokenizedSegment` 覆盖后的 object ref 释放。
7. `get_objects(keys)` 从 `self.objects` 读取 refs；缺失 key 必须抛 `KeyError`。
8. session release 时删除该 session 记录的 object key，并释放对应 ref。
9. 调整 `SessionServer` 写入路径：把 routed experts ref 作为 `insert` 的独立参数传入。

不做：

1. 不实现生命周期状态转换。
2. 不改变 `export_training_trace` 的状态要求。
3. 不实现 ref_count、owner map 或跨 session object 复用。
4. 不实现 overwrite / 同 key 重复 put 的旧 ref 清理。

代码 review 关注点：

1. `TokenizedSegment.expert_key` 只允许 `str | None`，不能保存真实 `ray.ObjectRef`。
2. `self.objects` 是唯一持有 routed experts `ObjectRef` 的地方。
3. object 写入和删除必须集中在 `RolloutTraceStore` 中。
4. `TokenizedSegment.expert_key` 的运行时语义是 `str | None`。
5. `get_objects` 缺失 key 必须抛 `KeyError`，不能静默跳过。

测试范围：

1. `insert(..., routed_experts=obj_ref)` 后，`TokenizedSegment.expert_key` 是字符串 key。
2. `get_objects([object_key])` 返回该 object key 对应的 ref。
3. `get_objects([None])` 报错。
4. `get_objects(["missing"])` 报错。
5. 删除 session 时释放该 session 记录的 ref。
6. `TokenizedSegment.expert_key` 传入 `ObjectRef` 时构造失败。

### 2.3 状态转换核心 helper

开发内容：

1. 实现 `_set_state(session_id, next_state)`。
2. 实现 `_maybe_release(session_id)`。
3. 实现内部 `_release_session(session_id, trie)`。
4. 删除 public actor method `release(session_id)`，或改成内部不可外部调用的私有方法。

不做：

1. 不实现 rollout / trainer 语义 API。
2. 不改 `export_training_trace`。

代码 review 关注点：

1. `_set_state` 不能调用 `get_or_create`。
2. missing session 在 `_set_state` 中必须抛 `KeyError`。
3. `_set_state` 每次状态更新后调用 `_maybe_release`。
4. `_release_session` 必须释放 trie tree 和该 session 记录的 object ref，再删除 `self.sessions[session_id]`。
5. 第一版不实现模块级目标状态转换表；语义 API 自己校验允许的入口状态。

测试范围：

1. `_set_state` 更新状态成功。
2. missing session 抛 `KeyError`。
3. `_maybe_release` 对 missing session 幂等 no-op。
4. `ToBeReleased` 触发物理删除 session。

### 2.4 Rollout 侧语义 API

开发内容：

1. 实现 `mark_rollout_status`。
2. 实现 `mark_commit_failed`。
3. 实现 `mark_rollout_discarded`。

不做：

1. 不接入 producer / replay buffer。
2. 不修改 rollout worker。

代码 review 关注点：

1. `mark_rollout_status` 不创建 session。
2. `COMPLETED` 只能从 `RolloutRunning` 到 `RolloutFinished`。
3. `ABORTED + enable_partial_rollout=True` 保持 `RolloutRunning`。
4. failed / filtered / expired / aborted without partial 进入 `ToBeReleased` 并释放。
5. release-like missing session 返回 `Released`。

测试范围：

1. `COMPLETED` 进入 `RolloutFinished`。
2. `FILTERED` 释放。
3. `FAILED` / `EXPIRED` 释放。
4. `ABORTED + enable_partial_rollout=True` 保持 `RolloutRunning`。
5. `ABORTED + enable_partial_rollout=False` 释放。
6. 非 `RolloutRunning` 收到 rollout status 报错。
7. release-like missing session 返回 `Released`。

### 2.5 export_training_trace 状态化

开发内容：

1. `export_training_trace` 不再 `get_or_create`。
2. missing session 抛 `KeyError`。
3. 非 `RolloutFinished` 抛 `RuntimeError`。
4. 成功导出后进入 `TrainRunning`。
5. trace 不完整时进入 `ToBeReleased` 并释放，然后抛 `ValueError`。

不做：

1. 不修改 trainer 数据消费逻辑。
2. 不新增 trainer rank ack。

代码 review 关注点：

1. 必须严格要求 `RolloutFinished`。
2. 失败路径不能遗留 session。
3. 成功路径不能释放 session。
4. routed experts 返回的是 session 级 `expert_key` 或 `None`。

测试范围：

1. missing session 抛 `KeyError`。
2. `RolloutRunning` 状态调用抛 `RuntimeError`。
3. 完整 trace 成功导出并进入 `TrainRunning`。
4. prefix 不完整时释放并抛 `ValueError`。

### 2.6 Trainer 侧语义 API

开发内容：

1. 实现 `mark_train_finished(session_id)`。
2. 实现 `mark_train_abandoned(session_id)`。

不做：

1. 不接入 train controller。
2. 不设计 trainer 内部 rank/materialize 计数。

代码 review 关注点：

1. `mark_train_finished` 只允许从 `TrainRunning` 调用。
2. 正常路径必须记录 `TrainFinished`，再进入 `ToBeReleased`。
3. `mark_train_abandoned` 只允许从 `TrainRunning` 直接进入 `ToBeReleased`。
4. missing session 返回 `Released`，用于兼容迟到事件。

测试范围：

1. `TrainRunning -> TrainFinished -> ToBeReleased -> Released`。
2. abandoned 从 `TrainRunning` 释放。
3. 非 `TrainRunning` 调用报错。
4. missing session 返回 `Released`。

### 2.7 TTL 和诊断 API

开发内容：

1. 实现 `gc_stale_sessions(ttl_seconds)`。
2. 实现 `list_sessions(state=None)`。

不做：

1. 不启动后台线程。
2. 不在 Trace Store 内部设置默认 TTL。
3. 不接入外部定时调用方。

代码 review 关注点：

1. 只清理超时的 `RolloutRunning`。
2. 不清理 `RolloutFinished` / `TrainRunning`。
3. 返回被释放的 session_id 列表。
4. `list_sessions` 只做观测，不触发状态变化。

测试范围：

1. 超时 `RolloutRunning` 被释放。
2. 未超时 `RolloutRunning` 不释放。
3. 超时 `RolloutFinished` / `TrainRunning` 不释放。
4. `gc_stale_sessions` 返回释放 session_id。
5. `list_sessions()` 返回状态快照。
6. `list_sessions(state="RolloutRunning")` 可过滤。

### 2.8 兼容读写行为收口

开发内容：

1. 确认并收口 `insert` / `search` / `keys` 行为。
2. missing session 的 `insert` / `search` / `keys` 自动创建空 `RolloutRunning` session。
3. 非 `RolloutRunning` 的 `insert` 记录 error 日志并跳过写入。
4. `RolloutRunning` / `RolloutFinished` / `TrainRunning` 允许 `search` / `keys`。
5. `ToBeReleased` 的 `search` / `keys` 记录 error 日志并返回空结果。

不做：

1. 不改变 `get_store()`。
2. 不修改 `SessionServer` 调用方式。

代码 review 关注点：

1. 保持现有 `SessionServer.on_request` prefix cache 行为。
2. 不能允许已完成或待释放 session 继续写入。
3. `search` / `keys` 自动创建行为只适用于普通读路径，不适用于状态转换 API。

测试范围：

1. missing `search` 创建空 `RolloutRunning` session。
2. missing `keys` 创建空 `RolloutRunning` session。
3. `RolloutFinished` / `TrainRunning` 允许 `search` / `keys`。
4. 非 `RolloutRunning` 的 `insert` 记录 error 日志并跳过写入。
5. `ToBeReleased` 的 `search` / `keys` 记录 error 日志并返回空结果。

## 3. 建议执行顺序

建议按以下顺序执行：

1. Trie session metadata 基础。
2. actor 级 routed experts object registry。
3. 状态转换核心 helper。
4. Rollout 侧语义 API。
5. `export_training_trace` 状态化。
6. Trainer 侧语义 API。
7. TTL 和诊断 API。
8. 兼容读写行为收口。

理由：

1. 前两步先建立 session 数据结构和 object 生命周期基础。
2. 第三步建立统一状态切换和释放入口。
3. 第四到第六步补齐业务语义 API。
4. 第七步补齐 orphan session 运维能力。
5. 第八步最后统一校验兼容行为，避免前面分散修改造成遗漏。

## 4. 全量收口测试

所有小功能完成后，执行全量收口：

```bash
python -m unittest tests.rl.test_trace_store_lifecycle
python -m compileall -q xtuner/v1/rl/rollout/trace_store.py tests/rl/test_trace_store_lifecycle.py
```

如果实现过程中修改了现有调用方，再按实际影响补充相关测试。
