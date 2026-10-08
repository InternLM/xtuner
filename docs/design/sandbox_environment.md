# xtuner sandbox agent loop：环境接口

> 适用范围：`xtuner/v1/rl/agent_loop/sandbox_agent_loop/`，不涉及 agent loop、manager 与 trainer
> 基线：`main` `e7299bbc`（已包含 #2032 的依赖组：`SandboxSpec.dependencies` / `provisioner`）
>
> 本文是 xtuner 内部的架构调整，不涉及任何具体的 sandbox 平台或任务格式。
> 促成本调整的典型用例见附录 A，外部 provider 的最小实现见附录 B。

---

## 1 背景

一条轨迹在 `sandbox_agent_loop` 中的执行流程：

```
AgentInSandboxLoop._run_item(item)
  runner = _resolve_runner(item.pipeline, str(item.uid))   # 每条轨迹 create_object 一个 Runner
  runner.run(item)
    pool_cfg = deepcopy(self._pool) ⊕ item.pipeline_overrides["pool"]
    pool = create_object(pool_cfg)                          # 每条轨迹一个 SandboxPool，其中的 provider 也是新建的
    client = await pool.get(infer_name, record=item.infer)  # 创建依赖组、轮询健康、provisioner、失败整组重试
    infer.run(client, item, record)
    judger.run(item, pool, record)                          # judger 内部 pool.get(judger_name)
    finally: await pool.release_all()                       # 逐组、逆序 provider.delete
```

代码分四层，各层的可替换性如下：

| 层 | 粒度 | 能否从配置替换 |
|---|---|---|
| `AgentInSandboxLoop` | 一批轨迹 | — |
| `Runner` | 一条轨迹的编排 | 能：`item.pipeline` 指向任意有 `run(item)` 的对象 |
| `SandboxPool` | 一条轨迹用到的一组 sandbox | **不能**：`Runner` / `Judger` 直接依赖这个类 |
| 底层 provider（lagent `SandboxProvider`） | 单个 sandbox：`create(**kw) -> (client, id)` / `delete(id)` | 能 |

#2032 之后，`SandboxPool` 以"依赖组"为单位管理 sandbox：一个名字对应 `[*spec.dependencies.items(), ("primary", spec)]`，
按声明顺序逐个创建并等健康，全部健康后调用可选的 `provisioner(primary_client, {dep_name: client})`；
任何一步失败逆序删除整组、退避后整组重试，最多 `max_attempts` 次；创建过程中被取消时以 `asyncio.shield` 保护回滚删除。

---

## 2 问题

"一条轨迹的环境"这一层只有 `SandboxPool` 一种实现，且不可替换：

| 限制 | 影响 |
|---|---|
| **环境的表达力止于依赖组** | 组成可以经 `pipeline_overrides["pool"]["specs"]` 逐条轨迹给出，但只能是"每个名字一个 primary 加一层私有依赖"。服务间网络、共享卷、按依赖条件启动、多个名字共处一个环境（例如 agent 与 verifier 是同一组服务中的两个），都无法表达 |
| **环境实现写死为 `SandboxPool`** | 能替换的只有单 sandbox 的底层 provider，它一次只看到一个 sandbox，承载不了组级的编排 |
| **provider 每条轨迹重建一次** | 无法持有跨轨迹的状态：连接池、后台回收、缓存（创建限流的令牌桶按 key 在进程内共享，是唯一的例外） |
| **释放会被取消打断** | `Runner.run` 的 `finally` 直接 `await pool.release_all()`；`_release_group` 只捕获 `Exception`。分离式训练的权重同步中，`pause_produce` 最多等待 300 秒（`PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S`）后取消仍在运行的轨迹；取消落在逐个删除期间时，后面的删除不再执行，sandbox 只能等 TTL（默认 11700 秒）过期 |
| **就绪之后的故障会进入 reward** | 环境在 agent 执行期间坏掉（辅助 sandbox 被平台删除、服务进程丢失）时，验证在残缺的环境中照常给出 0 分 |

---

## 3 目标与非目标

### 3.1 目标

1. 定义 `Environment` / `EnvironmentProvider` 两个协议，`Runner` / `Judger` 只依赖协议；
2. `SandboxPool` 实现 `Environment`，**现有配置不改动即可运行，现有测试不改动即可通过**；
3. 同一配置的 provider 在进程内（按事件循环）共享一个实例，能持有跨轨迹的状态；
4. 释放不被调用方的取消打断；
5. 可选的 `check()`：就绪之后的基础设施故障使样本作废，而不是记 0 分；
6. 本轨迹的环境变量 `env_vars` 与不透明的任务描述 `task` 随请求传给 provider。

### 3.2 非目标

- 不改变 sandbox 客户端接口（`execute` / `upload_bytes` / `download_file` / `health_check` / `is_pid_running` / `aclose`）；
  `SandboxStage`、hook、entry 不需要改动。
- 不改变 `SandboxSpec` 的现有字段与校验。
- 不改变 `AgentLoop`、`AgentLoopManager`、trainer：本文的改动全部在 `sandbox_agent_loop/` 内。
- 不引入 provider 的启停钩子；provider 自行懒启动，正确性不依赖进程退出时的清理（第 6 节）。
- 不在 `Runner` 中做环境级重试（第 8 节）。
- 不引入任何特定平台、任务格式的概念。

---

## 4 设计概览

```
AgentInSandboxLoop                          不改
  └ Runner                                  只编排；不知道 SandboxPool
      │  provider = 由配置得到（第 6、11 节）
      │  env      = await provider.open(request)
      │  client   = await env.get(infer_name)
      │  infer → validate → env.check() → 写分数
      │  finally: release(env)              不被取消打断（第 9 节）
      │
      └ Environment（协议）
          ├ SandboxPool            现有实现，补上 names / info / release
          │    └ lagent SandboxProvider（不变）
          └ 外部实现               在 xtuner 之外，不 import xtuner
```

| 概念 | 生命周期 | 职责 |
|---|---|---|
| `EnvironmentProvider` | 进程内按配置共享（第 6 节） | 按请求打开环境；持有跨轨迹的资源 |
| `Environment` | 一条轨迹 | 按名字提供 sandbox 客户端；报告就绪之后的故障；释放全部资源 |
| `Runner` | 一条轨迹 | 构造请求、调用各阶段、写记录、保证释放 |

内置的 provider：

| 类 | 用法 | `open(request)` |
|---|---|---|
| `SandboxPoolProvider` | 新形式 `Runner(environment=...)` | 返回 `SandboxPool(provider, request.sandboxes, env_vars=request.env_vars, ...)` |
| `_PoolTemplate`（私有） | 旧形式 `Runner(pool=...)` 的内部表示，每条轨迹一个，不共享 | 返回 `create_object(合并后的 pool_cfg)`，与现在完全相同 |

两种形式在 `Runner` 中走同一条代码路径，区别只在 provider 从哪里来。

---

## 5 接口

新文件 `sandbox_agent_loop/environment.py`：

```python
class EnvironmentRequest(BaseModel):
    """一条轨迹对环境的请求。"""
    model_config = ConfigDict(extra="forbid")

    rollout_id: str                                   # str(item.uid)
    group_id: str | None = None                       # str(item.group_id)
    sandboxes: dict[str, SandboxSpec] = {}            # 本轨迹显式声明的具名 sandbox（可带依赖组）
    env_vars: dict[str, str] = {}                     # 本轨迹的环境变量，5.3 节
    task: dict[str, Any] = {}                         # item.environment，由 provider 解释，xtuner 不读


class SandboxInfo(BaseModel):
    image: str | None = None                          # open 之后即可知
    workspace_path: str | None = None                 # open 之后即可知
    env_id: str | None = None                         # 就绪之后才有
    url: str | None = None                            # 就绪之后才有；不发布地址时为 None
    metadata: dict[str, Any] = {}                     # 并入该名字所在 stage 的 record.metadata


@runtime_checkable
class Environment(Protocol):
    @property
    def names(self) -> frozenset[str]: ...
    async def get(self, name: str, *, record: StageRecord | None = None) -> Any: ...
    def info(self, name: str) -> SandboxInfo: ...
    async def release(self) -> None: ...


@runtime_checkable
class EnvironmentProvider(Protocol):
    async def open(self, request: EnvironmentRequest) -> Environment: ...
```

`Environment` 上可选的健康检查（实现了就调用，没实现视为总是健康）：

```python
async def check(self) -> BaseException | None: ...   # 5.6 节
```

### 5.1 接口约定

| 方法 | 约定 |
|---|---|
| `open` | 不为本轨迹创建远程资源，所以失败时没有要释放的东西。provider 的一次性初始化（连接、后台任务、启动时的残留清理）可以放在第一次 `open` 中，并发的首次调用只初始化一次；初始化失败时每次 `open` 都抛出该错误 |
| `names` | `open` 返回后即确定。包括 `request.sandboxes` 的键，以及 provider 根据 `task` 提供的名字；依赖组的私有成员不在其中 |
| `get(name, record=)` | 名字不在 `names` 中时抛 `KeyError`；首次调用时就绪并缓存，之后返回同一个客户端；同一名字的并发调用只就绪一次；失败时抛出异常，并**尽量**写 `record.error` / `record.metadata`（没写的由调用方补写，第 10 节）；平台层的重试在这里完成（第 8 节） |
| `info(name)` | 名字不在 `names` 中时抛 `KeyError`；`image` / `workspace_path` 在 `open` 之后可知，`env_id` / `url` 在就绪之后可知 |
| `release` | 幂等；不抛异常；可以耗时，也可以只把删除交给 provider 自己的回收器、立即返回 |
| `check` | 可选；不抛异常；5.6 节 |

`get` 返回的客户端必须实现现有 sandbox 客户端接口。

### 5.2 环境错误的识别

xtuner 按属性识别环境错误，而不是按类型，外部 provider 不需要 import xtuner：

```python
class EnvErrorInfo(NamedTuple):
    stage: str
    category: str
    retryable: bool


def env_error_info(exc: BaseException) -> EnvErrorInfo | None:
    retryable = getattr(exc, "retryable", None)
    if not isinstance(retryable, bool):
        return None
    return EnvErrorInfo(getattr(exc, "stage", "unknown"), getattr(exc, "category", "unknown"), retryable)
```

识别结果只用于记录（第 10 节）：xtuner 不按 `retryable` 改变控制流（第 8 节）。

### 5.3 `env_vars`

接口保证两件事：

1. 经 `get(name)` 得到的客户端，其**每一次** `execute` 都能看到它；
2. 值按字面传递：provider 不把值拼进 shell 命令串，值中的 `"`、`$`、反引号不会被 shell 解释。

| 规则 | 说明 |
|---|---|
| 作用范围 | `names` 中每个名字的客户端。依赖组的私有成员不注入 |
| 与 `spec.env_vars` | `spec.env_vars` 在前，`request.env_vars` 覆盖同名键 |
| 与 entry 的 `env` | 互不影响。entry 级的 `env`（含 `_inject_session_id` 注入的 `XTUNER_SESSION_ID`）照旧经 `exec_in` 作用于该 entry 的命令 |
| 注入方式 | 由 provider 决定。`SandboxPool` 合并进 primary 的 `env_vars`，随 `provider.create(env_vars=...)` 在创建时注入，sandbox 内的进程都能看到；其他实现可以在每次执行时作为该命令的环境传入 |
| 敏感信息 | 有的平台在查询 sandbox 时会原样返回创建参数中的环境变量。口令类值不应通过 `env_vars` 传递 |

sandbox 中不经客户端启动的进程（例如任务自己的服务进程）是否看到它，不在接口内。

### 5.4 请求的构造

`AgentRolloutItem` 增加一个字段，由数据侧（tokenize 函数）写入：

```python
class AgentRolloutItem(BaseModel):
    ...
    environment: dict[str, Any] = Field(default_factory=dict)   # 不透明，原样作为 request.task
```

它与 `metadata` 一样不被 xtuner 解释；与 `pipeline_overrides` 不同，它不是对 Runner 配置的覆盖，而是交给 provider 的样本属性
（例如 `{"manifest": "<任务描述文件的路径>"}`）。

| 请求字段 | 来源 |
|---|---|
| `rollout_id` | `str(item.uid)`（`generate_sample` 在运行前总会给 `uid` 赋值） |
| `group_id` | `str(item.group_id)`；为 `None` 时为 `None` |
| `sandboxes` | `Runner` 配置的 `sandboxes`，深合并 `pipeline_overrides["sandboxes"]` |
| `env_vars` | `Runner` 配置的 `env_vars`，被 `pipeline_overrides["env_vars"]` 覆盖同名键 |
| `task` | `item.environment` |

### 5.5 `SandboxSpec`

不改动。`dependencies` / `provisioner` 由 `SandboxPool` 原样支持；
其他 provider 不支持某个字段时，在 `open` 中抛 `retryable=False` 的环境错误。

### 5.6 就绪之后的基础设施故障

`get` 只能报告就绪之前的失败。就绪之后环境仍可能坏掉，而 agent 与验证可能照常返回。

| 规则 | 说明 |
|---|---|
| 检查时机 | `Runner` 在 `validate` 返回之后、把分数写入 item 之前调用一次 `env.check()`；`Judger` 不调用 |
| 有故障 | `Runner` 把它写进 `item.infer.error`（`stage="sandbox:environment.check"`，category 取错误的 `category`）与 `env_error_*`（第 10 节），轨迹失败，训练模式下样本丢弃。写在 infer 记录上：现有的异常处理按 `item.infer.error` → 第一个 judger 错误 → `stage="runner"` 的顺序取轨迹的错误，写在这里才不会被顶替 |
| 什么算基础设施故障 | 由 provider 定义；约定是 provider 自身或平台的失败，包括 provider 没能维持任务声明的环境语义（例如任务要求自动重启的服务没有被重启）；**不包括**任务内服务进程的崩溃、健康检查失败本身——这些可能正是 agent 的行为结果 |
| 未知不等于健康 | `check` 可以做有时限的确认（例如向平台查询一次）；无法在时限内确认健康时应返回错误，而不是 `None` |
| 时限 | `Runner` 以 `check_timeout`（默认 30 秒）包住调用；超时按故障处理（`category="check_timeout"`） |
| 没实现 `check` | 视为没有故障，行为与现在相同（`SandboxPool` 不实现） |

provider 也可以在发现故障后让之后的 `get` 与客户端调用立即失败，使 agent 阶段提前结束；这不替代 `check`，
因为故障可能发生在 agent 最后一次调用之后、验证之前。

---

## 6 provider 实例的共享

`Runner` 每条轨迹 `create_object` 一次，它持有的对象都是每条轨迹一份。跨轨迹的 provider 放在 `environment.py` 的进程内注册表中：

```python
def shared_provider(config: Mapping[str, Any]) -> EnvironmentProvider:
    """同一配置、同一事件循环返回同一个 provider 实例；第一次调用时 create_object。"""
```

| 规则 | 说明 |
|---|---|
| 键 | `(当前运行的事件循环, 规范化的配置)`。规范化：`type` 取类的全限定名，其余字段按 `json.dumps(sort_keys=True)`；不可 JSON 序列化的配置抛 `TypeError`，避免以对象 `id` 为键导致每条轨迹一个实例 |
| 按事件循环区分 | Ray async actor 中每个并发组有自己的事件循环，provider 持有的连接与后台任务只能在创建它的循环上使用。注册表以 `WeakKeyDictionary` 按循环保存，循环被回收后对应条目随之消失 |
| 作用域 | 进程内同一配置共享一个实例，训练与评测的 agent loop 若配置相同也共享。provider 不能假定自己是本进程唯一的实例：启动时清理"上一次运行的残留"，只能清理按自己的实例标识（例如写进它创建的每个 sandbox 标签中的启动 ID）判定为无主的资源 |
| 配置中不放逐条轨迹的值 | 逐条轨迹的内容走请求（5.4 节）。配置不同即实例不同 |
| 现有先例 | 创建限流的令牌桶（`get_shared_async_token_bucket`）已按 key 在进程内共享 |

**没有启停钩子。** 一次性初始化在 provider 第一次 `open` 时完成（5.1 节）；xtuner 不在进程退出时调用 provider。
provider 不能依赖退出时的清理保证正确性：进程可能被强制终止，残留由 TTL 或 provider 自己的残留清理处理。

**事件循环并不总在运行。** 共卡训练（`RLColocateTrainer`）只在 `asyncio_run(produce_batch(...))` 期间运行事件循环，
训练步之间循环停止，provider 的后台任务（续期、回收、心跳）随之暂停。依赖周期性后台任务维持正确性的 provider，
要么把这些任务放在自己的线程中，要么让它的超时与 TTL 覆盖最长的训练步。

---

## 7 `Runner` 与 `Judger` 的改动

### 7.1 `Runner`

```python
class Runner:
    def __init__(
        self,
        *,
        environment: dict[str, Any] | None = None,           # 新形式：provider 配置，经 shared_provider 共享
        sandboxes: dict[str, SandboxSpec | dict] | None = None,
        env_vars: dict[str, str] | None = None,
        pool: SandboxPool | dict[str, Any] | None = None,    # 旧形式，11.2 节
        infer: SandboxStage | dict[str, Any],
        validate: Judger | dict[str, Any],
        check_timeout: float = 30.0,
    ): ...

    async def run(self, item: AgentRolloutItem) -> AgentRolloutItem: ...
```

`run` 的主体：

```python
provider = self._provider(item)          # 新形式：shared_provider(self._environment)；旧形式：_PoolTemplate(合并后的 pool_cfg)
request = self._build_request(item)      # 5.4 节；旧形式下 sandboxes / env_vars 为空
env: Environment | None = None
try:
    env = await provider.open(request)
    infer_name = _stage_sandbox_name(self.infer, env)          # 用 env.names 校验
    _fill_record(item.infer, infer_name, env.info(infer_name)) # image / workspace，获取失败时记录中也有
    infer_client = await env.get(infer_name, record=item.infer)
    _fill_record(item.infer, infer_name, env.info(infer_name)) # env_id / url / metadata
    ...infer...（不变）
    score = float(await self.validate.run(item, env, validate_record))
    if (fault := await self._check(env)) is not None:          # 5.6 节
        _record_env_error(item.infer, fault, stage="sandbox:environment.check")
        return self._fail(item, item.infer.error)
    item.reward = score
    item.status = RolloutStatus.COMPLETED
    return item
except Exception as exc:
    ...（不变：按 item.infer.error → judger 错误 → stage="runner" 提升）
finally:
    self._log_final(...)
    if env is not None:
        await _release(env)                                    # 第 9 节
```

trace span、计时与最终日志保持不变。

### 7.2 `Judger`

参数从 `pool: SandboxPool` 改为 `env: Environment`：
名字校验用 `env.names`，`client = await env.get(name, record=record)`，`image` / `workspace` 从 `env.info(name)` 读取。
`SandboxPool` 满足 `Environment` 协议，把 `SandboxPool` 实例传给 `Judger.run` 的现有代码不受影响。

---

## 8 重试

重试只在 `Environment` 内部：

| 实现 | 重试 |
|---|---|
| `SandboxPool` | 现有的整组重试（`max_attempts`、退避 `min(2**attempt, 8)` 秒、逆序回滚），不变 |
| 外部实现 | 自行决定，包括是否把整个环境重来一次 |

`Runner` 不重试：`open` 或 infer sandbox 的 `get` 失败，轨迹即失败，训练模式下样本丢弃。
只有环境自己知道哪些失败重来有用、重来需要先清理什么；`Runner` 再套一层重试会与内部重试相乘。
agent 开始执行之后出现的环境错误同样不重试，避免重复执行耗时的 agent 阶段。

---

## 9 释放与取消

```python
_RELEASING: set[asyncio.Task] = set()


async def _release(env: Environment) -> None:
    task = asyncio.create_task(env.release())
    _RELEASING.add(task)
    task.add_done_callback(_RELEASING.discard)
    await asyncio.shield(task)
```

| 规则 | 说明 |
|---|---|
| 正常路径 | `run` 等释放完成再返回，与现在相同。共卡训练在 `asyncio_run` 返回后停止事件循环，所以正常路径必须等释放完成，不能只交给后台 |
| 被取消 | 取消打断的是 `shield` 外的等待，释放任务继续在事件循环中完成；取消立即向上传播 |
| 引用 | 释放任务在模块级集合中保留引用直到完成，不被垃圾回收 |
| 取消的来源 | 分离式训练的权重同步：`pause_produce` 等待 300 秒后取消剩余轨迹，`cancel_and_drain` 再等 5 秒后不再等待；释放在后台继续。共卡训练中被取消的释放在下一次 `produce_batch` 时继续 |
| 两种形式 | 都这样释放；旧形式的 `release_all` 也由此不再被取消打断 |

进程退出时仍未完成的释放不再等待，sandbox 由 TTL 或 provider 的残留清理回收（第 6 节）。

---

## 10 错误与记录

获取失败时 `record.error` 的口径不变：

```python
RolloutError(stage=f"sandbox:{name}.acquire", category="acquire", type=type(exc).__name__, message=...)
```

`SandboxPool` 与现在一样自己写入；环境没有写入时，由 `Runner` / `Judger` 以 `record.error = record.error or RolloutError(...)` 补写。
`open` 失败时 `Runner` 写 `stage="sandbox:environment.open"`，category 取错误的 `category`（不是环境错误时为 `"open"`）。

错误带 5.2 节的属性时，`record.metadata` 在现有键（`sandbox_create_attempts`、`sandbox_create_to_ready_time_s`、
`sandbox_acquire_rate_limit_wait_s`）之外增加：

| key | 值 |
|---|---|
| `env_error_stage` | `stage` |
| `env_error_category` | `category` |
| `env_error_retryable` | `retryable` |

`info(name).metadata` 并入该名字所在 stage 的 `record.metadata`。轨迹的最终状态和样本丢弃规则不变：训练模式下失败的样本被丢弃，不产生 reward。

---

## 11 配置与兼容

### 11.1 新形式

```python
Runner(
    environment={"type": SandboxPoolProvider, "provider": {...}, "max_attempts": 3},
    sandboxes={"main": {...}, "verifier": {...}},
    env_vars={"SOME_VAR": "1"},
    infer=..., validate=...,
)
```

`SandboxPoolProvider` 的构造参数就是 `SandboxPool` 除 `specs` 以外的参数：

```python
SandboxPoolProvider(provider, *, max_attempts=3, health_max_wait_sec=600.0, health_poll_interval_sec=2.0,
                    creates_per_sec=3.0, creates_burst=1,
                    create_rate_limit_key="xtuner.sandbox_agent_loop.sandbox_acquire")
```

它只持有配置，`open` 返回一个新的 `SandboxPool`；底层 provider 仍由每个 `SandboxPool` 各自 `create_object`，与现在相同，
所以共享 `SandboxPoolProvider` 不引入新的跨轨迹状态。`request.task` 被忽略。

换用其他 provider 只改 `environment`，`infer` / `validate` 不变。不同 provider 下请求的形态可以不同：
`SandboxPoolProvider` 的组成全部来自 `sandboxes`；由任务决定组成的 provider 从 `task` 得到组成，`sandboxes` 只放额外的独立 sandbox。

### 11.2 旧形式

```python
Runner(pool={"type": SandboxPool, "provider": {...}, "specs": {...}, ...}, infer=..., validate=...)
```

- 每条轨迹把 `pipeline_overrides["pool"]` 深合并进 `pool_cfg`，包装成一个 `_PoolTemplate`，它的 `open` 返回 `create_object(pool_cfg)`。
  覆盖可以触及底层 provider 的参数（例如按任务给底层 provider 不同的卷配置），所以旧形式的 provider 不共享、每条轨迹构建一次，与现在相同；
- 传入预先构建好的 `SandboxPool` 实例（测试用法）时，`open` 直接返回它；此时 `pipeline_overrides` 非空报错，与现在相同；
- `SandboxPool` 的构造参数、`get` / `validate_name` / `env_id` / `url` / `spec` / `release_all` 保留。

### 11.3 互斥

- `environment` 与 `pool` 必须且只能设置一个；
- `sandboxes` / `env_vars` 只在新形式下有效，与 `pool` 同时设置时报错；
- 新形式下 `pipeline_overrides` 只接受 `sandboxes`、`env_vars` 键，旧形式下只接受 `pool` 键。

---

## 12 `SandboxPool` 的改动

类名与构造参数不变，新增：

| 新增 | 行为 |
|---|---|
| `env_vars` 构造参数 | 合并进每个公开名字的 primary 的 `spec.env_vars`，不进入依赖成员（5.3 节） |
| `names` | `frozenset(specs)` |
| `info(name)` | `image` / `workspace_path` 取自 spec；就绪后加上 `env_id` / `url`；`metadata` 为该名字的创建指标 |
| `release()` | 即 `release_all()` |
| 按名字加锁 | 同一名字的并发 `get` 只创建一次（5.1 节）；现有调用路径中同一名字不会并发获取，行为不变 |

依赖组、provisioner、健康轮询、整组重试、取消时的回滚、创建限流、逆序释放都不变。

---

## 13 测试

| 测试 | 内容 |
|---|---|
| 现有测试 | 不改动，全部通过（含 #2032 的依赖组、provisioner、回滚、取消测试） |
| 兼容性 | 旧形式与新形式 + `SandboxPoolProvider` 对同一假 provider 产生相同的 create/delete 调用序列和记录 |
| 接口契约 | 假 provider 覆盖：`open` 不为轨迹创建资源；`get` 并发去重；`release` 幂等且不抛异常；依赖成员不在 `names` 中；`info` 在就绪前后的字段 |
| 共享 | 同一配置、同一循环返回同一实例；配置不同或循环不同返回不同实例；不可序列化的配置报错；旧形式不经注册表 |
| `env_vars` | 注入 primary、不注入依赖成员；覆盖 `spec.env_vars` 同名键；不影响 entry 的 `env`；含 `"`、`$`、反引号的值原样到达 |
| 请求构造 | `item.environment` 原样成为 `task`；`pipeline_overrides` 的 `sandboxes` 深合并、`env_vars` 覆盖；`uid` / `group_id` 转为字符串；不允许的键报错 |
| 取消与释放 | `Runner.run` 在 `get`、infer、validate 期间被取消时，取消立即传播、释放仍执行完；正常路径释放完成后才返回；每个打开过的环境恰好释放一次；`open` 失败不释放 |
| `check` | 返回故障时轨迹失败、`item.infer.error` 的 stage 为 `sandbox:environment.check`、记录 `env_error_*`、不产生分数；超时按故障处理；未实现 `check` 的环境行为不变 |
| 记录 | 获取失败时 `record.error` 与 `record.metadata` 字段正确；外部环境未写 `record.error` 时由调用方补写；获取失败的记录带镜像与工作目录；不 import xtuner 的外部异常类同样被识别 |

---

## 14 实现计划

一个 PR，按以下顺序分 commit，每个 commit 单独通过现有测试：

1. 新增 `environment.py`：协议、`EnvironmentRequest`、`SandboxInfo`、`env_error_info`、`shared_provider`；
2. `SandboxPool` 新增 `env_vars` / `names` / `info` / `release`、按名字加锁；新增 `SandboxPoolProvider`；
3. `Runner` / `Judger` 改用协议，旧形式经 `_PoolTemplate` 走同一路径，释放改为第 9 节的形式；
4. `AgentRolloutItem.environment`、请求构造、`pipeline_overrides` 的新键；
5. `check`；
6. 测试与文档。

---

## 附录 A 典型用例：按任务描述的多服务环境

一类 agentic RL 任务（例如网站运维、漏洞复现）的环境由多个服务组成：agent 工作区、反向代理、应用、数据库、缓存。
每个任务用自己的描述文件（例如 Docker Compose）给出服务、网络、共享卷与启动依赖，不同任务各不相同。

一种实现方式是把每个服务放在一个独立的 sandbox 中，由 provider 按依赖条件启动服务、让服务以服务名互连、提供共享存储，
并在后台回收整组 sandbox。这样的实现与 xtuner 的接口关系如下：

| 实现的需要 | 现在为什么做不到 | 本文中的对应 |
|---|---|---|
| 组成由每个样本的任务描述决定 | 依赖组只能表达"一个 primary 加一层私有依赖"，没有网络、卷、启动依赖 | `item.environment` → `request.task`，由 provider 解释（5.4 节） |
| agent 与 verifier 是同一组服务中的两个 | `SandboxPool` 中每个名字是独立的依赖组 | 一个 `Environment` 提供多个名字（5.1 节） |
| 跨轨迹的后台回收、限流、连接池 | provider 每条轨迹重建一次 | 进程内按配置共享的 provider（第 6 节） |
| 进程崩溃后清理上一次运行的残留 | 没有初始化时机 | 第一次 `open` 时初始化，按自己的实例标识判定残留（5.1、第 6 节） |
| 释放不被取消打断、可以交给后台 | `release_all` 在被取消的任务中运行 | 第 9 节；`release` 可以只交给回收器、立即返回 |
| 区分平台的临时失败与任务本身的错误 | 整组重试对任何异常一视同仁 | 重试在环境内部，由实现决定（第 8 节） |
| 给 agent 所在 sandbox 注入本轨迹的变量，而不改变任务服务自身的环境 | 只能改 spec 或靠针对特定 entry 的注入 | `request.env_vars`，注入方式由 provider 决定（5.3 节） |
| 服务在 agent 执行期间丢失时不产生 reward | 验证在残缺的环境中给出 0 分 | 可选的 `Environment.check()`（5.6 节） |

这类实现放在 xtuner 之外，按第 5 节接口的结构实现、不 import xtuner；xtuner 中不包含任何相关代码，
配置中以导入路径引用它（`Runner(environment={"type": "<包>.<模块>.<类>", ...})`）。

## 附录 B 外部 provider 的最小实现

只用于说明接口的形状；错误类只需带 5.2 节的三个属性。

```python
class EnvError(RuntimeError):
    def __init__(self, message, *, stage, category, retryable):
        super().__init__(message)
        self.stage, self.category, self.retryable = stage, category, retryable


class MyProvider:
    def __init__(self, endpoint: str, max_concurrency: int = 64):   # 配置可 JSON 序列化（第 6 节）
        self._endpoint = endpoint
        self._init_lock = asyncio.Lock()
        self._client = None

    async def open(self, request):
        async with self._init_lock:                                  # 一次性初始化，5.1 节
            if self._client is None:
                self._client = await connect(self._endpoint)
                await self._client.cleanup_orphans()
        spec = load_task(request.task["manifest"])                   # 只在本地解析，不建远程资源
        return MyEnvironment(self._client, spec, request)


class MyEnvironment:
    def __init__(self, platform, spec, request):
        self._platform, self._spec, self._request = platform, spec, request
        self._ready: asyncio.Task | None = None                      # 整组只启动一次

    @property
    def names(self):
        return frozenset(self._spec.services)

    async def get(self, name, *, record=None):
        if name not in self.names:
            raise KeyError(name)
        if self._ready is None:
            self._ready = asyncio.ensure_future(self._platform.start_group(self._spec, self._request))
        group = await asyncio.shield(self._ready)                    # 失败时抛 EnvError
        return group.client(name, env_vars=self._request.env_vars)

    def info(self, name):
        svc = self._spec.services[name]
        return SandboxInfo(image=svc.image, workspace_path=svc.workdir)

    async def release(self):
        self._platform.schedule_delete(self._request.rollout_id)    # 交给后台回收，立即返回

    async def check(self):
        return await self._platform.probe(self._request.rollout_id)  # 故障返回 EnvError，健康返回 None
```
