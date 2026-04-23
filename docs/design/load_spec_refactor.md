# LoadSpec 设计

> 面向 `xtuner/v1/utils/load_spec.py` 与 `xtuner/v1/model/base.py` 的加载/保存路径。
> TP 设计（`dense_tp.md`）依赖本文档描述的抽象。

## TL;DR

LoadSpec 描述 xtuner 运行时 tensor 与 HF safetensors 之间的**纯布局映射**。对一个
param，它回答两件事：

1. 这个 param 由哪些 HF key 组成？怎么拼？ — `global_hf_keys` + `fused_dim`
2. 本 rank 持有全量 tensor 的哪一块？ — `shards`（按外到内顺序施加）

加载/保存执行路径不直接读 LoadSpec，而是调用 `plan_hf_load()` /
`plan_hf_save(...)` 拿到一份**不可变的 plan**，按 plan 驱动 IO 与通信。

**核心约束**：LoadSpec 只承担"同 dtype 下的形状/索引映射"。fp8 的量化反量化、
padding 的 zero-fill 等 dtype 语义都住在 `base.py` 的 load/save 路径里，
LoadSpec 不感知。

---

## 1. 设计理念

### 1.1 单一抽象，两条正交轴

原先三类映射（SAME / FUSED / SHARD）统一成一个 schema 上的两个正交维度：

| 问题 | 表达 |
| --- | --- |
| 这个 param 对应几个 HF key？怎么拼？ | `len(global_hf_keys)`；多 key 时 `fused_dim` 指定拼接维 |
| 本 rank 持有哪一块？ | `shards`（可为空；按施加顺序排列） |

消费方用派生属性 `is_fused` / `is_sharded` 查询，**不需要**任何枚举分支。

### 1.2 多维切分按顺序叠加

`shards` 是列表，原生支持 TP × FSDP、EP × FSDP 等多轴组合。每条
`ShardDescriptor.start/end` 的含义是"在**前面所有** descriptor 切完之后的
子 tensor 上的偏移"。这条规则完全对齐 DTensor `placements` 从 `mesh_dim=0` 到
`mesh_dim=N-1` 逐步施加的语义 —— 你可以把 `shards[i]` 理解成 `placements[i]`
在"此刻本 rank 实际持有"这个问题上的等价形式。

### 1.3 Plan 是冻结快照

`plan_hf_load()` / `plan_hf_save(...)` 返回的是 Pydantic dataclass：

- 一次性从当前 LoadSpec 状态计算出执行所需的全部信息；
- 不持有对 LoadSpec 的引用；
- 执行器（`_load_hf_param` / `unshard_tensors_for_hf_save` / `_split_hf_tensors_for_save`）
  只读 plan，**不读** LoadSpec。

这条边界保证"布局规划"和"IO/通信执行"解耦。未来要接入新的持久化格式（例如
DCP），只需要替换 plan 的消费者，不牵涉 LoadSpec 内部结构。

### 1.4 fp8 与 LoadSpec 解耦

LoadSpec 是"同 dtype 下的布局描述"。fp8 涉及的两件事 —— 量化/反量化、运行时
padding —— 归属如下：

- **运行时 padding**：用 `LoadSpec.origin_shape` 表达 checkpoint-visible shape
  （剥掉运行时 padding 之后）。今天这个字段的唯一来源是 fp8 tensor metadata；
  它只记录 shape，不记录 dtype / wrapper 类型。
- **量化/反量化**：只在 `base.py._to_float8` / 反量化分支里现场判断（通过
  `is_float8_weight(tensor)`）。LoadSpec 不包含 `runtime_is_float8` 这类
  dtype-specific 字段。

### 1.5 Spec → Plan → Executor 的分层

```
┌────────────────────┐  plan_hf_load()    ┌──────────────┐
│                    │ ──────────────────▶│ HFLoadPlan   │──▶ _load_hf_param
│     LoadSpec       │                    └──────────────┘
│  (pure layout)     │  plan_hf_save(...) ┌──────────────┐
│                    │ ──────────────────▶│ HFSavePlan   │──▶ unshard_tensors_for_hf_save
└────────────────────┘                    └──────────────┘                    │
                                                                              ▼
                                                                      _split_hf_tensors_for_save
```

"Spec 是源、Plan 是派生、Executor 只依赖 Plan"。这条线保持单向。

---

## 2. 数据模型

### 2.1 `ShardDescriptor`

```python
class ShardDescriptor(BaseModel):
    dim: int                           # 被切的维
    start: int                         # 在"前面切完的 sub-tensor"上的起点
    end: int                           # 在"前面切完的 sub-tensor"上的终点
    group: dist.ProcessGroup           # 产生这次切分的通信组
```

`group` 是 load/save 双向通信域。load 时只需要知道本 rank 的范围；save 时需要沿
`group` 做 all-gather 复原全量 tensor。

### 2.2 `LoadSpec`

```python
class LoadSpec(BaseModel):
    name: str                          # xtuner 侧 fully-qualified param name
    global_hf_keys: list[str]          # 对应的 HF key 列表（按 fused_dim 拼接顺序）
    global_shape: tuple[int, ...]      # 全量 tensor（fused 之后）的 runtime shape
                                       # 可能包含运行时 padding（例如 fp8 的 FSDP 对齐 pad）
    fused_dim: int | None = None       # 多 HF key 时的拼接维；单 key 时必须为 None
    shards: list[ShardDescriptor] = [] # 从外到内的切分列表
    origin_shape: tuple[int, ...] | None = None  # checkpoint-visible shape after runtime padding is trimmed
                                       # None 表示"runtime shape 就是 checkpoint shape"
```

派生属性：

```python
is_fused            # len(global_hf_keys) > 1
is_sharded          # bool(shards)
unpadded_global_shape  # origin_shape or global_shape
```

**不变量**（`model_post_init` 强制）：

- `is_fused` ⇔ `fused_dim is not None`；
- 每条 shard 的 `start/end` 必须落在"前面切完之后的 sub-tensor"范围内；
- 若 `origin_shape` 给定，它的秩与 `global_shape` 相同，且每维 `≤ global_shape`。

### 2.3 `HFLoadPlan`

`plan_hf_load()` 的产出：

```python
class HFLoadPlan(BaseModel):
    name: str
    hf_keys: list[str]                 # 本 rank 实际需要读的 HF key
    fused_dim: int | None = None       # 多 key 时的拼接维
    slices: list[LoadSlice] = []       # 读完拼接后，再做的 narrow 列表
    zero_fill: bool = False            # 本 rank 完全落在运行时 padding 区，跳过 IO
```

`slices` 的 start/end 是**相对已加载 tensor 的坐标**，不是相对 `global_shape`。
zero_fill=True 时 `hf_keys` 和 `slices` 都为空。

### 2.4 `HFSavePlan`

`plan_hf_save(...)` 的产出，承载两类信息：

```python
class HFSavePlan(BaseModel):
    name: str
    hf_keys: list[str]                  # 当前 save tensor 最终要写/同步的 HF keys
    global_shape: tuple[int, ...]
    unpadded_global_shape: tuple[int, ...]
    fused_dim: int | None = None
    distributed_save: bool = False
    preserves_shards: bool = False      # True 表示 hf_keys 来自保留 shard 后的局部 tensor
    unshard_steps: list[SaveShardStep] = []      # 所有 shard 的逆操作 + preserved 标记
```

`SaveShardStep` 记录一次 shard 在"施加前的 runtime shape / checkpoint-visible
shape"两个快照 —— save 执行时倒序跑每一步、all-gather 还原、narrow 回
checkpoint-visible shape。`preserved` 标记把某些 shard 排除在 all-gather 之外
（见 §3.3）。`HFSavePlan.hf_keys` 始终是执行器要处理的 key 集合：普通 save 下
它是完整 HF key list，preserved shard save 下它是当前局部 shard 覆盖的 key list。

---

## 3. 计划生成

### 3.1 `plan_hf_load()`

不接受参数 —— 本 rank 的所有信息已经在 LoadSpec 里。步骤：

1. 计算本 rank 最终持有的区间 `final_intervals`（顺序应用 `shards`）；
2. 用 `unpadded_global_shape` 裁剪掉运行时 padding 部分；若裁完为空，返回
   `zero_fill=True`；
3. 若 `is_fused`，按 `fused_dim` 上的区间算出需要的 HF key 下标范围（floor/ceil
   支持 mid-key shard，例如 FSDP 在 EP-local 专家 tensor 内部再切）；
4. 对每个 dim，如果"最终区间"比"加载后的 tensor 区间"窄，生成一条 `LoadSlice`。

### 3.2 `plan_hf_save(distributed_save=, preserve_process_group=, gather_process_group=)`

三个参数对应三种 save 策略，互斥使用：

| 参数 | 用途 |
| --- | --- |
| `distributed_save=True` | HF save：非 fused tensor 只在 rank0 写；fused tensor 的 HF key 在 save rank 间分配 |
| `preserve_process_group=ep_group` | RL 权重同步：保留 EP 在 `fused_dim` 上的 shard，每个 EP rank 只流自己的 expert key；其他 shard 照常 all-gather |
| `gather_process_group=fsdp_group` | FSDP-only all-gather：只 gather 这个 group 的 shard，其他 shard 保留 |

策略统一落到 `_preserved_shard_indices` 这一步上 —— 决定哪些 `LoadSpec.shards`
需要保留。之后 `_save_shard_steps` 给每个 shard 生成带 `preserved` 标记的
`SaveShardStep`。若有 preserved shard，`LoadSpec` 直接从这些 shard 推导
`HFSavePlan.hf_keys`；save plan 只暴露最终要写/同步的 HF keys，以及
`preserves_shards` 说明这些 keys 来自局部 tensor 还是完整 tensor。

### 3.3 preserve vs gather 的正交性

`preserve_process_group` 是"显式保留某个 group"的策略，`gather_process_group`
是"显式 gather 某个 group（其余保留）"的策略。两者不能同时使用（assert 拦截）。
在今天的代码里：

- 普通 HF save：两者都不传，全部 all-gather；
- RL 权重同步：传 `preserve_process_group=ep_group`；
- `_fsdp_foreach_allgather`：传 `gather_process_group=fsdp_group`，只做 FSDP
  层的 all-gather，不动 EP / TP。

---

## 4. 执行

### 4.1 加载路径

```python
def _load_hf_param(self, param, load_spec, loader):
    plan = load_spec.plan_hf_load()
    if plan.zero_fill:
        # 本 rank 只持有运行时 padding，写 0 返回
        local_tensor.zero_()
        return []
    # 按 plan.hf_keys 逐个读（fp8 走 dequant 分支，这里 base.py 现场处理）
    loaded_tensors = self._load_hf_keys(plan, loader, ...)
    # 拼接 + narrow 全部交给 safetensors_to_params
    self.safetensors_to_params(loaded_tensors, local_tensor, plan)
```

`safetensors_to_params` 的签名是 `(safetensors, local_tensor, plan)`。三个 MoE
子类（`gpt_oss`、`qwen3_5_text`、`qwen3vl_text`）按 `plan.name` 做 reshape /
transpose 等模型特有变换后，调通用的 `_apply_load_slices` + `_copy_loaded_tensor_to_local`。

### 4.2 保存路径

所有 save 场景（HF save、RL 权重同步、FSDP-only gather）共用一条管道：

```python
save_items = [HFSaveItem(tensor, load_spec.plan_hf_save(...)) for ...]
full_tensors = unshard_tensors_for_hf_save(save_items)
for full_tensor, item in zip(full_tensors, save_items):
    names, tensors = self._split_hf_tensors_for_save(full_tensor, item.save_plan)
```

`unshard_tensors_for_hf_save` 自带**依赖感知的批量 foreach all-gather**：

- 同一个 tensor 的多个 step 必须串行（例如 "先还原 FSDP，再还原 EP"）；
- 不同 tensor 的 step 如果 `(group, dtype)` 兼容，可以 foreach 批到同一次 NCCL 调用。

每一轮由 `_build_ready_save_unshard_groups` 从每个 pending 队列取头部 step，按
group + dtype 分桶；`_foreach_all_gather_save_shards` 跑一次批量 gather；下一轮
再消费队列的下一层。MoE EP+FSDP 的 save 就是这样两轮跑完的。

### 4.3 `HFSaveItem`

```python
class HFSaveItem(NamedTuple):
    tensor: torch.Tensor
    save_plan: HFSavePlan
```

这是**跨 LoadSpec 和 BaseModel 边界**的 bundle：一边是 runtime tensor（模型侧
概念，带 fp8 wrapper / DTensor wrapper），一边是纯布局的 `HFSavePlan`。它的
归属地是 `base.py` ——`load_spec.py` 保持"不认识模型侧概念"。
`unshard_tensors_for_hf_save` 的签名使用两个平行列表（`list[torch.Tensor]` +
`list[HFSavePlan]`）而不是 `list[HFSaveItem]`，避免 `load_spec.py` 反向依赖
`base.py`。

---

## 5. 调用时机

`_init_load_spec` 被定位为"从当前 DTensor 布局反推 HF 映射的纯函数"。
调用约定：**谁改 param 布局谁负责重算，后者覆盖前者**。

| 时机 | 调用方 | spec 代表 |
| --- | --- | --- |
| 子类 `__init__` 末尾 | 子类自己 | 构建完成时的布局（EP-only / Replicate / 其它 init-time 切分） |
| `parallelize(tp_mesh)` 结束 | `BaseModel.parallelize` | TP + 已有切分 |
| `fully_shard` 结束 | `BaseModel.fully_shard` | 叠加 FSDP（训练态） |
| `Float8Handler.pad_for_fsdp` 回调 | 回调内 | fp8 pad 后的真实 shape |

`from_hf` / `save_hf` 入口有 assert 兜底：

```python
assert "load_spec_mapping" in self.__dict__, (
    f"{type(self).__name__}.__init__ must call self._init_load_spec() at the end."
)
```

这条约定是硬契约；子类若跳过会在第一次 load/save 时被抓。

---

## 6. 示例

### 6.1 Dense, tp=2, fsdp=4, `q_proj.weight`

```python
LoadSpec(
    name="layers.0.self_attn.q_proj.weight",
    global_hf_keys=["model.layers.0.self_attn.q_proj.weight"],
    global_shape=(n*d, h),
    fused_dim=None,
    shards=[
        ShardDescriptor(dim=0, start=tp_start,   end=tp_end,   group=tp_group),
        ShardDescriptor(dim=0, start=fsdp_start, end=fsdp_end, group=fsdp_group),
    ],
)
```

`fsdp_start/end` 相对于"已经被 TP 切过的 sub-tensor"而言，不是相对
`global_shape`。

### 6.2 MoE, ep=8, fsdp=4, fused expert weight

```python
LoadSpec(
    name="layers.0.experts.fused_w1w3.weight",
    global_hf_keys=[f"model.layers.0.mlp.experts.{i}.gate_proj.weight" for i in range(64)]
                 + [f"model.layers.0.mlp.experts.{i}.up_proj.weight"  for i in range(64)],
    global_shape=(128 * I_padded, H),    # I_padded 含 fp8 FSDP 对齐 pad
    fused_dim=0,
    shards=[
        ShardDescriptor(dim=0, start=ep_start,   end=ep_end,   group=ep_group),
        ShardDescriptor(dim=0, start=fsdp_start, end=fsdp_end, group=fsdp_group),
    ],
    origin_shape=(128 * I, H),           # 剥掉 pad 后的 checkpoint shape
)
```

RL 权重同步调用 `plan_hf_save(preserve_process_group=ep_group)` —— EP shard 被
标记 preserved，保存管道只做 FSDP 还原，结果留在 EP-local 坐标系；再由
`_request_ep_sequential_update` 按 EP rank 顺序广播。

### 6.3 embed_tokens, 纯 FSDP

```python
LoadSpec(
    name="embed_tokens.weight",
    global_hf_keys=["model.embed_tokens.weight"],
    global_shape=(V, H),
    fused_dim=None,
    shards=[ShardDescriptor(dim=0, start=fsdp_start, end=fsdp_end, group=fsdp_group)],
)
```

---

## 7. 为什么这样设计

几个关键取舍的归档。

### 7.1 为什么 `shards` 是列表而不是单轴四元组

旧的 `(dim, shard_start, shard_end, group)` 只表达一刀。TP × FSDP 或 EP × FSDP
是常见组合，旧 schema 只能靠"加载时临时推导第二刀"这种硬编码绕过（
`FSDP_SHARD_DIM == 0` 就是这条路径的残留）。列表 + DTensor 施加顺序是最小的
统一表达。

### 7.2 为什么删 `LoadEnum`

`SAME/FUSED/SHARD` 给定 `global_hf_keys` 和 `shards` 后是可派生的。保留它相当于
同一份状态的两种表达，下游分支要同步维护。直接用 `is_fused` / `is_sharded` 两个
独立 bool 可以正交表达所有组合（包括原本需要新造 `FUSED_SHARD` 的情况）。

### 7.3 为什么 fp8 不进 LoadSpec

LoadSpec 的定位是"同 dtype 下的映射"。fp8 涉及的反量化需要的是 tensor 的真实
dtype / wrapper 类型，这些只有在 IO 路径里拿到 runtime tensor 才能判断。若把
`runtime_is_float8` 放进 spec，一方面是状态重复（`is_float8_weight(tensor)` 已经
是事实来源），另一方面污染 LoadSpec 的语义 —— 它不再是纯布局描述。

`origin_shape` 是 checkpoint-visible shape。它今天只服务 fp8 runtime padding，
但仍然只携带 shape 信息；fp8 的 dtype / wrapper 判断不进入 LoadSpec。

### 7.4 为什么 `unshard_tensors_for_hf_save` 住在 `load_spec.py`

尽管它做的是分布式 all-gather，但它**只依赖 HFSavePlan + 一个通信原语**。把它
放在 `load_spec.py` 让"spec → plan → 执行"三层都在一个文件里闭环，调用方
（base.py）只需要准备 `(tensor, plan)` 对，不需要理解 shard 调度。

若将来 `unshard_tensors_for_hf_save` 进一步膨胀，可以拆到独立模块（例如
`save_runner.py`），但当前规模尚不需要。

### 7.5 为什么保存不用 `_fuse_contiguous_chunks_without_alloc`

旧代码对 `dim == 0` 的单 tensor all-gather 用过一个零拷贝 view 合并优化。这条
优化只在"一次 gather 一个 tensor"时成立 —— 当前批量 foreach 把多个 tensor 交错
塞进同一个扁平缓冲区，per-tensor chunks 不再连续，这条路径失效。换掉 NCCL 调用
次数（O(num_tensors) → O(rounds)）比 dim=0 多一次 cat alloc 更划算。如果某个
特定场景发现这次 trade-off 不值，可以单独给那条路走非批量路径，但默认策略保持
批量。

---

## 8. 测试

核心测试都在 `tests/utils/test_load_spec.py`：

- `TestLoadSpecSchema`：字段契约 + `shards` 顺序验证；
- `TestHFLoadPlan`：`plan_hf_load` 在 fused / non-fused / fp8 padding 下的产出；
- `TestHFSavePolicy`：`distributed_save` 的 HF key 分配规则。

行为等价性由 `tests/model/test_qwen3_dense.py::test_save_hf` 和
`tests/model/test_qwen3_moe.py::test_save_hf` 的 safetensors bit-equal 保证。
