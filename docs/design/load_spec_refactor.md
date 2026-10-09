# LoadSpec、DTensor 与 DCP 设计

## 结论

HF checkpoint、DTensor runtime layout 和 DCP checkpoint 描述的是同一个逻辑
tensor，但面向的协议不同：

- `LoadSpec` 记录 checkpoint 映射与 runtime 分片的稳定事实。
- `HFLoadPlan` / `HFSavePlan` 把这些事实编译成当前 rank 可直接执行的 HF I/O
  程序。
- `ShardDescriptor` 同时描述 continuous shard 和 Expert TP even interleave。
- DCP 不经过 HF plan；它用 `compute_runs()` 把一个离散 local shard 展开成多个
  标准全局 chunk，从而保持分布式保存、原地加载和跨拓扑 reshard。

~~~mermaid
flowchart LR
    A["HF keys + canonical runtime layout"] --> B["LoadSpec：稳定事实"]
    B --> C["HFLoadPlan：读取与 local copy"]
    B --> D["HFSavePlan：逆分片与 padding trim"]
    E["DTensor placements"] --> B
    E --> F["compute_runs"]
    F --> G["DCP WriteItem / ReadItem"]
~~~

核心边界是：HF plan 负责“checkpoint 格式与 runtime tensor 之间如何搬运”，DCP
planner 负责“local storage 对应哪些全局坐标”。两者复用相同的布局数学，不强行共用
协议对象。

## 一、LoadSpec 与 Plan 的职责

### 1.1 为什么需要两层对象

可以把 `LoadSpec` 理解成地图，把 Plan 理解成一次具体行程：地图不会保存“今天从哪条
路走”，而同一张地图可以生成完整保存、保留 EP 或只 gather FSDP 等不同路线。

| 对象 | 负责 | 不负责 |
| --- | --- | --- |
| `LoadSpec` | 参数名、HF keys、fused dim、runtime/global shape、checkpoint 可见 shape、forward shard history | 当前 rank 的 copy offsets、collective 顺序、save policy |
| `HFLoadPlan` | 选择本 rank 所需 keys、拼接、调用 canonical adapter、执行 copy regions、清零 runtime padding | 推断模型格式、执行 collective |
| `HFSavePlan` | 标记 preserved shards、逆序 gather、最终 FP8 trim、输出 keys | 保存稳定布局事实 |
| `SaveShardStep` | 一个 descriptor 的逆操作、操作前 runtime shape、是否 preserved | 模型格式转换、跨参数策略 |
| 模型 adapter | HF layout 与 XTuner canonical layout 的 transpose/reshape/flatten | EP、ETP、FSDP、rank 和 process group |

Plan 生成后，executor 不再回读 `LoadSpec`。因此 Plan 是 self-contained 的
rank-local 程序，也可以独立测试。

### 1.2 LoadSpec 保存哪些 shape

- `global_shape`：所有 runtime shard 之前的完整 shape，可能包含 FP8 kernel 所需的
  尾部 padding。
- `origin_shape`：HF checkpoint 实际可见的 shape；没有 FP8 padding 时为空或等于
  `global_shape`。
- `local_shape`：真实 runtime local tensor shape，用于校验 descriptor 推导结果。

`LoadSpec` 不保存旧设计中的 `explicit_owned_regions` 或
`save_requires_full_reconstruct`。regions 属于某次 load plan 的编译结果；是否 gather
以及保留哪个 group 属于某次 save plan 的策略。

## 二、统一的分片描述

### 2.1 ShardDescriptor

```python
class ShardDescriptor(BaseModel):
    dim: int
    group: dist.ProcessGroup
    interleave_factor: int = 1
```

- `interleave_factor == 1`：普通 continuous `Shard`，支持 PyTorch 原有的 uneven
  分片。
- `interleave_factor > 1`：Expert TP even interleave；factor 是每个 rank 持有的
  有序连续 run 数。
- descriptor 按 full runtime tensor 到 local tensor 的语义顺序排列，典型顺序是
  `EP continuous → ETP interleave → FSDP continuous`。
- save 严格反向执行：`FSDP gather → ETP gather/deinterleave → EP gather`。

even interleave 设当前维长度为 `N`，group size 为 `M`，factor 为 `F`，要求：

```text
N % (M * F) == 0
run_size = N / (M * F)
```

rank `r` 持有第 `(j * M + r)` 个 run，`j = 0 ... F - 1`。当前不支持 uneven
interleave；不整除会明确报错。

### 2.2 DTensor placement 如何表达 Expert TP

MoE column-parallel 权重需要 EP 分 expert、TP 再在每个 expert/projection 内部分列：

```python
placements = (
    Shard(0),
    InterleavedShard(0, num_local_stripes=local_experts * fused_projections),
)
```

`InterleavedShard` 继承 PyTorch 私有 `_StridedShard`。其 `split_factor` 不是“当前
rank 数”，而是当前切分前已经存在的逻辑 stripe 数；在这里就是本 EP rank 的
`expert × projection` 数。

DTensor 构造时 placements 按 mesh 维从左到右作用；从 placements 推导 carving
order、HF save 重建时则按相反方向撤销。FSDP2 还会在 mesh 最左侧 prepend 一个
`_StridedShard` 标签，但它实际切的是已经完成 EP/ETP 的 local parameter。因此
`LoadSpec.from_tensor()` 在唯一的转换边界上做两件事：

1. 真正的 `InterleavedShard` 转成 `interleave_factor > 1`。
2. FSDP prepend placement 归一化成最后应用的 continuous descriptor。

后续 plan 不再依赖 `_StridedShard` 类型或 `DTensorSpec.shard_order`。

### 2.3 通俗例子：两位仓库员各拿每个货架的一半

设 fused W1/W3 的 dim-0 共 16 行，业务顺序为：

```text
expert 0 gate: [0,1,2,3]    expert 0 up: [4,5,6,7]
expert 1 gate: [8,9,10,11]  expert 1 up: [12,13,14,15]
```

`EP=2, TP=2` 时，每个 EP rank 只有一个 expert，因 gate/up 融合所以
`interleave_factor=2`：

| `(ep,tp)` | local tensor 对应的 global rows |
| --- | --- |
| `(0,0)` | `[0,1,4,5]` |
| `(0,1)` | `[2,3,6,7]` |
| `(1,0)` | `[8,9,12,13]` |
| `(1,1)` | `[10,11,14,15]` |

这像两位仓库员分别负责每个货架的左半或右半，而不是一人搬走完整货架。若误用第二个
`Shard(0)`，TP rank 会拿到连续的完整 projection 子集，gate/up 的列并行语义就错了。

这种 `(Shard, InterleavedShard)` 同维布局在部分 PyTorch 版本无法归约出
`shard_order`，因此 `full_tensor()` / `redistribute()` 不能作为稳定实现。XTuner
训练直接使用 `DTensor.from_local()` / `to_local()`，HF 和 DCP 则使用本文的自定义
布局编译路径。

## 三、HF Load

### 3.1 Plan 生成

`LoadSpec.plan_hf_load()` 从完整 canonical tensor 开始，按 descriptors 正向计算
当前 rank 的 ownership：

1. continuous shard 生成一个 slice；uneven 时复用 PyTorch shard size/offset
   语义。
2. even interleave 生成多个有序 runs。
3. 后续 FSDP continuous shard 切的是已拼接的 ETP-local tensor，segment compiler
   再把它映射回 global runs。
4. 用 `origin_shape` 裁掉 checkpoint 中不存在的 FP8 padding。
5. 选择覆盖这些 regions 的最小 HF-key envelope，并编译为
   `HFLoadPlan.copy_regions`。

编译使用连续 segments 表达 ownership，不会按 tensor 大小展开逐元素索引。

### 3.2 固定执行顺序

~~~mermaid
flowchart LR
    A["读取 plan.hf_keys"] --> B["沿 fused_dim 拼接"]
    B --> C["模型 hf_tensor_to_canonical"]
    C --> D["校验 canonical shape"]
    D --> E["执行 source-to-local copies"]
    E --> F["未写 FP8 padding 保持为 0"]
~~~

必须先 canonicalize 再按 ownership copy。例如 GPT-OSS 的 HF
`gate_up_proj` 先 transpose、重排 gate/up 并 flatten，之后 dim-0 才与
`ShardDescriptor` 描述的 expert-major runtime 维一致；先 slice HF tensor 会切错
语义维。

### 3.3 模型差异被限制在 adapter

| 模型 | HF → canonical 的主要差异 |
| --- | --- |
| GPT-OSS | `gate_up_proj` transpose、gate/up 重排并 flatten；`down_proj` transpose；bias 同步重排 |
| Qwen3.5 | 非 MTP expert 权重主要 flatten expert 维；MTP 保留自己的布局 |
| Qwen3-VL | `gate_up_proj` 的 HF 维序不同，需要 transpose 后 flatten |
| GLM | 一个 fused XTuner 参数对应多个 per-expert HF keys，先拼接再 flatten |

新增模型只需提供 key mapping 和 canonical adapter，不复制 EP/ETP/FSDP load
流程。

## 四、HF Save、padding 与重建

### 4.1 逆分片

`plan_hf_save()` 为每个 descriptor 生成 `SaveShardStep`，executor 逆序执行所有未
preserve 的 step：

- continuous：把各 rank local shard 补到 collective 所需等长，all-gather 后按
  rank concat，再 trim 到该 step 的 `shape_before_shard`。
- even interleave：各 rank/run 已等长；all-gather 后把 rank-major 数据重排成
  `[run][rank]`，不需要 collective padding。

例如 `TP=2, F=2`，gather 得到：

```text
rank0: [A0, A2]
rank1: [A1, A3]
```

deinterleave 后必须为 `[A0, A1, A2, A3]`，不能像 continuous shard 一样直接按
rank concat。

### 4.2 collective padding 与 FP8 padding

两类 padding 分层处理：

| padding | 原因 | 处理位置 |
| --- | --- | --- |
| collective padding | uneven continuous ranks 的 collective 输入必须等长 | 当前 continuous `SaveShardStep` merge 后 trim 到 runtime `shape_before_shard` |
| FP8 padding | runtime kernel 对齐，HF checkpoint 不存在这些尾部元素 | 所有目标 gather 完成后，从 `runtime_output_shape` 一次 trim 到 `output_shape` |

通俗例子：runtime 长度 10、HF 可见长度 9，以 3 ranks continuous shard。三个
local 长度是 `4, 4, 2`，第三个 rank 先临时补成 4；gather 得到长度 12 后先 trim
到 runtime 长度 10，这一步只去 collective padding；所有逆分片结束后再 trim 到
HF 长度 9，这一步才去 FP8 padding。这样外层 gather 不会把已经提前裁掉的 FP8
尾部重新当作 collective padding 补回来。

load 侧没有 collective trim：plan 只复制 `origin_shape` 内存在的数据，并先把未写
runtime padding 清零。

### 4.3 Save policy

| 场景 | preserved shards | 实际逆分片 | 输出语义 |
| --- | --- | --- | --- |
| full / distributed HF save | 无 | FSDP → ETP → EP | 完整 checkpoint-visible canonical tensor |
| preserve EP | EP | FSDP → ETP | 连续的 EP-local canonical tensor，供 RL expert-key sync |
| only gather FSDP | EP、ETP | 仅 FSDP | 完整 EP/ETP-local runtime tensor，供 layer-wise IPC |

`distributed_save` 是写出分配策略，不是部分重建策略；当前仍先正确重建完整
canonical tensor，再决定哪些 rank 写哪些 keys。

### 4.4 reconstruct_full_tensor

`reconstruct_full_tensor(dt)` 仍是 public convenience API，因为 PyTorch
`DTensor.full_tensor()` 不稳定支持 `(Shard, InterleavedShard)`。它现在只是薄封装：

1. 从 DTensor placements 构造 descriptor history。
2. 生成不 preserve 任何 shard 的 full-runtime save steps。
3. 调用与 `HFSavePlan` 相同的逆分片 executor。
4. 返回 `global_shape` 的 runtime tensor。

它不做模型格式转换，也不 trim `origin_shape`；`distributed_save` 和 preserve group
属于 `HFSavePlan` 策略，不进入这个 API。生产 HF save 不再通过
`save_requires_full_reconstruct` 走特殊旁路。

## 五、DCP 的独立协议

### 5.1 为什么默认 DCP 不够

普通 DTensor rank 可由一个 `local_shape + global_offset` 描述；interleaved rank
对应多个不连续区间。若把 `[0,1,4,5]` 错当成从 offset 0 开始的连续四行，DCP 会把
后两行写到 global rows 2、3，checkpoint 看似成功但内容错误。

`compute_runs()` 只根据 global shape、mesh coordinate 和 placements 做几何计算，
不读取 tensor value，也不发 collective。它把 local tensor 拆为若干“local 连续且
global 也连续”的 `Run`：

```python
Run(
    global_offset=(global_row, 0),
    sizes=(num_rows, hidden_size),
    local_start=local_row,
    local_size=num_rows,
)
```

### 5.2 通俗例子：给一本交错装订的书编页码

沿用 `(ep=0,tp=0)` 的 local rows `[0,1,4,5]`。它在内存中是连续四行，但
`compute_runs()` 返回两段：

```text
Run(global rows [0,2), local rows [0,2))
Run(global rows [4,6), local rows [2,4))
```

就像一本小册子依次装订了原书第 0、1、4、5 页；保存时必须在目录中记录成两段，
不能声称它是原书第 0 到 3 页。

Save planner 为它生成：

```text
Chunk(offset=(0,0), size=(2,H)) <- local.narrow(0, 0, 2)
Chunk(offset=(4,0), size=(2,H)) <- local.narrow(0, 2, 2)
```

Load planner 在当前目标拓扑重新计算 runs，让 DCP 用 saved chunk 与 destination
chunk 的全局坐标求交，并把数据直接写入对应 local `narrow()` view。

~~~mermaid
flowchart LR
    A["Interleaved DTensor local storage"] --> B["compute_runs：多个全局连续区间"]
    B --> C["SavePlanner：每个 run 一个 WriteItem"]
    C --> D["DCP metadata + distributed storage"]
    D --> E["LoadPlanner：目标拓扑 runs"]
    E --> F["saved/destination chunk 求交"]
    F --> G["原地写入 local views"]
~~~

### 5.3 分布式保存与跨拓扑恢复

- 每个 rank 只写自己已有的 local views，不 materialize full tensor，也不把权重集中
  到 coordinator。
- metadata 保存 FQN、global shape、offset 和 size，不保存“rank N 的业务含义”。
- 因此可执行 `save EP=2,TP=2 → load EP=1,TP=4`：新 TP ranks 按当前 placements
  声明目标 runs，DCP 从旧 chunks 中读取相交区域。
- model 参数和 optimizer 的 `exp_avg` / `exp_avg_sq` 都使用相同 planner，才能完整
  resume。
- `TrainEngine.save_dcp/load_dcp` 显式传入
  `InterleavedShardSavePlanner/LoadPlanner`；正确性不依赖 Torch 2.7 的可选 monkey
  patch。

`InterleavedShardSavePlanner` 继承 `XtunerCacheSavePlanner`，但 cache 只复用 plan、
metadata、`WriteResult` 等 control-plane 信息，不比较权重值，也不会省略本次 tensor
写盘。显式构造 planner 的默认路径当前以正确性为主，并未开启 plan cache。

### 5.4 与 HF plan 的边界

HF 要处理 key 拼接、canonical adapter、save policy 和 FP8 trim；DCP 已有自己的
`WriteItem` / `ReadItem`、global metadata 和 storage 协议。因此 DCP 不调用
`reconstruct_full_tensor`，也不经过 `HFSavePlan`。二者只共享 even interleave 的
几何语义。

## 六、约束与验证

当前有意保留以下边界：

- HF `ShardDescriptor` 支持 uneven continuous，但 ETP 只支持 even interleave。
- `compute_runs()` 当前只支持 tensor dim-0 sharding，strided run 要求均匀切分。
- `InterleavedShard` 和 DCP planner 使用部分 PyTorch 私有 API，升级 PyTorch 时需要
  回归。
- FSDP prepend placement 必须在 EP/ETP 后应用；不能按 placements 的表面顺序直接
  推导数据 ownership。

行为测试应覆盖：continuous even/uneven、ETP even interleave、EP→ETP→FSDP、
FP8 load padding、collective+FP8 save padding、full/preserve EP/only gather FSDP、
`reconstruct_full_tensor`、DCP 同拓扑 round-trip 与跨拓扑 reshard，以及 GPT-OSS、
Qwen3.5、Qwen3-VL、GLM 的 canonical adapter。

## 总结

当前设计把 Expert TP 的特殊性收敛到三个清晰位置：

1. `ShardDescriptor.interleave_factor > 1` 描述 HF I/O 所需的 even runs。
2. HF load segment compiler 与 save deinterleave 分支负责 rank-local 数据搬运。
3. DCP `compute_runs()` 把离散 storage 转成标准全局 chunks。

模型与 `BaseModel` 主流程不再判断 `InterleavedShard`，HF save 和
`reconstruct_full_tensor` 也不再维护两套重建算法。continuous 与 interleave 共用
布局描述、计划调度和 padding 边界，但保留各自最直观的 merge 算法。
