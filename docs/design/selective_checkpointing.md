# Selective Checkpointing & Activation Offload

## 背景

目前 XTuner 的 recompute（gradient checkpointing）和 activation offload 都只能做到
`DecoderLayer` 级别：在 FSDP sharding 阶段，对整层 `layer` 套 `checkpoint_wrapper`
（见 `xtuner/v1/model/moe/moe.py:1039`），并由 `fsdp_config.recompute_ratio`
（`xtuner/v1/config/fsdp.py:18`）控制重算前若干层。

这种粒度太粗：一层里 attention / MoE / MLP 的显存和重算代价差异很大，无法只重算其中一部分。
本设计的目标是把 recompute 和 activation offload 下沉到 **region（埋点区间）级别**：模型作者在
forward 里用 `checkpoint_record` 埋点，每个模型在 `model_cfg` 里以 marker 区间声明「哪些段要
重算 / offload」，再由 `recompute_ratio` 控制作用到哪几层。区间可**跨模块**，不受架构边界约束。

## Goals / Non-Goals

**Goals**（按优先级排序）

- **[最高优先] 在 torch 2.10 上用非 legacy（`use_reentrant=False`）的 checkpoint 替换现有 legacy 接口。**
  这是本设计的前置项，需落实两件事：①domino EP（`intra_layer_micro_batch > 1`）下非 reentrant 能正确
  重算；②非 legacy 走 `saved_tensors_hooks`，`DecoderLayer` 可返回 TypedDict、`seq_ctx` 等可 dict 直传，
  `_check_signature_of_forward` 随之可删、model / mtp 简化。（后续 region 级 SAC recompute 也依赖此升级，
  见「SAC：region 级 recompute」。）**先做这一步，再据其结论落地后续 schema 与实现。**
- 支持 region（埋点区间）级别的 recompute，用户通过 `recompute_cfg`（语义枚举）声明要留驻哪些子结构，粒度可跨模块。
- `recompute_cfg` 是每个模型的 per-layer 默认方案，`recompute_ratio` 决定这套方案作用到哪些 layer。

**Non-Goals（本 PR 不做，但需保证不冲突）**

- **activation offload 的实现本身**。本 PR 只做 recompute，`activation_offload_cfg` 的 schema
  与 recompute 对齐、预留位置，保证后续开发不会与本次改动冲突，具体 offload 机制（落在
  `async_save_on_cpu` 上、以及与环境变量 `XTUNER_ACTIVATION_OFFLOAD` 的 BC）留到后续设计。
- 脱离代码埋点的全局 monkeypatch。recompute 区间只通过模型 forward 里显式的 `checkpoint_record`
  埋点界定，不做对任意自由函数的运行时替换。

## 第一优先：调研升级到非 legacy checkpoint wrapper

> 这是本设计的前置调研项，先于下面的 schema / 实现完成。本章只谈 wrapper 升级本身，就两件事；SAC
> 对非 legacy 的依赖与相关约束放在「SAC：region 级 recompute」里讲。

**现状**：`checkpoint_wrapper`（`xtuner/v1/model/utils/checkpointing.py`）封装 legacy 的
`torch.distributed.algorithms._checkpoint.checkpoint_wrapper` + `CheckpointImpl.REENTRANT`。
reentrant 版是一个 `autograd.Function`，要求需要求梯度的 tensor **以位置参数直传直返**；`dict` 嵌套
tensor 会被当成单个非 tensor 参数，wrapper 判定「没有输入需要梯度」→ 断梯度 / 报错。
`_check_signature_of_forward`（`checkpointing.py:28`）那套严格签名检查正是为在这种情况下早失败而存在。

升级到 `use_reentrant=False` 要落实两件事：

**① domino EP 正确性。** 现在用 `REENTRANT` 的主要原因是 domino EP（`trainer_cfg` 里
`intra_layer_micro_batch > 1`）下重算要能正常进行。先复现旧接口为何需要 reentrant / 非 reentrant
会失败的现象与根因，再验证 `use_reentrant=False` 在 `intra_layer_micro_batch > 1` 下能否正确重算。

**② TypedDict I/O + 删签名检查。** `use_reentrant=False` 用 `saved_tensors_hooks` 实现，对嵌套输入 /
返回值天然友好。升级后 `DecoderLayer` 可以**返回 TypedDict 而不是位置约定的 tuple**、`seq_ctx` /
`position_embeddings` 等可作为 dict 直传，mtp 那类第二路输出不必再挤进 tuple 某个约定位置、由下游按
index 解包。因此本步的**直接产出**包括：decoder layer 返回值 tuple → TypedDict，删除
`_check_signature_of_forward`，model / mtp 层代码随之简化。机制已用 POC（玩具模型）验证：非 legacy 下
「嵌套 dict 输入 + TypedDict 输出」的梯度与无 checkpoint **精确一致**、到达 dict 内每个 tensor；同一个
dict 进 / 出的 layer 用 legacy REENTRANT 则直接报 `element 0 of tensors does not require grad`。落到
真实 layer 需覆盖 `seq_ctx` / `position_embeddings` 等实际结构。

**验收**：两件事都到位后，domino EP 下做 grad-norm parity + 显存对比（升级前后），确认数值一致、显存
收益符合预期。

## SAC：region 级 recompute

region 级 recompute 的主体是 SAC（selective activation checkpointing）：整层套一个 checkpoint、由
policy 逐 op 裁决存或重算。用户侧的 `recompute_cfg`（语义枚举）只是这套机制最终的**表达形式**，先讲机制、再落到配置。

### 机制

recompute 落在 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False, context_fn=...)`
的 SAC 路径上，而**不是**对 submodule 逐个套 wrapper。被选中的整个 `DecoderLayer` 套**一个**
checkpoint，`context_fn` 由 torch 的 `create_selective_checkpoint_contexts(policy_fn)` 构造，
`policy_fn` 逐 op 裁决：默认 `MUST_RECOMPUTE`（全重算），被点名的区间返回 `MUST_SAVE`（该段激活留驻、
不重算）。用户看到的是 region 粒度（marker），底层是 op 粒度的 policy 在执行。

```python
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

def policy_fn(ctx, op, *args, **kwargs) -> CheckpointPolicy:
    # marker 状态机决定当前是否处于某个 SAVE 区间
    return CheckpointPolicy.MUST_SAVE if _in_save_interval() else CheckpointPolicy.MUST_RECOMPUTE

ctx_fn = lambda: create_selective_checkpoint_contexts(policy_fn)
out = checkpoint(layer_forward, *args, use_reentrant=False, context_fn=ctx_fn)
```

这条路径**必须** `use_reentrant=False`——`create_selective_checkpoint_contexts` / `context_fn`
只有非 legacy 才支持，因此它与「第一优先」的非 legacy wrapper 调研是同一件事：调研结论为真是本 schema
成立的前提。

**默认口径的选择**：policy 对每个 op 只回答「SAVE（存下来、不重算）还是 RECOMPUTE（丢掉、backward
重算）」。于是「默认 + 点名例外」有两种反向配法：

- **A**：默认 `MUST_RECOMPUTE`，点名的区间设成 `MUST_SAVE` —— 默认全重算，点名少数**不重算**。
- **B**：默认 `MUST_SAVE`，点名的区间设成 `MUST_RECOMPUTE` —— 默认全存，点名少数**重算**。

两种 SAC 都能做、也都能跨模块，差别只是 `policy_fn` 默认返回值那一行。**本设计只用 A**：recompute 的
目的是省显存，常态一定是「重算绝大多数、只把少数重算代价高的段（如 flash-attn）留驻」，所以默认全重算、
点名 SAVE 才符合直觉，`recompute_cfg` 的语义也单一。不为对称性把 B 也塞进 schema。

### 埋点：`checkpoint_record`

模型作者在 forward 里埋**点位**（不是区间），标注可寻址的语义边界：

```python
def forward(self, x):
    checkpoint_record("attn.core")
    ...
    checkpoint_record("mlp.up")
    ...
```

`checkpoint_record(name)` 是命令式点位，不在 SAC 会话内时是 no-op。它翻转一个 context-local 的
「当前活跃区间」状态机，policy 据此决定当前 op 存还是重算。

**为什么是点位 marker 而不是 `with` 上下文**：重算区间可能**跨模块**——例如 `A.a2 + B.b1`
（A 尾部 + B 头部），二者分处 `A.forward` 与 `B.forward`，中间隔着上层的 `A()→B()` 调用。`with`
块受词法作用域限制，无法横跨两个兄弟模块的 forward；命令式 marker 依赖**运行时程序顺序 + 外部状态**，
天然可跨。

### 用户配置：语义枚举，不是 marker 字符串

用户**不直接填 marker 字符串**（那反人类，且要求用户了解模型内部埋点）。面向用户的是一组**语义枚举**，
每个枚举表示「某类子结构不重算（留驻）」，与 `recompute_ratio`（选层）配合：

```python
class RecomputeUnit(Enum):
    SAVE_ATTN = auto()        # attention 段不重算、留驻
    SAVE_MOE_GATE = auto()
    ...

# model config
recompute_cfg: list[RecomputeUnit] | bool | None
```

枚举 → 具体 marker 区间的映射，仿照 `default_compile_cfg`，定义在每个 model 的默认 cfg 里（各模型的
attention / MoE 结构不同，同一个 `SAVE_ATTN` 落到的 marker 区间也不同）：

```python
@property
def default_recompute_cfg(self) -> dict[RecomputeUnit, list[tuple[str, str]]]:
    return {
        RecomputeUnit.SAVE_ATTN: [("attn.core", "attn.out")],
        RecomputeUnit.SAVE_MOE_GATE: [("moe.gate", "moe.dispatch")],
    }
```

三层职责因此各自稳定：

- **用户**：config 里挑语义枚举（`SAVE_ATTN`），不碰字符串、不需了解内部结构。
- **model 作者**：定义枚举 → marker 区间，每个模型一次、随架构而定（与 `checkpoint_record` 埋点同处一层）。
- **内部**：把选中的枚举解析成 marker 区间，喂给 policy 的状态机（下节）。

> 与 `compile_cfg` 的 `bool | None | dict` 三态一致：`None` / `True` 用 model 默认方案，`False` 关闭
> （全重算、不留驻）。是否再开放「直传区间」的高级口径（调试用）留待实现时定，默认路径只有枚举。

### 内部表示：marker 区间 → policy

枚举解析出的每个 `(start, end)` 是一段 `[start, end)` 半开区间（`end` 取「下一步起点 marker」），
区间内的 op 被标 `MUST_SAVE`。区间重叠 / 嵌套用「活跃区间集合非空即 SAVE」定义；policy 只做「命中
start 置位、命中 end 复位」的状态机，不做配对断言。

这带来一个稳健性红利：marker 区间配不平（如 `end` 落在没走到的分支）只会**多留驻一段显存，绝不影响
梯度**——SAVE 与 RECOMPUTE 两条路数值都精确正确。因此 model 作者写 `default_recompute_cfg` 时无需
担心边界配对的正确性。

### 状态生命周期（实现关键）

会话状态必须在**被 checkpoint 的 callable 顶部**重新绑定、每趟重置。因为 `use_reentrant=False`
的 backward 会**重跑整个 forward** 来重算，marker 随之重新命中；若状态挂在 module 全局而非随 callable
重置，forward 与 recompute 两趟的 SAVE 集合会错位。用 `contextvars` 在 callable 入口 set、
`finally` reset。

### 待验证的 SAC 风险点

玩具模型 POC 已验证：梯度精确、跨模块区间成立、SAVE 段确实不重算。但下面两点玩具模型覆盖不到，需在
真实 layer 上验证：

- **op 序确定性**：SAC 的 save-list 靠 forward 与 recompute 两趟 **op 顺序一一对齐**。
  `intra_layer_micro_batch > 1`（domino）下 micro-batch 交织会不会改变 op 顺序，是本方案能否落地的
  头号拦路虎。
- **torch.compile 交互**：SAC 的 `context_fn` + `saved_tensors_hooks` 在 compile 下的行为需单独确认。

## 寻址语义：与 `recompute_ratio` 的两步分工

生效范围两步式，两者正交、不是覆盖关系：

1. **选层**：`recompute_ratio` 决定哪些 layer 套上 SAC checkpoint（沿用现有 `_should_recompute` 语义）。
2. **选区间**：在被选中的 layer 内部，`recompute_cfg` 选中的枚举解析成 marker 区间，抠出**留驻不重算**的段，其余全重算。

即 `recompute_ratio` 定义「模板铺到前多少层」，`recompute_cfg` 定义「每层内部哪些段不重算」。

> 旧设计的「按 module 类匹配 + 最外层去重」不再适用：SAC 下整层只有**一个** checkpoint，不存在
> 嵌套 wrapper 的双重 checkpoint 问题，SAVE / RECOMPUTE 由 policy 逐 op 裁决，没有「谁套谁」。

## recompute 与 activation offload 的关系

两者不互斥，但作用对象正好互补：SAC 下 offload 的对象是被**留驻的那部分激活**（`MUST_SAVE` 段）
——被重算的段本就不落地、无需 offload，能省显存的只有留驻下来的 SAVE 张量。

本 PR **只实现 recompute**。对 offload 只有一个约束：`activation_offload_cfg` 复用 `recompute_cfg`
同一套语义枚举面向用户、同一套「枚举 → marker 区间」的 model 默认映射机制，且 SAC 的 saved-tensor
注入点要为后续 offload 的 hook 机制（预期落在 `async_save_on_cpu` 上、按留驻 tensor 建 saved-tensor
hook）留出干净入口，不写死实现细节。offload 的具体机制与环境变量 BC 留到后续设计文档。

## FSDP 包裹顺序

FSDP（`fully_shard`）永远是**最外层**：先对 `DecoderLayer` 套上 SAC checkpoint，再对该 layer
做 `fully_shard`。由于 `fully_shard` 以整层为单位、SAC checkpoint 也以整层为边界，backward 重算
发生时该层参数已由 FSDP all-gather 就位，重算路径无需额外的 gather 时机处理。

## 环境

torch 2.10 环境路径：`/mnt/shared-storage-user/yehaochen/miniconda3/envs/py312-pt210/bin/torchrun`

## 开发计划：分工与 stacked PR

**硬约束**：非 legacy 升级（第一优先）是**门禁**，不能与后续并行——domino EP 下
`use_reentrant=False` 能否正确重算的结论，决定整套 SAC schema 是否成立。因此是「门禁串行 → 通过后扇出」，
不是一次并行 N 块。

切成 **3 个 owning agent / 3 层 stacked PR**：

| 层 | agent | 内容 | 主要文件面（尽量不重叠） |
| --- | --- | --- | --- |
| **A（base）** | 非 legacy 基座 | wrapper → 非 legacy + domino EP 验证；`DecoderLayer` 返回 tuple → TypedDict、`seq_ctx` dict 直传、删 `_check_signature_of_forward`、改 model / mtp 调用点。**并定义共享契约**：`checkpoint_record`（先落 no-op 桩）、`RecomputeUnit` 枚举、枚举解析出的区间类型 | `checkpointing.py`、decoder layers、model / mtp、sharding 路径 |
| **B（stack 在 A 上）** | SAC 引擎 | `checkpoint_record` 背后的 `contextvars` session + `policy_fn` + `create_selective_checkpoint_contexts` 接线；FSDP 路径换成整层 SAC；op 序 / `torch.compile` 在真实 layer 上验证 | 新 SAC 引擎模块 + 包裹调用点 |
| **C（stack 在 A 上，与 B 并行）** | 配置 + 逐模型 | `recompute_cfg` 三态解析（`bool \| None \| list[RecomputeUnit]`）+ 与 `recompute_ratio` 打通；各模型 `default_recompute_cfg`；在模型 forward 里埋 `checkpoint_record` | config 模块 + 各 model 文件 |

**为什么合并原「非 legacy 升级」与「TypedDict 重构」为一个 A**：两者都要改 `checkpointing.py`、decoder
layer、sharding 路径，拆两个 agent 会在同一批文件上冲突；同一 owner 一次到位。

**为什么 B、C 能真并行**：B 只碰引擎与包裹点、不碰模型 forward；C 只碰 config 与模型文件、不碰引擎内部。
C 埋的 `checkpoint_record` 在 B 合入前就是 no-op，无害；两者只在 A 的契约上相遇。

**唯一 seam**：FSDP 包裹处「把解析出的区间应用到选中层」这一行归 **B** 拥有（它是机制）；C 只负责把
config 解析成区间、通过契约函数交给 B，不直接改包裹点，避免 B / C 同时改 sharding 路径。

**stack 顺序**：`A → B`、`A → C`，B / C 互为 sibling；A 合入后各自 rebase 到 A，平行推进。
