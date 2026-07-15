# FSDP 梯度 reduce-sum 改造设计

## 1. 背景

当前训练里每个可训练参数拿到的梯度,是**全局 token-mean 梯度**(等价于 sp=1、dp=1、grad-acc=1 下单卡算出来的梯度)。这个正确结果不是某一处显式算出来的,而是靠**前向归一化 + 三处 ×world_size 注入 + 三处 reduce 的除法**精确相消得到的:

**前向侧(loss 归一化,不变)**
- `CELossContext.build_batches` 里对 loss_weight 除以全局 token 数 `global_denominator`(WORLD 的 `all_reduce(SUM)` 得到)。所以每个 rank 的 loss 已经是"本地 token 求和 ÷ 全局 token 数"。

**梯度侧(三因子相消,本次要拆掉)**

| 反向注入 ×world_size 的位置 | 抵消它的归约 |
|---|---|
| `CELossContext.forward` 末尾 `all_reduce(SUM, WORLD)`(autograd 版,反向也是 `all_reduce(SUM)`,标量 grad_output=1 → 每个 rank 注入 ×world_size) | FSDP2 reduce-scatter 默认 `ReduceOp.AVG`(÷fsdp_size) |
| 同上,专家参数只在 experts_fsdp 子维累加,残留 ×ep_size | `MoE.scale_and_reduce_grad` 对 `.experts` 参数 `div_(ep_size)` |
| `BalancingLossContext.finalize` 用 `all_reduce_autograd(SUM)`;`ZLossContext.accumulate` 显式 `× world_size` | `MoE.scale_and_reduce_grad` 对 replicated 参数 `div_(flat_mesh.size())` 后再 `all_reduce(SUM)` |

三条链路各自 ×world_size 再 ÷(相应 reduce 维度),乘积等于 1,得到正确的全局 mean 梯度。

**问题**:这套机制把"loss 语义"和"梯度归约实现"耦合在一起,最典型的就是 `scale_and_reduce_grad` 里对专家梯度 `div_(ep_size)` 这种只有读过全链路才能理解的补偿缩放。任何新并行维度、新 loss 项都要重新推一遍 world_size 的相消关系,极易出错。

## 2. 目标

把梯度归约统一成**纯 SUM**,让梯度正确性只依赖一条可局部推理的不变式,消除所有 ×world_size 注入和补偿性除法。

**改造后的不变式**
1. 前向 loss 归一化**保持不变**:每个 rank 的 loss 仍是"本地 token 求和 ÷ 全局 token 数",即全局 loss 的本地分量。
2. 任何 loss 项都**不再向反向注入 ×world_size**(既不做 autograd 的 WORLD all_reduce,也不显式乘 world_size)。
3. 所有跨 rank 的梯度归约(FSDP reduce-scatter、`scale_and_reduce_grad` 的手动 all_reduce)一律 **SUM,不做任何除法**。
4. 对外展示/日志需要的"全局 loss 标量",用 **detached** 的 all_reduce 单独算,不进入 autograd 图。

在此不变式下,每个 rank 反向得到本地分量的梯度,跨 rank SUM 后自然等于全局 loss 对该参数的梯度,无需任何补偿。

## 3. 非目标

- 不改变 loss 的前向数值语义(loss 曲线、归一化分母口径不变)。
- 不改变 EP / FSDP / SP 的 mesh 布局与切分方式。
- 不处理 pipeline parallel 的梯度归约(PP 有独立的跨 stage 归约路径,需单独评估,见 §6 风险)。
- 不把 `noaux_router` 的 `e_score_correction_bias` 零初始化修复纳入本次改动(独立 bug,单独 PR)。

## 4. 总体方案

四组改动,**必须作为一个原子变更同时落地**——只改其中一部分会破坏三因子相消,产生错误但不报错的梯度:

1. **FSDP reduce-scatter 改 SUM**:所有 `FSDPModule` 设置 gradient divide factor = 1 且强制 sum 通信。
2. **`scale_and_reduce_grad` 去除补偿除法**:专家参数不再 `div_(ep_size)`,replicated 参数不再 `div_(flat_mesh.size())`,只保留 SUM all_reduce。
3. **三项 loss 去除 ×world_size 注入**:CE / balancing / z-loss 统一改为"本地分量 + 全局 detached 分母",不再走 autograd all_reduce 或显式乘 world_size。
4. **日志 loss 单独 detached 归约**:返回给 trainer 的展示 loss 由本地分量经 detached all_reduce 还原为全局值。

## 5. 详细设计

### 5.1 FSDP reduce-scatter 改 SUM

**改法**。对模型里每一个被 `fully_shard` 包裹的 `FSDPModule`,在分片完成后成对调用:

```python
module.set_gradient_divide_factor(1.0)
module.set_force_sum_reduction_for_comms(True)
```

两个都是 torch 2.10 的公开 API(`torch.distributed.fsdp._fully_shard`)。

**为什么必须成对**(关键坑,已在 torch 2.10 实测,见 §7)。单独 `set_gradient_divide_factor(1.0)` + `reduce_dtype=bf16` 会**静默把所有梯度清零**:factor≠group_size 时 FSDP 内部走 `_make_nccl_premul_sum(1/factor)`,而 NCCL PreMulSum 在 bf16 下返回全零(2.10 未修)。`set_force_sum_reduction_for_comms(True)` 让通信改走 `_div_if_needed`(factor=1 时是 no-op)+ 纯 `ReduceOp.SUM`,绕开 PreMulSum,bf16 下得到精确 SUM,且**无需升 fp32、不损带宽**。

**落点**。在 `BaseModel` 上新增一个公开方法(如 `set_gradient_reduce_sum()`),遍历 `self.modules()` 挑出 `FSDPModule` 实例统一设置;在 `BaseModel.fully_shard`、`MoE.fully_shard`、`DenseModel.fully_shard` 这三个分片入口的**根方法末尾各调用一次**。由于 `fully_shard` 是就地包裹,嵌套分片后的子 `FSDPModule` 仍在 `self.modules()` 里,一次根级遍历即可覆盖逐层分片的所有模块。

**版本约束**。这两个 API 从 torch 2.10 起提供。方法入口处应做一次可用性检查(`hasattr`),缺失时抛出明确错误说明需要 torch ≥ 2.10,而不是让 factor 静默回退成清零路径。

### 5.2 `MoE.scale_and_reduce_grad` 去除补偿除法

该方法在所有 micro-batch 反向之后、clip_grad_norm 之前调用,负责处理 FSDP **不**归约的那部分参数。参数分三类,改法各异:

- **FSDP 分片参数(Shard placement)**:由 §5.1 的 reduce-scatter 处理,本方法不碰。
- **专家参数(名字含 `.experts`)**:每个专家独占一个 EP rank,跨 EP 无需归约;其 FSDP 分片只在 experts_fsdp 子维,已由 §5.1 SUM 归约。**删除** `div_(ep_size)` 分支里的缩放(直接 `continue`)。
- **replicated 参数(含 Replicate 维的 DTensor)**:FSDP 不归约,由本方法手动 all_reduce over replicate group。**删除** all_reduce 前的 `grad.div_(flat_mesh.size())`,保留末尾 coalesced 的 `all_reduce(SUM)`。

分桶 + coalesced all_reduce 的结构不变,只删两处 `div_`。**同步更新注释**:现有注释写的是 "keep the effective average" / "yields the mean across replicas",改后不再是 mean,注释必须一起改,否则会误导(stale comment)。

**验证点**:确认 `_fully_shard` 里被 `ignored_params` 排除的 fp32 参数(dense 的 fp32_keys / lm_head)的梯度归约路径是否覆盖在上述三类中。这些参数不走 FSDP reduce-scatter,若它们是 replicated DTensor 则由本方法处理;若存在其它归约路径,需确认同样是无除法的 SUM。

### 5.3 三项 loss 去除 ×world_size 注入

三项改动是**同构**的:去掉 ×world_size 的来源,让每个 rank 的 loss 停留在"本地分量"这一层,全局分母保持 detached。

- **CE**(`CELossContext.forward`):删除末尾对 loss 的 autograd `all_reduce(SUM, WORLD)`。返回的 loss 从"全局值"变成"本地分量"。前向的 `build_batches` 里 `loss_weight /= global_denominator` 保持不变(那是分母,不是注入)。

- **Balancing loss**(`BalancingLossContext.finalize`):把 `routing_weights_sum_global = all_reduce_autograd(local_gating_sum, ...)` 改为**直接使用 `local_gating_sum`**;`seqlen_global` / `tokens_per_expert_global` / `scale_global` 这些**已经是 detached 的全局分母,保持不变**。这样 loss 变成"本地路由权重和 ÷ 全局 seqlen",即全局 balancing loss 的本地分量,梯度只经本地部分,交由 §5.1/§5.2 的 SUM 归约聚合。数学上与 CE 完全一致:`all_reduce_autograd` 是唯一带梯度的跨 rank 注入,去掉即可。

- **Z-loss**(`ZLossContext.accumulate`):删除全局平均分支里的 `× world_size`(即 `loss * num_tokens_local / denom_global`,不再乘 world_size)。相应地 `accumulate` 签名里的 `world_size` 参数变为无用,应一并删除并清理调用方传参。

改后三项 loss 的前向数值都从"全局"变为"本地分量",其**跨 rank 之和**才等于原来的全局 loss —— 这正是 §5.4 日志归约要还原的量。

**aux loss 只有 global-average 一种模式**。历史上 balancing / z-loss 有 `*_global_average` 开关切换"全局平均 / 每 rank 局部"两种模式;后者在 reduce-sum 下(replicated router 梯度按 SUM 聚合)会把梯度放大 world_size 倍,语义与 reduce-sum 不兼容,已连同开关一并删除。现在分支条件只看 `dist.is_initialized()`:分布式走全局平均(用 reduced 统计量),**单进程(dist 未初始化,如 reference/eval)是它的 W=1 特例**——此时 `tokens_per_expert_global == tokens_per_expert_local`、`seqlen_global == valid_tokens`、`scale_global` 与 mean 逐项相等,故单进程分支数值与全局分支在 W=1 下完全一致,保留它只为在无 process group 时仍可运行。

### 5.4 日志 / 展示 loss 的 detached 归约

**统一原则(不变式 #4)**:所有对外展示的 loss 标量一律还原为**全局值**,通过 detached 归约实现,不进入 autograd 图。改造前后展示曲线数值不变。

**当前展示管线的隐患(务必修对,否则曲线静默掉一个 world_size 因子)**。今天每个 rank 的 loss 前向数值已是**全局值**(CE 靠 autograd all_reduce、balancing 靠 `all_reduce_autograd`、z-loss 靠 `× world_size` 与 `/world_size` 相消),而 `BaseModel.post_micro_batch_forward` 对每个 `*_loss` 字段做的是 **mean-over-ranks** 归约(`all_reduce(t.div_(world_size), SUM)`),`mean(全局)=全局` 才对得上。§5.3 把三项 loss 变成**本地分量**后,`mean(本地)=全局/world_size`,`reduced_llm_loss` / `reduced_balancing_loss` / `reduced_z_loss` 以及 train_step 的 `total_loss`(trainer 里标为 `local_loss`)会全部掉 world_size 倍。

**改法**。展示值必须由每项 loss 的**本地分量**经 detached **SUM** 归约还原(不是 mean),因为改造后 `SUM(本地)=全局`:
- **CE** 已有本地分量记录(`extra_info["local_base_loss"]`),直接用它做 detached SUM。
- **Balancing / z-loss 需新增 detached 本地分量 log record**:balancing 的全局标量**无法**从其数值反推本地分量(本地值依赖每 rank 的 `local_gating_sum` 向量,不是均匀标量因子),因此必须在 `finalize` / `accumulate` 内单独计算并保存一份 detached 的本地分量;z-loss 现有的 `_running_loss_for_log` 里烘焙了 `× world_size`,不能直接用,同样改存本地分量。
- **展示归约由 mean 改为 SUM**:`post_micro_batch_forward`(及 train_step 的 `total_loss` 展示口径)对上述本地分量做 detached `all_reduce(SUM)`,不再 `div_(world_size)`。
- **用于 backward 的 loss 保持本地分量**(靠 §5.1/§5.2 的 FSDP/手动 SUM 聚合梯度),与用于展示的 detached 记录**分属不同张量**,不可混用。

**C2/C3 边界的落法**(见 §9):C2 在 backward 仍为全局的状态下,先新增本地分量 log record 并把展示切到"detached SUM 本地分量"——此时 `SUM(本地)` 恰等于今天的全局展示值,**曲线不变,行为中性**;C3 再翻转 backward 为本地分量,展示已提前指向本地分量,曲线保持全局不变。

## 6. 风险

- **bf16 静默清零(最高)**:FSDP factor=1 单独设置在 bf16 下清零梯度且不报错。缓解:强制 `set_force_sum_reduction_for_comms(True)` 成对调用 + 版本检查(§5.1)。回归测试必须在 **bf16** 下断言梯度非零且等于 SUM,不能只测 fp32(fp32 会掩盖此 bug)。

- **部分改动导致隐性错误梯度(高)**:四组改动任一遗漏(例如只改 FSDP 不改 loss,或漏掉 z-loss 的 world_size),都会让某类参数的梯度尺度偏差一个 world_size / ep_size 因子,**不报错**,只表现为收敛异常。缓解:作为原子 PR 同时落地;用 §8 的 parity 脚手架在 EP=1 / EP=2 双配置下断言 grad_norm 比值。

- **compose 模型的 fully_shard 覆写(中)**:qwen3_vl vision、intern_s1 等在各自 modeling 文件里直接调 `fully_shard`。若它们覆写了 `fully_shard` 且未走 §5.1 的根级 post-pass,其子模块的 reduce factor 不会被设置,退回 AVG,与主干 SUM 不一致。缓解:post-pass 遍历 `self.modules()` 从根模型统一设置,可覆盖作为子模块的 vision/projector;需逐个确认 compose 模型的分片入口最终会触发一次根级 post-pass。

- **ignored_params 的 fp32 参数(中)**:被 FSDP 排除的 fp32 参数不走 reduce-scatter,其归约路径需确认落在 `scale_and_reduce_grad` 的无除法 SUM 分支内(§5.2 验证点)。

- **PP 未覆盖(中)**:pipeline parallel 有独立的跨 stage 梯度处理,本设计不改。若与 PP 组合使用,需单独确认 PP 侧不依赖当前的 AVG 语义。列为非目标。

- **grad accumulation / intra-layer micro-batch(低)**:多 micro-batch 下每次 backward 都 reduce-scatter SUM 并累加进 `.grad`,`scale_and_reduce_grad` 在全部 backward 后统一手动归约。SUM 对累加天然可组合,无额外改动,但回归测试应含 grad-acc > 1 的用例。

## 7. torch 2.10 实测依据

`scratchpad_reduce_sum_fsdp210.py`(2 卡,bf16 reduce),四种配置对同一组"每 rank 不同输入"的梯度:

| 配置 | 结果 |
|---|---|
| 默认(factor=None) | = MEAN(现状 AVG) |
| `set_gradient_divide_factor(1.0)` 单独 | **全零**(PreMulSum bf16 bug) |
| factor(1.0) + `set_force_sum_reduction_for_comms(True)` | = **SUM 精确**(采用) |
| factor(1.0) + `reduce_dtype=fp32` | = SUM 精确(可用,非必需) |

结论:bf16 下 force-sum 是干净解,无需升 fp32。

## 8. 验证方案

- **单元级(bf16 必测)**:2 卡小 FSDP 模型,断言 reduce 后梯度 == 各 rank 本地梯度之和(非零、非 mean)。即 §7 的固化版本。
- **端到端 parity**:`scratchpad_ep_grad_parity.py` 脚手架,分布式全梯度 `full_tensor()` vs 单进程全局 batch 手工 token-mean 参考,EP=1 / EP=2 断言全局 grad_norm 比值 ≈ 1(噪声地板实测约 0.15%,逐参数 gate ~5% 属 bf16 路由抖动,非误差)。
- **等价性**:同一 engine 先跑旧(mean + loss all_reduce + div)、再切新(sum)对比 grad_norm 比值,确认新方案复现当前梯度(已实测 EP=1=1.0013 / EP=2=1.0006)。
- 回归用例覆盖 grad-acc > 1、EP > 1、balancing/z-loss 开启三种情形。

## 9. Stacked PR 与 commit 粒度

**核心约束**:梯度正确性上,"翻转到 SUM"这一步**不可再拆**——FSDP reduce、`scale_and_reduce_grad`、三项 loss 的 ×world_size 是三因子相消,拆开任一因子都会产生**不报错的错误梯度**(§6)。因此栈的设计原则是:把所有**行为中性**的准备工作沉到栈底,让**语义翻转只发生一次**,集中在一个尽量小的 commit 里,使 `git bisect` 能精确定位、每个下层 commit 都能独立通过 CI。

栈自底向上如下,每个 commit 依赖前一个:

**C1 `[FSDP] Add reduce-sum gradient reduction helper`** — 行为中性
- 新增 `BaseModel.set_gradient_reduce_sum()`:遍历 `FSDPModule` 调用 `set_gradient_divide_factor(1.0)` + `set_force_sum_reduction_for_comms(True)`,入口做 torch ≥ 2.10 可用性检查。
- **不接入任何 `fully_shard`**,纯新增,训练行为不变。
- 附带 §7 的 bf16 单元测试(2 卡 toy FSDP,断言 reduce 后 == 各 rank 本地梯度之和)。该测试直接调用新 helper,C1 内即可通过。
- 独立可审:只需确认 FSDP API 用法与 bf16 清零坑的规避正确。

**C2 `[Loss] Split logging loss from backward loss`** — 行为中性
- 给 `BalancingLossContext` / `ZLossContext` 新增 detached **本地分量** log record(类比 CE 已有的 `local_base_loss`);z-loss 的本地分量不含 `× world_size`。
- 展示管线(`post_micro_batch_forward` 及 train_step `total_loss` 口径)改为对这些本地分量做 detached **SUM** 归约(不再 mean `div_(world_size)`)。
- **backward 仍用当前的 autograd all_reduce / `× world_size`**(loss 前向此时仍是全局值)。此状态下 `SUM(本地分量)` 恰等于今天的全局展示值,故**展示曲线与梯度均不变**,行为中性。
- 单独把 §5.4 的展示管线(本地分量记录 + mean→SUM)隔离出来,使 C3 的语义翻转 diff 最小、聚焦于梯度。

**C3 `[FSDP][Loss] Switch gradient reduction to SUM`** — 唯一的语义翻转,原子
- 在 `BaseModel/MoE/DenseModel.fully_shard` 三个根方法末尾调用 `set_gradient_reduce_sum()`。
- `MoE.scale_and_reduce_grad`:删专家 `div_(ep_size)`、删 replicated `div_(flat_mesh.size())`,注释同步改。
- CE:删末尾 autograd `all_reduce`(展示已在 C2 分离,反向自然变本地分量)。
- Balancing:`all_reduce_autograd(local_gating_sum)` → 直接用 `local_gating_sum`。
- Z-loss:删全局平均分支的 `× world_size`。
- 附带端到端 parity + 新旧等价性回归测试(§8),这些测试只有在本 commit 后才成立,故随本 commit 落地。
- 这是全栈唯一改变梯度语义的 commit,以上各因子必须同时改,不可再分。

**C4 `[Refactor] Remove unused world_size plumbing`** — 行为中性清理
- `ZLossContext.accumulate` 去掉不再使用的 `world_size` 参数并清理调用方;若 `all_reduce_autograd` 已无引用则一并删除。
- 按"接口签名变更单独提交"的惯例,把签名改动从 C3 的语义 diff 里分离,保持一份干净模型(CLAUDE.md 原则 8)。

**独立于本栈**:`noaux_router` 的 `e_score_correction_bias` 零初始化修复 —— 与 reduce-sum 无关的 latent bug,单开一个 PR,不进此栈。

**开发流程**:按 C1→C4 顺序建 commit;review/自查发现的问题用 `git commit --fixup=<对应 commit sha>` 生成 fixup,最后 `git rebase -i --autosquash`(非交互)折叠回对应 commit,保持栈的每个 commit 语义纯净。
