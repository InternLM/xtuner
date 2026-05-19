# 设计文档：MTP Loss Context 内聚化

## 背景与问题

当前 MTP loss 的实现分散在两处：

1. **`xtuner/v1/model/moe/moe.py`（`build_loss_ctx_batch`，L325-347）**：为每个 MTP depth 复用 `lm_loss_cfg`（`CELossConfig`）创建 loss context，不携带任何 MTP 语义；
2. **`xtuner/v1/model/moe/moe.py`（forward，L695-699）**：在 forward 时对 `loss_kwargs.shifted_labels` 做 in-place 的 `roll_packed_tensor`，偏移量由循环变量 `idx` 硬编码：

```python
for idx, (mtp_hidden, mtp_ctx) in enumerate(zip(mtp_outputs, mtp_loss_ctx_list)):
    mtp_ctx.loss_kwargs.shifted_labels = roll_packed_tensor(
        shifted_tensor, seq_ctx.cu_seq_lens_k, -idx - 1, dim=-1, fill_value=-100
    )
```

**存在的问题：**

1. **封装被破坏**：roll 逻辑泄露到 model forward，loss layer 对外暴露了实现细节；
2. **隐藏的权重计算 bug**：`build_batches` 在 roll 之前执行，看到的是未滚动的原始 labels，导致 boundary 位置未被 `ignore_idx` mask 过滤，`loss_weight` 计算有误；
3. **In-place 副作用**：直接修改 `loss_kwargs.shifted_labels`，语义不透明。

---

## 目标

1. 在 `xtuner/v1/loss/mtp_loss.py` 中实现 `MTPLossConfig` / `MTPLossKwargs` / `MTPLossContext`；
2. 将 `roll_packed_tensor` 逻辑移入 `MTPLossConfig.build()`，且在 SP split **之前**完成 roll；
3. 精简 `moe.py` 的 `build_loss_ctx_batch` 和 forward。

---

## 设计

### 核心思路：roll 提前到 `build()` 执行

将 roll 从 forward 时机提前到 `build()` 时机，带来两个好处：

- **SP 正确性**：roll 在 SP split 之前操作完整序列和完整 `cu_seq_lens`，split 之后每个 rank 拿到的是已经正确滚动的 shard；
- **`build_batches` 正确性**：`build_batches` 看到的 `shifted_labels` 已经是滚动后的状态，boundary 填 `-100` 会被 `loss_weight[shifted_labels == loss_cfg.ignore_idx] = 0.0` 正确屏蔽，无需覆写。

### `MTPLossKwargs(CELossKwargs)`

无新增字段，仅作类型标记，与 `CELossKwargs` 完全兼容。

### `MTPLossConfig(CELossConfig)`（内部实现，不对用户导出）

`mtp_depth: int` 在构造时传入（1-indexed），`build()` 签名保持标准的 `(data, sp_mesh)` 不变，以复用 `_build_loss_ctx` 的现有流程。该类不加入公开导出。

```python
class MTPLossConfig(CELossConfig):
    mtp_depth: int  # 第 1 个 MTP layer 对应 depth=1，shift=-1

    def build(self, data: dict, sp_mesh: DeviceMesh | None = None) -> "MTPLossContext | None":
        if "shifted_labels" not in data:
            return None

        shifted_labels = data["shifted_labels"]        # full sequence，SP split 之前
        cu_seq_lens = data["seq_ctx"].cu_seq_lens_k    # full cu_seq_lens

        # roll 在 SP split 之前，fill_value=-100 保证 boundary 被正确 mask
        rolled = roll_packed_tensor(
            shifted_labels, cu_seq_lens, shifts=-self.mtp_depth, dim=-1, fill_value=-100
        )

        loss_kwargs = MTPLossKwargs(shifted_labels=rolled).to(DEVICE)
        if sp_mesh is not None and sp_mesh.size() > 1:
            loss_kwargs = loss_kwargs.sp_split(sp_mesh)

        return MTPLossContext(self, loss_kwargs)
```

`data["seq_ctx"]` 在所有 `data_batch` 中均有（`base.py:L694` 已确认），无需改动数据格式。

### `MTPLossContext(LMHeadLossContext)`

无需覆写任何方法，完全继承 `LMHeadLossContext` 的 `forward()`、`build_batches()`、`eager_mode()`、`chunk_mode()` 等逻辑。

### `moe.py build_loss_ctx_batch`

模型内部按 depth 构造 `MTPLossConfig`，复用 `_build_loss_ctx` 标准流程：

```python
for mtp_idx in range(self.config.mtp_config.num_layers):
    mtp_loss_cfg = MTPLossConfig(
        **self.config.lm_loss_cfg.model_dump(),
        mtp_depth=mtp_idx + 1,
    )
    mtp_loss_ctx_list = self._build_loss_ctx(mtp_loss_cfg, _data_batch, sp_mesh)
    if mtp_loss_ctx_list is not None:
        mtp_loss_ctx_list = MTPLossContext.build_batches(
            mtp_loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=sp_mesh
        )
        for i, mtp_loss_ctx in enumerate(mtp_loss_ctx_list):
            if "mtp" not in res[i]:
                res[i]["mtp"] = []
            res[i]["mtp"].append(mtp_loss_ctx)
```

### `moe.py forward`

删除 L696-699 的 roll mutation 块，forward 中只保留 `mtp_ctx` 的直接调用。

---

## 变更清单

| 文件 | 操作 |
|---|---|
| `xtuner/v1/loss/mtp_loss.py` | 实现 `MTPLossKwargs`、`MTPLossConfig`（内部）、`MTPLossContext` |
| `xtuner/v1/model/moe/moe.py` | `build_loss_ctx_batch` 改用 `MTPLossConfig`；forward 删除 roll mutation |
| `xtuner/v1/loss/__init__.py` | 导出 `MTPLossContext`（不导出 `MTPLossConfig`）|
