# merge main 前待办

## 1. rblock 确定性产品改动

当前分支还没有真正的产品代码改动；`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK` 只出现在调查文档和实验脚本中。

TODO:

- 在确定性模式下补产品代码入口，建议放到 `xtuner/v1/utils/misc.py::set_deterministic()`，统一设置：
  - `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`
  - 必要时同时设置 `torch._inductor.config.dynamic_scale_rblock = False`
- 确认设置时机早于 `torch.compile` 触发；否则 Inductor 已经读取配置后再设置无效。
- 覆盖主训练入口后，再确认 RL worker 是否需要同步走同一个 `set_deterministic()`，避免训练路径行为不一致。
- 为该行为补一个轻量测试：在 `XTUNER_DETERMINISTIC=true` 时，验证 deterministic 初始化后 rblock 配置被关闭；非 deterministic 模式不改变用户显式配置。

## 2. `tests/profiler/numerics_test.py` 保留前清理

`numerics_test.py` 有保留价值，但目前仍像调查脚本，进 PR 前需要收敛成稳定的端到端确定性验证工具。

TODO:

- merge 最新 `upstream/main` 后更新 API 用法，尤其是 loss context 构造；当前脚本里的 `loss_cfg.build(shifted_labels=..., sp_mesh=None)` 可能需要改成 main 上的公开调用方式。
- 明确脚本目标：是 text-only Qwen3.5 MoE 梯度确定性，还是 Qwen3.5-VL 端到端确定性；删除 `xTODO` 和过时注释。
- 清理与注释不一致的输入预生成逻辑：当前注释说先在 CPU 预生成，但实际传入 `device=torch.device("cuda")`。
- `setup_prober_list(...)` 建议加 CLI 开关或默认关闭，避免数值复现脚本总是写 AccProber dump。
- 删除或参数化本地路径/环境假设；`run_test.sh` 里的硬编码模型路径不要随 PR 一起进。
- 收敛比较指标：明确 bitwise、grad sum 还是相对误差阈值，输出稳定、退出码语义清晰。
- 跑 `ruff-format` / `ruff` / `git diff --check`，并修掉 trailing whitespace。

## 3. 当前 rblock 对应测试状态

当前没有产品级 rblock 测试。

已有的是实验/复现脚本：

- `tests/profiler/mha_kpath_triton_probe.py` 支持 `--no-dynamic-scale-rblock`，会设置 `torch._inductor.config.dynamic_scale_rblock = False`。
- `tests/profiler/mha_kpath_triton_probe.sh` 对比 dynamic rblock 开/关后的复现结果。
- `tests/profiler/qwen35vl_determ.sh` 默认导出 `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`。

这些能证明问题和 workaround，但不等价于产品代码测试。产品改动落地后仍需要补一个直接覆盖 deterministic 初始化行为的测试。
