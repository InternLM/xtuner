# merge main 待办完成记录

## 1. rblock 确定性产品改动

已完成。

- 产品入口落在 `xtuner/v1/utils/misc.py::set_deterministic()`。
- 确定性模式会统一设置：
  - `CUBLAS_WORKSPACE_CONFIG=:16:8`
  - `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`
  - `torch._inductor.config.dynamic_scale_rblock = False`
- `xtuner/v1/train/trainer.py` 和 `xtuner/v1/rl/trainer/worker.py` 已复用公共 `set_deterministic()`，保证主训练和 RL worker 走同一逻辑。
- `tests/profiler/qwen35vl_determ.py` 的 standalone 确定性入口也复用该函数，避免脚本路径遗漏 rblock 设置。

## 2. rblock 对应测试

已完成。

- 新增 `tests/utils/test_misc.py`。
- 覆盖 `XTUNER_DETERMINISTIC=true` 时关闭 Inductor dynamic rblock。
- 覆盖非 deterministic 模式不修改用户显式设置的 `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK` 和
  `torch._inductor.config.dynamic_scale_rblock`。

## 3. `tests/profiler/numerics_test.py` 保留前清理

已完成。

- 新增并收敛为 text-only Qwen3.5 MoE 梯度确定性记录/比较脚本。
- 不再写死本地模型路径；通过 `--hf-path` 或 `QWEN35_MOE_PATH` 传入。
- 输入在 CPU 预生成，前向前再搬到 CUDA，注释与行为保持一致。
- `AccProber` 默认关闭，仅通过 `--prober` 显式启用。
- loss context 已按当前 main API 使用 `loss_cfg.build(data={"shifted_labels": ...}, sp_mesh=None)`。
- 比较语义固定为 per-rank gradient shard bitwise hash；任意 hash diff 返回 exit code 2。
