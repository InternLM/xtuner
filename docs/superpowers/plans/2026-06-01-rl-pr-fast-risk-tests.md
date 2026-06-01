# RL PR-fast 风险逻辑单测开发计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 在 `tests/rl/fast/pr_fast` 中补齐目前容易在真实训练启动后才暴露的问题：训练数据构造、SingleTurnAgentLoop batch judge / pause 控制流、RolloutController fake routed 最小失败分支、CPUResourceManager 资源校验。

**整体原则:** 全部测试都属于 PR-fast，不依赖真实模型、真实数据路径、真实 rollout backend、GPU 或长时间 Ray 服务。测试优先使用真实 `RolloutState`，对 rollout controller / Ray 资源部分使用轻量 fake 对象和 patch。

**当前暂不覆盖:** `ComposedJudger` 先不加测试，因为该类正在其他开发者那里重构；RolloutController 的 timeout、deterministic seed、concurrency、metadata 暂不加；SingleTurnAgentLoop 的单样本 tokens 补齐、非 batch judge 逐条打分暂不加。

---

## 文件落点

- Create: `tests/rl/fast/pr_fast/test_prepare_train_data.py`
  - 覆盖 `RLTrainer._prepare_train_data()` 的训练数据 contract。
- Create: `tests/rl/fast/pr_fast/test_single_turn_agent_loop.py`
  - 覆盖 `SingleTurnAgentLoop.generate_group()`、`run_judger()`、`pause()` 的 batch judge 和 pause 控制流。
- Modify: `tests/rl/fast/pr_fast/test_rollout_logic.py`
  - 在现有 rollout worker / utils 逻辑测试中追加 `RolloutController.generate()` 的 fake routed 最小覆盖。
- Modify: `tests/rl/fast/pr_fast/test_cpu_pg.py`
  - 在现有 CPU/Ray 工具测试中追加 `CPUResourceManager.register()` / `validate_or_raise()` 覆盖。
- Modify: `tests/rl/fast/pr_fast/README.md`
  - 代码完成并跑完后更新测试范围和最新耗时。

---

## Task 1: `_prepare_train_data` 训练数据 contract

**测试文件:** `tests/rl/fast/pr_fast/test_prepare_train_data.py`

**辅助对象:**
- 用 `RLTrainer.__new__(RLTrainer)` 构造 trainer，不走完整 trainer 初始化。
- 设置 `trainer._advantage_estimator` 为 fake estimator，`compute(rewards_tensor, group)` 返回固定 tensor。
- 设置 `trainer.tokenizer` 为 fake tokenizer，仅用于兜底分支；主测试路径不依赖真实 tokenizer。
- 设置 `trainer.logger` 为 `MagicMock()`。
- 输入使用真实 `RolloutState`。
- patch `xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC=True`，避免 `random.shuffle(data_batches)` 影响断言。

### 需要测试的逻辑和检查点

1. **文本主路径：prompt、response、label、logprob、advantage 的布局**
   - 输入：
     - `prompt_ids=[10, 11, 12]`
     - `response_ids=[20, 21, 22]`
     - `logprobs=[0.1, 0.2, 0.3]`
     - `response_mask=[1, 0, 1]`
     - `reward={"score": 1.0}`
     - fake advantage 返回 `1.5`
   - 检查点：
     - `seq_ctx.input_ids == [[10, 11, 12, 20, 21]]`
     - `shifted_labels == [[-100, -100, 20, -100, 22]]`
     - `rollout_logprobs == [[0.0, 0.0, 0.1, 0.2, 0.3]]`
     - `advantage == [1.5, 1.5, 1.5, 1.5, 0.0, 1.5]`
     - `len(advantage) == shifted_labels.numel() + 1`
     - `seq_ctx.rollout_routed_experts` 保留 rollout state 上的 `routed_experts`
     - `info_dict["training_samples"] == 1`
     - `info_dict["training_tokens"] == 5`
     - reward、prompt_len、response_len 的 mean/min/max 与输入一致。

2. **多 sample group：每个 response 使用自己的 reward 和 advantage**
   - 输入同一个 prompt 下两个 `RolloutState`，fake advantage 返回 `[1.5, -2.0]`。
   - 检查点：
     - 返回两个 `data_batches`。
     - 第一个 batch 的 `advantage` 使用 `1.5`，第二个 batch 使用 `-2.0`。
     - `info_dict["batch_size"] == 2`。
     - `info_dict["rewards/min"]`、`rewards/max`、`rewards/mean` 聚合两个样本。
     - `advantages/mean|min|max` 使用 `actual_advantages[:-1]` 统计，也就是训练统计长度，不是完整 `advantage` 长度。

3. **VLM / multimodal 分支**
   - 输入：
     - `prompt_ids=[1]`
     - `extra_fields["train_prompt_ids"]=[100, 101]`
     - `response_ids=[102, 103]`
     - `position_ids` 是 3D numpy array，prompt 长度为 2。
     - `mm_info` 包含 `pixel_values` 和 `image_grid_thw`。
   - 检查点：
     - 使用 `train_prompt_ids`，不是 `prompt_ids`，所以 `seq_ctx.input_ids == [[100, 101, 102]]`。
     - 3D `position_ids` 被扩展到和 `input_ids` 最后一维等长。
     - `seq_ctx.image_grid_thw.dtype == torch.long`。
     - `seq_ctx.pixel_values` 被保留。

4. **无效 group 被跳过**
   - 输入一个 `Status.COMPLETED` group 和一个 `Status.FAILED` 或 `Status.FILTERED` group。
   - 检查点：
     - 只返回 valid group 对应的 `data_batches`。
     - invalid group 不计入 `training_samples`。
     - `trainer.logger.error` 被调用。

5. **失败 contract**
   - 分别测试：
     - 缺少 `reward["score"]` 时触发 `AssertionError`。
     - `len(logprobs) != len(response_ids)` 时触发 `AssertionError`。
     - `len(response_mask) != len(response_ids)` 时触发 `AssertionError`。
     - `len(input_ids) > pack_max_length` 时触发 `AssertionError`。
   - 检查点：
     - 每个异常测试只覆盖一个失败原因，便于定位。

**执行顺序:**
- [ ] 新增 `test_prepare_train_data.py`，先写文本主路径。
- [ ] 跑单个测试函数：`python -m pytest tests/rl/fast/pr_fast/test_prepare_train_data.py::TestPrepareTrainData::test_text_path_builds_shifted_training_tensors -q`
- [ ] 补多 sample、VLM、invalid、失败 contract。
- [ ] 跑整个文件：`python -m pytest tests/rl/fast/pr_fast/test_prepare_train_data.py -q`

---

## Task 2: `SingleTurnAgentLoop` batch judge / pause 控制流

**测试文件:** `tests/rl/fast/pr_fast/test_single_turn_agent_loop.py`

**辅助对象:**
- 用 `SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)` 构造实例，不触发真实初始化。
- 手动设置：
  - `rollout_ctl.generate.remote`
  - `rollout_ctl.pause_generation.remote`
  - `sample_params`
  - `judger`
  - `enable_batch_judge=True`
  - `_pause_event = asyncio.Event()`
  - `logger = MagicMock()`
- 输入使用真实 `RolloutState`。

### 需要测试的逻辑和检查点

1. **只有一组样本全部 COMPLETED 时，batch judger 才会被触发**
   - 输入两个 state。
   - fake rollout 对两个 state 都返回 `Status.COMPLETED`。
   - fake judger 一次性接收 list，并给每个 state 写入 reward。
   - 检查点：
     - `judger.judge` 只调用一次。
     - `judger.judge` 收到的是 `list[RolloutState]`，长度为 2。
     - 返回列表长度为 2。
     - 返回顺序和输入顺序一致，例如 uid 仍是 `[1, 2]`。
     - 两个返回 state 都带有 reward。
     - 每个 state 的 `sample_params` 都被设置为 agent loop 的 `sample_params`。

2. **batch judge 模式下，如果组里存在 ABORTED 样本就不打分**
   - 输入两个 state。
   - fake rollout 返回一个 `Status.COMPLETED`，一个 `Status.ABORTED`。
   - 检查点：
     - `judger.judge` 不被调用。
     - 返回列表顺序仍和输入一致。
     - COMPLETED 样本保持 COMPLETED，ABORTED 样本保持 ABORTED。
     - 两个样本都没有新增 reward。

3. **batch judge 模式下，如果组里存在其他非 COMPLETED 样本也不打分**
   - 输入两个 state。
   - fake rollout 返回一个 `Status.COMPLETED`，一个 `Status.FAILED` 或 `Status.FILTERED`。
   - 检查点：
     - `judger.judge` 不被调用。
     - 返回列表顺序和输入一致。
     - 非 COMPLETED 状态保持原样。
   - 说明：
     - 当前源码只显式判断 `Status.ABORTED` 时跳过 batch judge；如果这个测试失败，说明需要先讨论是否把 contract 扩展为“全组 COMPLETED 才打分”。这个测试对应用户期望的行为，不是简单复制当前实现。

4. **pause 期间 slow judger 被取消并标记 ABORTED**
   - fake judger `judge()` 长时间 sleep。
   - patch `xtuner.v1.rl.agent_loop.single_turn_agent_loop.DEFAULT_JUDGER_CANCEL_TIMEOUT_S=0.01`。
   - 启动 `run_judger(group)` 后设置 `_pause_event`。
   - 检查点：
     - 返回 group 中每个 state 都 `status == Status.ABORTED`。
     - 每个 state 的 `finish_reason == "abort"`。
     - 每个 state 的 `reward is None`。

**明确不测:**
- 不测 `generate_sample` 补齐 `tokens`。
- 不测非 batch judge 模式下的单条逐个打分。

**执行顺序:**
- [ ] 新增 `test_single_turn_agent_loop.py`。
- [ ] 先实现全部 COMPLETED 才触发 batch judge，并检查返回保序。
- [ ] 再实现存在 ABORTED 时跳过 judge。
- [ ] 再实现存在其他非 COMPLETED 时跳过 judge；如果当前代码失败，先停下来反馈。
- [ ] 最后实现 pause/cancel 测试，确保 timeout 小于 0.05s。
- [ ] 跑整个文件：`python -m pytest tests/rl/fast/pr_fast/test_single_turn_agent_loop.py -q`

---

## Task 3: `RolloutController` fake routed 最小覆盖

**测试文件:** 追加到 `tests/rl/fast/pr_fast/test_rollout_logic.py`

**辅助对象:**
- 用 `RolloutController.__new__(RolloutController)` 构造实例，不启动真实 worker。
- 手动设置：
  - `config = SimpleNamespace(rollout_timeout=1.0, random_seed=0)`
  - `timeout_multiplier = 1.0`
  - `router`
  - `_tool_call_parser = None`
  - `_reasoning_parser = None`
  - `logger = MagicMock()`
- fake worker 提供 `generate.remote(rollout_state=...)`，返回 awaitable。

### 需要测试的逻辑和检查点

1. **成功路由到 active worker**
   - fake router 根据 `session_uid` 返回 fake worker。
   - fake worker 返回 `Status.COMPLETED` 的 state。
   - 检查点：
     - `router.get_worker` 使用 `rollout_state.session_uid`。
     - worker 收到同一个 `rollout_state`。
     - `generate()` 返回 worker 的结果。
     - state 没有被错误标记为 `Status.FAILED`。

2. **没有 active worker 时失败返回**
   - fake router 返回 `None`。
   - 检查点：
     - 返回原 state。
     - `status == Status.FAILED`。
     - `error_msg == "No active rollout worker available."`
     - worker 不会被调用。

**明确不测:**
- 不测 worker timeout。
- 不测 deterministic seed。
- 不测 concurrency 和 metadata。
- 不测 parser 应用逻辑。

**执行顺序:**
- [ ] 在 `test_rollout_logic.py` 中新增 `TestRolloutController` 测试类。
- [ ] 先写 no active worker 分支。
- [ ] 再写 success routed 分支。
- [ ] 跑新增类：`python -m pytest tests/rl/fast/pr_fast/test_rollout_logic.py::TestRolloutController -q`

---

## Task 4: `CPUResourceManager.register/validate_or_raise`

**测试文件:** 追加到 `tests/rl/fast/pr_fast/test_cpu_pg.py`

**辅助对象:**
- 不启动真实 Ray。
- 构造 `CPUResourceManager(accelerator_placement_groups=None)`。
- patch manager 实例的 `_build_resource_summary()`，返回由当前 `manager.pools` 动态计算的 summary。
- 使用真实 `CPUResourcesConfig`。

### 需要测试的逻辑和检查点

1. **register 成功并支持重名注册**
   - 外部 CPU 容量足够：`external_capacity_cpus=8`，`external_capacity_memory=16 GiB`，`max_node_external_cpus=4`。
   - 连续 `register("judger", config)` 两次。
   - 检查点：
     - `manager.pools` 包含 `"judger"` 和 `"judger#2"`。
     - 两个 pool 都是传入的 `CPUResourcesConfig`。
     - 不抛异常。

2. **CPU 总量不足时抛错并回滚本次注册**
   - 外部 CPU 容量为 1。
   - 注册 `num_workers=2, num_cpus_per_worker=1`。
   - 检查点：
     - `register()` 抛 `RuntimeError`。
     - 错误信息包含 `available_outside_accelerator_pg`。
     - `manager.pools` 不保留失败注册项。

3. **memory 总量不足时抛错并回滚本次注册**
   - 外部 memory 容量为 1 GiB。
   - 注册 `cpu_memory_per_worker=2 GiB`。
   - 检查点：
     - `register()` 抛 `RuntimeError`。
     - 错误信息包含 `memory requested`。
     - `manager.pools` 不保留失败注册项。

4. **单 worker CPU 超过单节点 external CPU 时抛错**
   - 总 external CPU 足够，但 `max_node_external_cpus=1`。
   - 注册 `num_cpus_per_worker=2`。
   - 检查点：
     - `register()` 抛 `RuntimeError`。
     - 错误信息包含 `largest node`。
     - 失败注册项被回滚。

5. **validate_or_raise 会累加已有 pools**
   - 先注册一个成功 pool。
   - 再把 summary 容量调低，直接调用 `validate_or_raise()`。
   - 检查点：
     - `validate_or_raise()` 对已有 pools 抛 `RuntimeError`。
     - 不修改已有 `manager.pools`。

**执行顺序:**
- [ ] 在 `test_cpu_pg.py` 中新增 `TestCPUResourceManager` 测试类。
- [ ] 先写动态 summary helper。
- [ ] 写成功注册和重名注册。
- [ ] 写 CPU、memory、max-node 三个失败回滚测试。
- [ ] 写已有 pools 的 `validate_or_raise()` 测试。
- [ ] 跑新增类：`python -m pytest tests/rl/fast/pr_fast/test_cpu_pg.py::TestCPUResourceManager -q`

---

## 最终验证

- [ ] 跑新增或修改过的测试文件：

```bash
python -m pytest \
  tests/rl/fast/pr_fast/test_prepare_train_data.py \
  tests/rl/fast/pr_fast/test_single_turn_agent_loop.py \
  tests/rl/fast/pr_fast/test_rollout_logic.py \
  tests/rl/fast/pr_fast/test_cpu_pg.py \
  -q
```

- [ ] 跑整个 PR-fast：

```bash
python -m pytest tests/rl/fast/pr_fast -q
```

- [ ] 更新 `tests/rl/fast/pr_fast/README.md`：
  - 新增本批覆盖范围说明。
  - 更新 latest measured result 和 wall time。

---

## 自检

- 文件落点已按要求调整：RolloutController 测试合并到 `test_rollout_logic.py`，CPUResourceManager 测试合并到 `test_cpu_pg.py`。
- SingleTurnAgentLoop 不再测试 tokens 补齐和非 batch judge 单条打分。
- RolloutController 不再测试 timeout、deterministic seed、concurrency、metadata。
- ComposedJudger 暂不添加测试。
- `_prepare_train_data` 的 `advantage` contract 按当前共识处理：`data_dict["advantage"]` 比 `shifted_labels` 多 1 个元素；metric 统计使用 `actual_advantages[:-1]`。
