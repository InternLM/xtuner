# 2026-10-09 单测失败归因

## 总结

本地 `a04ut/unit_test1009.log` 为当前 `d0877d638` 分支：**1056 collected、138 failed、899 passed、19 skipped**。PR [#2108 的 GitHub workflow](https://github.com/InternLM/xtuner/actions/runs/37946542582/job/113875228425#step:4:8212) 运行的是祖先提交 `9c1f93c6`：**984 collected、22 failed、928 passed、34 skipped**。本地多了 F6/VL 等后续提交和 72 个收集到的测试项，且 Python 依赖、缓存权限不同；138 与 22 不能直接视作同一批代码的回归差值。两个失败集合有 **16 个相同 case**，本地独有 122 个，CI 独有 6 个。

本地最显著的干扰是 **88 个 case 的 `/tmp/.triton` 写权限错误**。在清理这个干扰之前，它们只能归类为“本次运行首先被缓存权限阻断”，不能推断修好缓存后业务断言会全部通过。明确与当前 stack 新增测试/代码交互有关的有 **13 个跨测试编译状态污染**、**7 个 GLM-5.3 测试配置错误**；CI 还暴露了 GLM-5.3 新测试与项目固定 Transformers 版本不匹配的问题。

## 运行条件和证据

| 项目 | 本地 `zdev/run_test.sh` | GitHub workflow |
| --- | --- | --- |
| 提交 | `d0877d638`，含 #2108 之后的 F6/VL 提交 | PR #2108 `9c1f93c6` |
| 执行 | `source zdev/env.sh`，激活 `pt29_glm2`，`pytest tests/` | 8 GPU 容器，`source ci/scripts/CI_ENV.sh`（其中执行 `pip install -e .[all]`），`pytest tests` |
| Python / pytest | 3.12.13 / 9.0.3 | 3.12.3 / 8.1.1 |
| Transformers | 本地实测 5.17.0 | CI 缺少 `transformers.models.glm5_next`；`pyproject.toml` 固定 5.14.1 |
| 缓存 | `zdev/env.sh` 指向共享 `/tmp/.triton`；该目录属其他 UID，权限 `0775`，当前用户无写权限 | 独立容器内未出现该权限错误 |

本次仅做归因分析，没有改动测试脚本、依赖或产品代码。最小复现均使用 GPU 文件锁和独立 Triton 缓存：单独运行 `TestGlm53DenseDecoderLayer.test_forward_finite_and_shape_preserving` 通过；先运行 `TestBuildModel.test_build_moe` 再运行它，稳定复现 Dynamo `Unsupported`。单独编译 `tl_indexer_fwd_impl(heads=2, index_dim=4)` 复现 `For fp64 MMA, m_dim must be 8, got 16`；`index_dim=16` 可编译。

## 本地 138 个失败分类

| 编号 | 归属与直接原因 | 数量 | 关键证据 |
| --- | --- | ---: | --- |
| A | 本地环境：Triton 缓存无写权限 | 88 | 多种模型、算子和 Ray worker 最内层均为 `PermissionError: /tmp/.triton/...`；`os.access('/tmp/.triton', os.W_OK)` 为 `False`。含 1 个子进程已报权限错误、外层最终报 600 秒超时的 case。 |
| B | 本地依赖差异：Transformers 视频处理器私有 API 变化 | 13 | `qwen3_vl_tokenize_fn.py:930` 调 `_preprocess` 少 6 个必需参数；本地 5.17.0 的真实函数签名已核对。CI 同组测试通过。 |
| C | 本地 pytest 版本/插件：异步测试未执行 | 7 | `test_sandbox_pool.py` 用 `@pytest.mark.asyncio`，本地无 `pytest-asyncio`；pytest 9 报 `async def functions are not natively supported`。CI 的 pytest 8 将这 7 个标记为 **skipped**，没有验证功能。 |
| D | 资源配置：测试申请 32 GPU，Ray 只有 8 GPU | 1 | `test_update_weight_colocate` 的 `Not enough available GPUS in Ray cluster, 8.0 less than 32`；CI 同样失败。 |
| E | stack 测试顺序问题：全局 `torch.compile(fullgraph=True)` 污染 GLM-5.3 测试 | 13 | 7 个 compose、6 个 decoder 测试；`BaseModel._compile_overwrite` 通过 `setattr(cls, method_name, torch.compile(...))` 改写**类方法**。先构建 Qwen3 MoE 会全局编译基类 `DenseDecoderLayer.forward` / `MoEDecoderLayer.forward`；后续 GLM 测试的 `compile_cfg=False` 不会恢复类方法。最小复现见上。当前日志在 KDA 的 `cuda_utils.get_device_properties` 或被 `torch.compiler.disable` 包装的 `_run_recurrent_kda` 上报 `Unsupported`。 |
| F | stack 新增测试配置错误：要求训练已明确禁止的 indexer | 6 | `test_glm53_text_moe.py` 的 `_tiny_cfg()` 设 `freeze_dsa_indexer=False`；`NoPEDSAMLAConfig.build()` 明确拒绝未实现的可微 indexer，6 个测试均在建模阶段报相同 `ValueError`。 |
| G | stack 新增测试配置错误：小模型未选 torch sparse MLA 后端 | 1 | `test_backend_assignment_is_validated` 构造 4 个注意力头，但默认 `flash_mla_cudnn` 后端要求头数为 64 的倍数；在预期的赋值校验之前即报 Pydantic `ValidationError`。 |
| H | TileLang 与小尺寸测试配置不兼容 | 4 | GLM-5.2 两个测试文件都设 `index_head_dim=4`。TileLang 0.1.11 的 MMA 代码把 `k_dim=4` 走到 FP64 专用尺寸断言；最小编译复现，`index_dim=16` 可编译。CI 相同路径有 **5** 个失败；本地第 5 个先被 A 类缓存错误遮蔽。生产配置 `index_head_dim=128`，不能据此判定生产尺寸也失败。 |
| I | 既有测试的 CUDA allocator 断言不成立 | 1 | `test_resume_and_load_checkpoint_cfg` 的临时对象弱引用已释放，但 `memory_reserved()` 清理前后相等（本地 `90177536`，CI `150994944`）。`Trainer` 已执行 `gc.collect()` 和 `empty_cache()`；相等本身不能证明临时对象泄漏。需重审以 reserved bytes 严格下降作为成功条件是否有效。 |
| J | DeepEP / CUDA 运行失败，根因未确定 | 1 | `test_deepep_expert_tp_expert_only_grad_norm_matches_single_model_baseline` 各 rank 报 `CUDA error: unspecified launch failure`，最终进程 `-6`。外层 `Scalars are not equal` 比较的是**进程退出码**，不是梯度值；日志不足以判定是内核、驱动还是设备状态。 |
| K | 分布式 DCP 测试超时，根因未确定 | 1 | `test_dcp_round_trip_preserves_model_and_optimizer` 在 600 秒超时；rank 0 的采样栈停在测试 `finally` 的 `dist.barrier()`，无法从日志知道哪个 rank 先停住。 |
| L | Qwen3.5 视觉塔数值不一致，根因未确定 | 1 | `test_vision_tower_bitwise_parity` 的最大输出差 `1.0625`；同一测试在 #2108 CI 通过，且 #2108 到本地 HEAD 没修改该测试或 Qwen3.5 代码。Transformers 版本/权重状态需分别核对，不能仅凭这条断言归罪 stack。 |
| M | FlashMLA/cuDNN 反向数值不一致，根因未确定 | 1 | 前向和 softmax LSE 已通过，`q_actual.grad` 与 torch 参考有 93.1% 元素超出容差；#2108 CI 中该测试因运行时条件 **skipped**，所以没有可比的 CI 成绩。 |
| **合计** |  | **138** | 附录逐项列出全部失败 case。 |

E 是 stack 新增 GLM-5.3 测试与既有类方法全局编译机制之间的顺序依赖。F/G 是 stack 祖先提交 `9c1f93c6`（#2143）新增的配置约束，与后续 F6 小模型测试参数没有同步。H 的相同错误也出现在 #2108 CI，但现有证据仅定位到 TileLang 与 `index_head_dim=4` 测试配置的组合，不能推断生产尺寸的行为。

### 与 #2108 CI 的 22 个失败逐组对照

| CI 数量 | case | CI 的实际错误与当前状态 |
| ---: | --- | --- |
| 5 | GLM-5.2 `test_glm52_moe.py` 2 个、`test_glm52_mtp_checkpoint_repro.py` 3 个 | 均为 H 类 TileLang `fp64 MMA` 断言；当前本地 4 个相同断言，另 1 个先报缓存权限错误。 |
| 6 | `test_glm53_decoder_layer.py` 前 6 个 | CI 在 FLA `prepare_chunk_indices` 的缓存赋值遇 Dynamo `HigherOrderOperator: Mutating a variable ... (SideEffects)`。当前分支换了 KDA 内核入口，日志中的错误变成 E 类全局编译后无法越过 `@torch.compiler.disable`；两次都卡在“GLM KDA 被不合适的编译边界追踪”。 |
| 1 | `test_glm53_decoder_layer.py::...test_dense_layer_forward_compiles_with_dynamic_cu_seqlens` | CI 的 FLA `.tolist()` 导致图断裂，随后 `ConstraintViolationError`：声明为 dynamic 的 `cu_seqlens.size(0)` 被特化为 2。当前本地先报 A 类缓存错误，**不能证明此编译问题已修复**。 |
| 7 | GLM-5.3 与 HF 对齐：`test_glm53_dsa.py` 2 个、`test_glm53_kda.py` 2 个、`test_glm53_mhc.py` 2 个、`test_glm53_nope_dsa_mla.py` 1 个 | `ModuleNotFoundError: transformers.models.glm5_next`。#2108 增加了依赖该 HF 模块的测试，但项目仍固定 `transformers==5.14.1`。本地 5.17.0 已有该模块；其中 KDA 两个测试又被 A 类缓存错误遮蔽。 |
| 1 | `test_glm53_dsa.py::...test_routed_experts_use_the_same_clamp_as_shared_experts` | PR #2108 测试提前导入 F6 才提供的 `Glm53TextMoEConfig`；在 PR 提交上是 stack 依赖缺口，当前 HEAD 已导出该类且该测试通过。 |
| 1 | RL colocate 更新权重 | D 类 32 GPU 与 8 GPU 不匹配，本地/CI 一致。 |
| 1 | trainer resume/load checkpoint | I 类 `memory_reserved` 严格下降断言，两边一致。 |
| **22** |  |  |

## 后续复验确认

截至 2026-10-10，A 类缓存权限和 C 类异步插件已由本地环境修正。B 类改用 Transformers 的公开视频预处理 API；D 类去掉从其他测试继承的 `WORLD_SIZE`；E 类构造测试不再全局编译 decoder 类方法；F/G/H 类小模型测试改用实际支持的配置；I 类直接检查临时 CUDA tensor 的释放。上述原因和回归命令详见 `02_ut_fix.md`。

初始日志单独无法归因的 L 类，已在真实 Qwen3.5 checkpoint 中定位为 XTuner 视觉 RoPE 以 bf16 而 HF 以 fp32 计算频率，修正后原 4 rank bitwise parity 通过。M 类已由 rebase 带入的 `eb6ad8f2` 修复：FlashMLA 返回的自然对数 LSE 不再错误转为 log2 后传给 cuDNN backward。K 类 DCP 超时在单项和旧失败项顺序中均通过，未找到稳定根因。J 类在长前序后的真实 DeepEP 顺序中复现为 layout 与 dispatch 间的通信事件问题；仅将小型 layout 元数据改回异步事件路径后，前置 8 卡保存加载加 DeepEP 四项的干净代码连续 **3/3 轮、每轮 5 passed**。详细对照和限制见 `02_ut_fix.md`。

#2108 的动态 `cu_seqlens` 编译 case 在隔离 F5 上仍失败，但顶层 F6 已有 KDA custom-op 边界且该项通过，因此测试归属要从 F4 移到 F6。F5 测试提前导入 F6 的 `Glm53TextMoEConfig` 也已在 F5 改为测试本层公开的激活配置，F6 另加默认模型配置接线检查。最新 F5 隔离回归的 22 项中 **21 passed、1 failed**；唯一剩余失败发生于 Ray dashboard agent 的 GPU 探测超过其 15 秒端口文件等待窗口，不是 colocate 权重更新断言。

## 初始建议的验证顺序

1. 给本地 pytest 进程及其 Ray 子进程设置用户独享、可写的 `TRITON_CACHE_DIR`（并分开 Inductor/pytest 缓存），先重跑 A 类代表 case。这是排除 88 个阻断错误的前置条件；不要修改其他用户的 `/tmp/.triton`。
2. 固定并记录 Transformers/pytest 版本：GLM-5.3 HF 对齐需要提供 `glm5_next` 的版本；同时处理 Qwen3VL `_preprocess` 的接口差异，并为 `@pytest.mark.asyncio` 安装相应插件或改用已安装插件的标记。CI 目前把 7 个异步 case 跳过。
3. 修正 F/G 的新测试配置，隔离或恢复 E 的全局编译状态。随后重跑 GLM-5.3 compose/decoder/KDA 子集，才能判断被缓存错误遮蔽的业务断言。
4. H 类用与 TileLang 支持范围一致的 indexer 测试尺寸或专门覆盖小尺寸报错；对 I 的 allocator 断言使用能区分临时对象泄漏与活跃张量/缓存行为的条件。
5. 初始日志中的 J/K/L/M 应独立复现。K 需采集各 rank 卡住的位置；M 需固定 FlashMLA/cuDNN 版本并对 dQ 做最小复现。初始日志不足以把它们认定为此次 stack 的代码回归；复验结论已在上节更新。

## 附录：138 个本地失败 case 的归属

以下按**本次运行中首先可见的失败原因**归类；A 类修复权限后可能暴露第二个问题。每个条目均来自本地日志的 `short test summary info`。

### A（88）：本地 Triton 缓存写权限

```text
tests/engine/test_dense_train_engine.py::TestDenseEngine::test_dense_engine_train[cuda-1-1]
tests/engine/test_dense_train_engine.py::TestDenseEngine::test_dense_engine_train[cuda-1-2]
tests/engine/test_dense_train_engine.py::TestDenseEngine::test_dense_engine_train_swap_optimizer[cuda-1-1]
tests/engine/test_dense_train_engine.py::TestDenseEngine::test_dense_engine_train_swap_optimizer[cuda-1-2]
tests/engine/test_glm52_moe_train_engine.py::TestGlm52OptimizedEngine::test_sp2_ep4_micro2_compile_offload_train_step
tests/engine/test_glm52_moe_train_engine.py::TestGlm52PretrainedEngine::test_ep8_loss_curve_matches_reference
tests/engine/test_glm52_moe_train_engine.py::TestGlm52PretrainedEngine::test_tilewise_fp8_ep4_train_step
tests/engine/test_glm52_moe_train_engine.py::TestGlm52PretrainedEngine::test_tilewise_fp8_loss_curve_matches_bf16
tests/engine/test_moe_train_engine.py::TestMoEEngine::test_moe_engine_train
tests/engine/test_moe_train_engine.py::TestMoEEngine::test_moe_engine_train_and_save_hf
tests/engine/test_moe_train_engine.py::TestMoEEngine::test_moe_engine_train_freeze_routers[cuda-1-1]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_float8_dcp_resume[cuda-1]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_fp8_ep2_etp2_fsdp2_train[cuda-2-2]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_save_and_load[cuda-1-8]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_tensor_wise_fp8[cuda-1-8]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_tile_wise_fp8[cuda-1-8-0-01-0-01]
tests/engine/test_moe_train_engine_float8.py::TestMoEEngineFloat8::test_tile_wise_fp8[cuda-8-8-0-01-0-15]
tests/model/test_ep_load_metrics.py::TestEPLoadMetrics::test_forward_outside_train_step_is_not_counted
tests/model/test_ep_load_metrics.py::TestEPLoadMetrics::test_ratios_match_pinned_routing[((0, 1), (0, 2))-1-{'load': (1-5, 0-5), 'peak': (2-0, 1-0), 'straggler': 1-5}]
tests/model/test_ep_load_metrics.py::TestEPLoadMetrics::test_ratios_match_pinned_routing[((0, 1), (0, 2))-2-{'load': (1-5, 0-5), 'peak': (2-0, 1-0), 'straggler': 1-5}]
tests/model/test_ep_load_metrics.py::TestEPLoadMetrics::test_ratios_match_pinned_routing[((0, 1), (2, 3))-1-{'load': (1-0, 1-0), 'peak': (2-0, 2-0), 'straggler': 2-0}]
tests/model/test_fsdp_checkpoint.py::TestFSDPCheckpoint::test_mixed_dense_checkpoint_compile_allows_pytree_boundary
tests/model/test_glm52_mtp_checkpoint_repro.py::TestGlm52CompiledMTPCheckpoint::test_shared_mtp_depths_train_with_compile_and_topk_offload
tests/model/test_glm53_compose.py::TestGlm53ComposeSequenceParallel::test_image_splice_under_sp_matches_non_sp
tests/model/test_glm53_compose.py::TestGlm53ComposeFSDPBackward::test_text_and_mixed_media_ranks_backward
tests/model/test_glm53_decoder_layer.py::TestGlm53DecoderLayerCompile::test_dense_layer_forward_compiles_with_dynamic_cu_seqlens
tests/model/test_glm53_kda.py::TestKDAGate::test_fused_kda_gate_matches_naive_reference
tests/model/test_glm53_kda.py::TestKDAModuleParity::test_kda_module_matches_hf_single_document
tests/model/test_glm53_kda.py::TestKDAModuleParity::test_kda_chunk_backward_matches_hf
tests/model/test_glm53_kda.py::TestKDAModuleParity::test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards
tests/model/test_glm53_kda.py::TestKDASequenceParallel::test_forward_for_sp_matches_non_sp
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[None-1]
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[all2all-4]
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy[all2all-8]
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEGradientParity::test_full_crop_fsdp_gradients_match_hf
tests/model/test_gpt_oss_moe.py::TestGptOss::test_fsdp_accuracy[cuda-None-1-1]
tests/model/test_gpt_oss_moe.py::TestGptOss::test_fsdp_accuracy[cuda-all2all-2-2]
tests/model/test_gpt_oss_moe.py::TestGptOss::test_fsdp_accuracy[cuda-all2all-4-1]
tests/model/test_moe.py::TestMoE::test_moe_config[torch-bfloat16-cuda]
tests/model/test_moe.py::TestDistributedMoE::test_parallel_accuracy[torch-bfloat16-cuda-all2all-0-0]
tests/model/test_moe.py::TestDistributedMoE::test_parallel_accuracy[torch-bfloat16-cuda-all2all-1-2]
tests/model/test_qwen3_5.py::TestQwen3_5_VL::test_qwen3_5_vl_run[cuda-1-0-02]
tests/model/test_qwen3_5.py::TestQwen3_5_VL::test_qwen3_5_vl_run[cuda-2-0-02]
tests/model/test_qwen3_5.py::TestQwen3_5_VL::test_qwen3_5_vl_run[cuda-4-0-02]
tests/model/test_qwen3_5.py::TestQwen3_5_VL::test_qwen3_5_vl_run_mtp[cuda-1-0-01]
tests/model/test_qwen3_5.py::TestQwen3_5_VL::test_qwen3_5_vl_run_mtp[cuda-4-0-01]
tests/model/test_qwen3_5_dense.py::TestQwen3_5_VLDense::test_decoder_layer_bitwise_parity[cuda-0]
tests/model/test_qwen3_5_dense.py::TestQwen3_5_VLDense::test_model_forward_bitwise_reduced_layers[cuda]
tests/model/test_qwen3_5_dense.py::TestQwen3_5_VLDense::test_vl_forward_parity[cuda]
tests/model/test_qwen3_dense.py::TestQwen3Dense::test_sliding_windows[True-4-2048]
tests/model/test_qwen3_dense.py::TestQwen3Dense::test_sliding_windows[True-6-1024]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_fsdp_accuracy[cuda-None-1-qwen3_moe]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_fsdp_accuracy[cuda-None-1-qwen3_moe_fope]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_fsdp_accuracy[cuda-all2all-4-qwen3_moe]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_fsdp_accuracy[cuda-all2all-4-qwen3_moe_fope]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_fsdp_accuracy[cuda-all2all-8-qwen3_moe]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_sliding_windows[True-4-2048]
tests/model/test_qwen3_moe.py::TestQwen3MoE::test_sliding_windows[True-6-1024]
tests/model/test_qwen3_tile_embedding.py::TestQwen3Dense4B::test_qwen3vl_tie_embedding[cuda-1]
tests/model/test_qwen3_tile_embedding.py::TestQwen3Dense4B::test_tie_embedding[cuda-1]
tests/module/attention/test_dsa_mla.py::TestDSAAttention::test_compiled_attention_matches_eager
tests/module/attention/test_dsa_mla.py::TestAcceleratedSparseMLA::test_compiled_cudnn_backward_matches_tilelang
tests/ops/test_cute_dsl_indexer_topk.py::test_cute_dsl_indexer_matches_torch_for_packed_causal_ranges[True]
tests/ops/test_grouped_gemm_triton.py::test_grouped_gemm_triton
tests/ops/test_hc_post.py::TestHCPostFused::test_forward_matches_reference[1000]
tests/ops/test_hc_post.py::TestHCPostFused::test_forward_matches_reference[2048]
tests/ops/test_hc_post.py::TestHCPostFused::test_forward_matches_reference[4096]
tests/ops/test_hc_post.py::TestHCPostFused::test_forward_no_worse_than_reference_vs_fp32
tests/ops/test_hc_post.py::TestHCPostFused::test_backward_matches_reference
tests/ops/test_hc_post.py::TestHCPostFused::test_compile_fullgraph
tests/ops/test_lmdeploy_fp8_index.py::test_indexer_fp8_quant_matches_ue8m0_reference
tests/ops/test_lmdeploy_fp8_index.py::test_lmdeploy_adapter_uses_sequence_context_sp_ranges
tests/ops/test_rms_norm.py::TestNativeRMSNorm::test_compiled_backward_of_3d_input_costs_the_same_as_2d
tests/ops/test_sparse_mla_compile.py::TestSparseMLACompile::test_topk_indices_matches_eager
tests/ops/test_sparse_mla_compile.py::TestSparseMLACompile::test_sparse_mla_with_padded_indices_matches_eager
tests/optim/test_muon.py::TestNewtonSchulz::test_triton_vs_pytorch
tests/profiler/test_prober.py::TestAccProberForwardRecords::test_acc_prober_records_both_attention_types
tests/profiler/test_prober.py::TestAccProberForwardRecordsCompiled::test_acc_prober_records_with_compile
tests/profiler/test_prober.py::TestAccProberGatedDeltaNetInternalsCompiled::test_gated_deltanet_internals_with_compile
tests/profiler/test_prober.py::TestAccProberMoEMLPCompiled::test_moe_mlp_shared_experts_with_compile
tests/profiler/test_prober.py::TestAccProberMHAFullgraph::test_mha_fullgraph_with_prober_dumps_qk_norm
tests/rl/test_qwen35_vl_moe_async_train_2step.py::TestQwen35VLMoEAsyncTrain2Step::test_qwen35_vl_moe_async_train_2step_and_metrics
tests/rl/test_rl_colocate_trainer_integration.py::TestRLColocateTrainerIntegration::test_rl_train_with_sft
tests/rl/test_update_weight_disaggregated.py::TestUpdateWeightDisaggregated::test_lmdeploy_disaggregated_update_weight_and_generate
tests/train/test_glm52_sft_smoke.py::TestTinyGlm52SFT::test_one_step_sft_produces_finite_loss
tests/train/test_trainer.py::TestHooksConfig::test_async_hf_save_hook_timing
tests/train/test_trainer.py::TestHooksConfig::test_hooks_config
tests/utils/test_internal_metrics.py::TestInternalMetricsRecorder::test_internal_metrics_run
```

### B（13）：Transformers 视频处理器 API

```text
tests/datasets/test_qwen35_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_pretrain_video[False]
tests/datasets/test_qwen35_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_pretrain_video[True]
tests/datasets/test_qwen35_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_sft_video[False]
tests/datasets/test_qwen35_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_sft_video[True]
tests/datasets/test_qwen3_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_pretrain_video
tests/datasets/test_qwen3_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_sft_video[False]
tests/datasets/test_qwen3_vl_tokenize_fn.py::TestMLLMTokenizeFn::test_qwen3_vl_sft_video[True]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_fsdp_qwen3_run[cuda-1-False-0-01]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_fsdp_qwen3_run[cuda-2-False-0-01]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_fsdp_qwen3_run[cuda-8-False-0-01]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_qwen3vl_run[cuda-1-0-01]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_qwen3vl_run[cuda-2-0-01]
tests/model/test_qwen3_vl.py::TestQwen3VL::test_qwen3vl_run[cuda-8-0-01]
```

### C（7）：异步 pytest 插件

```text
tests/rl/test_sandbox_pool.py::test_group_creation_provisioning_primary_api_and_release_order
tests/rl/test_sandbox_pool.py::test_provision_failure_rolls_back_whole_attempt_before_retry
tests/rl/test_sandbox_pool.py::test_create_failure_rolls_back_only_returned_members[target]
tests/rl/test_sandbox_pool.py::test_create_failure_rolls_back_only_returned_members[agent]
tests/rl/test_sandbox_pool.py::test_unhealthy_member_rolls_back_the_group
tests/rl/test_sandbox_pool.py::test_cancellation_cleans_up_all_returned_members
tests/rl/test_sandbox_pool.py::test_rate_limiter_is_acquired_for_every_physical_create
```

### D（1）：32 GPU 申请超过 Ray 可用量

```text
tests/rl/test_update_weight_colocate.py::TestUpdateWeightColocate::test_lmdeploy_colocate_ipc_update_weight_and_generate
```

### E（13）：全局编译状态污染

```text
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_pure_text_forward
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_image_splice_matches_placeholder_count
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_video_splice_uses_type_2_and_flattens_grid
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_every_pack_calls_the_vision_tower_exactly_once[text]
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_every_pack_calls_the_vision_tower_exactly_once[image]
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_every_pack_calls_the_vision_tower_exactly_once[image_and_video]
tests/model/test_glm53_compose.py::TestGlm53ComposeForward::test_pack_with_an_image_sample_and_a_video_sample
tests/model/test_glm53_decoder_layer.py::TestGlm53DenseDecoderLayer::test_forward_finite_and_shape_preserving
tests/model/test_glm53_decoder_layer.py::TestGlm53DenseDecoderLayer::test_mhc_cfg_none_matches_plain_dense_decoder_layer
tests/model/test_glm53_decoder_layer.py::TestGlm53DenseDecoderLayer::test_grad_flows_through_hc_params
tests/model/test_glm53_decoder_layer.py::TestGlm53MoEDecoderLayer::test_forward_finite_and_shape_preserving
tests/model/test_glm53_decoder_layer.py::TestGlm53MoEDecoderLayer::test_mhc_cfg_none_matches_plain_moe_decoder_layer
tests/model/test_glm53_decoder_layer.py::TestGlm53MoEDecoderLayer::test_grad_flows_through_hc_params_and_experts
```

### F（6）：`freeze_dsa_indexer=False` 配置

```text
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEInitWeights::test_init_weights_covers_every_parameter
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEInitWeights::test_init_weights_matches_hf_init_for_gate_and_hc_params
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEFp32Params::test_only_the_sinkhorn_and_gate_scalars_are_pinned_to_fp32
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEForwardBackward::test_forward_backward_all_trainable_params_get_gradient
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEForwardBackward::test_mtp_block_builds_and_forwards
tests/model/test_glm53_text_moe.py::TestGlm53TextMoEForwardBackward::test_optimizer_step_updates_text_model
```

### G（1）：默认注意力后端的 64 头对齐要求

```text
tests/model/test_glm53_text_moe.py::TestNoPEDSAMLAConfigValidatesAssignment::test_backend_assignment_is_validated
```

### H（4）：TileLang 小尺寸 indexer 断言

```text
tests/model/test_glm52_moe.py::TestGlm52ExplicitDsaDataflow::test_model_forward_backward_with_explicit_dsa_dataflow
tests/model/test_glm52_moe.py::TestGlm52SequenceParallel::test_mtp_loss_and_gradients_match_full_sequence
tests/model/test_glm52_mtp_checkpoint_repro.py::TestGlm52CompiledMTPCheckpoint::test_topk_offload_uses_pinned_memory_and_restores_ids
tests/model/test_glm52_mtp_checkpoint_repro.py::TestGlm52MicroBatchMTPCheckpoint::test_nested_micro_batch_inputs_preserve_gradients
```

### I（1）：CUDA allocator 严格下降断言

```text
tests/train/test_trainer.py::test_resume_and_load_checkpoint_cfg
```

### J（1）：DeepEP CUDA launch failure，待查

```text
tests/engine/test_moe_train_engine_deepep_expert_tp.py::TestMoETrainEngineDeepEPExpertTP::test_deepep_expert_tp_expert_only_grad_norm_matches_single_model_baseline
```

### K（1）：DCP 分布式超时，待查

```text
tests/engine/test_glm52_moe_train_engine.py::TestGlm52CheckpointEngine::test_dcp_round_trip_preserves_model_and_optimizer
```

### L（1）：Qwen3.5 视觉塔数值差，待查

```text
tests/model/test_qwen3_5_dense.py::TestQwen3_5_VLDense::test_vision_tower_bitwise_parity[cuda]
```

### M（1）：FlashMLA/cuDNN 反向 dQ 差异，待查

```text
tests/ops/test_flash_mla_cudnn_sparse_mla.py::TestFlashMlaCudnnSparseMLA::test_forward_backward_matches_torch_reference
```

## 最终判断

本地 138 项已逐一映射到上述 13 类。其中 109 项首先由明确的本地环境或资源条件阻断（A–D）；29 项表现为编译状态、测试配置、算子限制、脆弱断言或尚待复现的数值/分布式问题（E–M）。#2108 CI 的 22 项失败只有 16 项与本地测试 ID 重合；应按相同代码提交和依赖环境复验，不能把本地独有的 122 个失败直接归因于 #2108。

后续修复使这 138 个原失败项均至少在单项或分段真实回归中通过；完整 `zdev/run_test.sh` 仍需在最终提交上跑完。另有 Ray 2.54.1 dashboard agent 的 `nvidia-smi` 探测偶发超过 raylet 的 15 秒启动窗口，属于尚未消除的环境时序风险。
