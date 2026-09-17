# UltraEP native patch

XTuner 的 UltraEP FSDP 路径要求 native UltraEP 提供 BF16 grad-reduce。补丁应在对应 UltraEP checkout 中应用，并覆盖 `ultra_ep/manager.py`、`csrc/ultra_ep.cpp`、`csrc/kernels/api.cuh`、`csrc/kernels/config.cuh`、`csrc/kernels/grad_reduce.cu` 及 native tests。

补丁的 ABI 约束：

- `grad_dtype=torch.bfloat16` 时，replica NVSHMEM buffer、master pointer pool 和 GR_EP kernel 均按 BF16 解释；不能只删除 Python dtype 检查。
- `weight_data_dtype` 与 `grad_dtype` 独立，保留 BF16 权重和 FP32 兼容路径（若 native 实现暂不支持双路径，必须在构造期明确拒绝 FP32）。
- kernel 完成 `m += r` 后，XTuner 将 staging 直接绑定为 FSDP 参数 `.grad`；native 不负责替换 FSDP 对象。

应用示例：

```bash
cd /path/to/UltraEP
git apply --check /path/to/xtuner/patch/ultraep/<generated-native-bf16.patch>
git apply /path/to/xtuner/patch/ultraep/<generated-native-bf16.patch>
python setup.py build_ext --inplace
```

当前仓库已提供可编译的 [ultraep-native-bf16-grad-reduce.patch](./ultraep-native-bf16-grad-reduce.patch)。该 patch 在 UltraEP `f18d8252` 工作树上通过 `git apply --check`，并在 H200 / CUDA 12.8 / SM90 上完成编译；未应用该 patch 时，XTuner 运行时仍应报错并停止，不能退回 FP32 staging。

native patch 保留 FP32 兼容路径，并新增 BF16 路径：任务表使用字节地址，普通 kernel 对 BF16 做原子累加，确定性 kernel 使用 FP32 shared-memory accumulator 后一次转换回 BF16。两个路径均要求每个梯度 shard 的字节数按 16B 对齐。

## 已提取的多 microbatch patch

`ultraep-multimicrobatch-slots.patch` 来自 UltraEP commit `f18d8252`，父提交为 `94cab09`（v1.0.0）。该 patch 覆盖 virtual-layer slot 对应的 replica weight、scale、grad、ready buffer，以及 weight-sync/grad-reduce 的 slot 偏移；它是 XTuner `intra_layer_micro_batch > 1` 路径的 native 前置 patch。

应用顺序：先应用 `ultraep-multimicrobatch-slots.patch`，再应用 `ultraep-native-bf16-grad-reduce.patch`，最后重新编译 native extension。两个 patch 都已在对应父提交工作树上通过 `git apply --check`。

## FSDP 集成回归

安装 DeepEP 和上述 patched UltraEP 后，在单机八卡运行：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 XTUNER_TEST_ULTRA_EP=1 XTUNER_TORCH_COMPILE=0 \
  python -m pytest tests/engine/test_moe_train_engine_ultraep.py -q
```

如果使用 `build_ext --inplace` 的产物，应先将 UltraEP checkout 加入 `PYTHONPATH`。测试默认跳过，避免普通单测环境依赖 native extra。

测试通过真实 `TrainEngine` 对照 DeepEP baseline，覆盖 EP8、EP4×DP2、层内多 microbatch 和梯度累积；每个用例使用独立进程，执行两次参数更新，不 mock XTuner Module。测试关闭 `torch.compile`，不测性能。

worker 结束前显式销毁其 native UltraEP manager，避免依赖 Python 解释器退出时的模块析构顺序。该操作仅用于即将退出的独立测试进程；普通 model close 仍保留可复用的进程级 manager。

容差沿用 `test_moe_train_engine_tpep.py` 的 BF16 标准：相对容差 `1.6e-2`，loss 和梯度范数的绝对容差 `1e-5`，逐元素梯度和参数的绝对容差 `1e-4`。额外检查每个参数的梯度范数，防止绝对容差掩盖小梯度丢失。每步检查两组 optimizer 更新后，将 candidate 参数对齐到 baseline 的更新值，再验证下一次 unshard/weight-sync；因此该测试验证逐步 FSDP/梯度正确性，不代表独立训练的长期 loss 曲线一致。
