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
