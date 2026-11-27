---
name: glm53-rl-alignment
description: 在 XTuner v1 和 LMDeploy PyTorch 之间迁移或核验 GLM-5.3-Flash 的文本 RL 配置。涵盖原版 BF16 训练、rollout 侧 MoE-only FP8、KPool Top-K、FP32 LM head 和路由回放，适用于 XTuner main 与 agent-sandboox-rl-preview。
---

# GLM-5.3-Flash 训推对齐

原版 BF16 checkpoint 的配方是训练侧保留 BF16 experts，rollout 侧只对 routed experts 使用动态 FP8。两侧使用 BF16 KPool 和 FP32 LM head。先从实际 worker 确认 XTuner、LMDeploy 的导入路径与 Git 提交，再检查 checkpoint 的 `text_config` 和已有的 `quantization_config`。已验证的版本、单层对照结果和复跑命令见[验证记录](references/verification.md)。

## 训练精度与路由回放

在创建 trainer 之前，设置传给训练 worker、rollout worker 和训练 agent loop 的配置对象。下面的 `model_cfg` 是文本模型配置，FSDP 使用 BF16 参数计算类型，rollout 使用 LMDeploy PyTorch 后端。

```python
import torch

model_cfg.float8_cfg = None
fsdp_cfg.param_dtype = torch.bfloat16
fsdp_cfg.fp32_lm_head = True
rollout_config.enable_float8 = True
rollout_config.fp32_lm_head = True
rollout_config.enable_return_routed_experts = True
training_sample_params.return_routed_experts = True
```

引擎开关允许采集 expert IDs，采样开关让训练请求取回这些 IDs。验证路由回放时，再检查返回的数据是否按 token 和层传入训练侧 router。多段请求还需核对每段返回的 token 区间。

对于读取环境变量的实验 launcher，可用以下设置映射到上述字段。`FP8=0` 关闭训练侧 FP8，此时 launcher 会忽略 `FP8_MOE_ONLY`。`ROLLOUT_FP8=1` 单独开启推理侧量化。

```bash
export FP8=0 ROLLOUT_FP8=1
export TRAIN_FP32_LM_HEAD=1 ROLLOUT_FP32_LM_HEAD=1
export ENABLE_RETURN_ROUTED_EXPERTS=1
```

## 两个训练分支的 KPool

两支均使用冻结的 DSA indexer，设置 `model_cfg.attention.freeze_dsa_indexer=True`、`indexer_balance_sp=False`。此配方的 sparse attention 后端是 `flash_mla_cudnn`。另用 `indexer_backend` 指定 KPool 的评分和选择路径。

| XTuner 分支 | 训练侧设置 |
| --- | --- |
| `main` | 设置 `model_cfg.attention.indexer_backend = "lmdeploy_bf16"` 和 `model_cfg.attention.indexer_kpool_fp32_prep = True`。构建器会为该后端补全未指定的预处理选项，显式传 False 会报错。 |
| `agent-sandboox-rl-preview` | 设置 `model_cfg.attention.indexer_backend = "tilelang"`，启动训练 worker 前执行 `export XTUNER_GLM53_KPOOL_BF16=1`。该分支在运行时探测 LMDeploy BF16 压缩接口，依赖或设备不匹配时可能回到原生路径。 |

已核对的 LMDeploy BF16 路径要求 SM90+、TileLang 和支持 `out_dtype` 的 LMDeploy KPool 压缩接口。支持的几何为 head dim 128、pool size 4、32/64/128 个 index heads、Top-K 2048 或 8192。key 归一化、head 权重投影及池化中间量使用 FP32，池化结果写回 BF16。

## LMDeploy rollout

在 LMDeploy 服务及计算 worker 启动前设置三个原生开关。Ray 部署时检查实际 worker 环境。KPool 精度会改变缓存布局，切换后需重建引擎和缓存。

```bash
export LMDEPLOY_FP8_MOE_ONLY=1
export LMDEPLOY_GLM53_KPOOL_BF16=1
export LMDEPLOY_MLA_DISABLE_DENSE_PREFILL=1
```

- **MoE**。XTuner 将 `enable_float8=True` 转为引擎的 `model_format="fp8"`。原版 BF16 权重由 LMDeploy 生成带有 `lmdeploy_patched=True` 的 FP8 配置，GLM builder 据此设置 `fp8_quant_scope="moe_only"`。核对最终量化方法应为 `moe → fp8`、`linear/norm → None`。对于已量化的 checkpoint，读取实际 scope。
- **KPool**。LMDeploy 默认保存旋转后的 FP8 池。第二个开关独立控制推理侧的 query 与缓存，使其使用未旋转的 BF16 值。
- **短 prefill**。第三个开关让 `max_kv_seqlen <= index_topk` 的请求也走 sparse DSA。原版 checkpoint 的阈值为 2048。所有历史 token 均可入选时仍可直接构造因果索引。
- **mHC post**。两个训练分支均已在默认实现中使用 FP32 系数与累加，最终写回 residual 的 dtype。
