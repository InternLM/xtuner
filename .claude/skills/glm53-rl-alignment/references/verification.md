# GLM-5.3-Flash BF16 验证记录

2026-11-27，在 NVIDIA H200 上使用三个固定提交验证了配置和 KPool 算子。

| 组件 | 分支 | 提交 |
| --- | --- | --- |
| XTuner | `main` | [79b9615c](https://github.com/InternLM/xtuner/commit/79b9615c53e1b23e11d4dc37fd7eb66bbcefb3df) |
| XTuner | `agent-sandboox-rl-preview` | [a58613e0](https://github.com/InternLM/xtuner/commit/a58613e0d3c99dd1df9374dc7fc08555cb076fa7) |
| LMDeploy | `main` | [6856d3b3](https://github.com/InternLM/lmdeploy/commit/6856d3b3dac037c53ca51bd8f23538f156053a3d) |

## 测量结果

使用原版 GLM-5.3-Flash-BF16 分片 checkpoint，文本侧 45 层。读取第一个带 indexer 的层（层号 3，从 0 开始），严格加载其全部 indexer 权重。在 CPU 上用种子 20261127 生成相同的 hidden states 和 query latents，再转为 CUDA BF16。序列长 2305，超过 `index_topk=2048`，后 257 行覆盖了历史 token 需要筛选的区间。两支训练端均使用 32 个 index heads、pool size 4、query chunk size 1024，关闭 SP。

| 对照 | 比较对象 | H200 结果 |
| --- | --- | --- |
| XTuner main 与 preview | 完整的 Top-K 索引张量，含元素顺序和 padding | `int32[2305, 1, 2560]` 逐元素相等 |
| XTuner 与 LMDeploy prefill | 使用真实 BF16 投影、缓存写入与 paged Top-K，逐行比较有效 token 集合 | 2305 行中有差异的行数为 **0** |

本次 XTuner main 输出张量原始字节的 SHA-256 为 `bbfaf2bd879718f35db33b5d90d26ab9e5697e6f7d13029c0f933c1e6cdb63dd`。该摘要用于辨认此组输入和环境的结果。完整 45 层 RL 的精度比较还需要同一批轨迹上的 logprob 和 mismatch K3 KL。

同批验证还覆盖了以下配置。

- 两个 XTuner 分支在 `model_cfg.float8_cfg=None` 时均为 routed experts 构造 `GroupedLinear`。
- 调用两个分支的 rollout worker 配置转换函数，提供测试用 Ray placement 元数据后，均构造出 LMDeploy PyTorch 引擎参数 `model_format="fp8"`、`enable_return_routed_experts=True`、`hf_overrides["fp32_lm_head"]=True`。训练请求的路由返回开关也已开启。LMDeploy 模型配置解析得到 `fp8_quant_scope="moe_only"`，KPool 缓存为 BF16。
- mHC BF16 stream 前向与梯度用例在每个 XTuner 分支各通过 4 项。LMDeploy 的 prefill 分派及 KPool 精度用例通过 43 项。两组上游测试的选择条件列于下方。

## 复跑单层对照

使用 SM90+ GPU 和能加载这三个提交的 CUDA 环境。实测环境包含 Python 3.12、PyTorch 2.9.1、CUDA 12.8、Transformers 5.17.0、TileLang 0.1.11 及 FlashMLA。将 `MODEL_PATH` 设为本地原版分片 checkpoint 目录，其中包含 `config.json`、`model.safetensors.index.json` 和权重文件。探针直接读取索引层权重，无需训练数据或 Ray 集群。[探针源码](../scripts/verify_kpool.py)

在 skill 根目录执行。每次 Python 调用独立导入指定 checkout，参考张量写入临时目录。

```bash
set -eu
: "${MODEL_PATH:?Set MODEL_PATH to your local GLM-5.3-Flash-BF16 checkpoint}"
probe="$PWD/scripts/verify_kpool.py"
work=$(mktemp -d)

git clone https://github.com/InternLM/xtuner.git "$work/xtuner-main"
git -C "$work/xtuner-main" switch --detach 79b9615c53e1b23e11d4dc37fd7eb66bbcefb3df
git -C "$work/xtuner-main" worktree add --detach "$work/xtuner-preview" a58613e0d3c99dd1df9374dc7fc08555cb076fa7
git clone https://github.com/InternLM/lmdeploy.git "$work/lmdeploy"
git -C "$work/lmdeploy" switch --detach 6856d3b3dac037c53ca51bd8f23538f156053a3d

export LMDEPLOY_GLM53_KPOOL_BF16=1
PYTHONPATH="$work/xtuner-main:$work/lmdeploy" \
    python "$probe" main "$MODEL_PATH" "$work/main.pt"

XTUNER_GLM53_KPOOL_BF16=1 PYTHONPATH="$work/xtuner-preview:$work/lmdeploy" \
    python "$probe" preview "$MODEL_PATH" "$work/main.pt"

PYTHONPATH="$work/lmdeploy" \
    python "$probe" lmdeploy "$MODEL_PATH" "$work/main.pt"
```

预期 preview 输出 `"identical": true`，LMDeploy 输出 `"rows_with_different_token_sets": 0`。探针会检查每行的因果范围、入选数量和重复 ID，发现差异时以非零状态退出。

若需复跑前述上游回归，接着在同一个 shell 中运行。

```bash
for branch in main preview; do
    (cd "$work/xtuner-$branch" && PYTHONPATH="$PWD:$work/lmdeploy" \
        python -m pytest -q tests/model/test_glm53_mhc.py -k bf16_streams)
done
(cd "$work/lmdeploy" && PYTHONPATH="$PWD" python -m pytest -q \
    tests/pytorch/kernel/test_glm53_prefill_switch.py \
    tests/pytorch/kernel/test_kpool_precision.py -k 'not fp8 and not paged_scores')
```

## 上游实现

- XTuner main 的后端、FP32 预处理和几何约束见 [NoPE DSA](https://github.com/InternLM/xtuner/blob/79b9615c53e1b23e11d4dc37fd7eb66bbcefb3df/xtuner/v1/model/moe/glm53/nope_dsa_mla.py) 和 [KPool adapter](https://github.com/InternLM/xtuner/blob/79b9615c53e1b23e11d4dc37fd7eb66bbcefb3df/xtuner/v1/ops/sparse_mla/lmdeploy_kpool.py)。
- preview 的运行时探测与 BF16 压缩见 [preview KPool](https://github.com/InternLM/xtuner/blob/a58613e0d3c99dd1df9374dc7fc08555cb076fa7/xtuner/v1/ops/sparse_mla/kpool.py) 和 [XTuner #2269](https://github.com/InternLM/xtuner/pull/2269)。
- rollout 缓存精度、短 prefill 分派和 MoE-only 量化条件见 [LMDeploy #5115](https://github.com/InternLM/lmdeploy/pull/5115)、[LMDeploy #5110](https://github.com/InternLM/lmdeploy/pull/5110) 和 [GLM-5.3 builder](https://github.com/InternLM/lmdeploy/blob/6856d3b3dac037c53ca51bd8f23538f156053a3d/lmdeploy/pytorch/configurations/glm5_next.py)。
- mHC post 的 FP32 累加见 [XTuner #2260](https://github.com/InternLM/xtuner/pull/2260)。
