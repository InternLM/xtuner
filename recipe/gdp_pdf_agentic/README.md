# GDP PDF agentic RL recipe

GDP PDF 多模 agentic RL 的任务层：多轮 PDF 工具、数据转换与训练配置。通用的多轮工具
循环在 `xtuner/v1/rl/agent_loop/multi_turn_tool_agent_loop.py`，本 recipe 只保留
GDP 专用的工具、数据与配置。

## 目录

- `agent_loop.py`：`GdpToolAgentLoopConfig`，在通用 loop 上注入 `GdpDocTools`。
- `tools.py`：`GdpDocTools`，pymupdf 读页、渲染页图、返回文本或 data URI。
- `configs/qwen35_vl_gdp_agentic_smoke.py`：Qwen3.6-35B-A3B + Qwen3.5-9B 共卡配置。
- `gdp_pdf_convert.py`：GDP rubric 数据转换（v2 口径）。
- `gdp_agentic_jsonl.py`：生成 agentic JSONL 与配套 config。
- `tests/test_gdp_agent_loop.py`：CPU 单测，mock rollout，验证多轮流与训练字段。

## 依赖

- xtuner：包含本 recipe 的分支（`feat/multi-turn-tool-agent-loop` 或统一 preview）。
- lmdeploy：官方 main（rollout 后端）。
- 数据：`GDP_DATA_DIR`（默认指向工作区 `data/gdp_pdf`），页图与 PDF 必须在训练 pod 可读。
- judge：Qwen3.5-9B（共卡）或兼容的 OpenAI chat completions 服务。

## 数据准备

```bash
python recipe/gdp_pdf_agentic/gdp_pdf_convert.py --rubric-version v2
python recipe/gdp_pdf_agentic/gdp_agentic_jsonl.py \
  "$GDP_DATA_DIR/gdp_rubric_smoke_v2.jsonl" \
  "$GDP_DATA_DIR/gdp_agentic_qwen_smoke_v2.jsonl" \
  --profile qwen35_vl --max-inline-pages 2 --write-config
```

## 提交

在包含 `data/gdp_pdf` 与 rjob 脚本的工作区执行，`CONFIG_PATH` 指向本 recipe：

```bash
export GDP_DATA_DIR=/path/to/glm53-multimodal-agentic-rl/data/gdp_pdf
export PYTHONPATH=$XTUNER_PATH
RUN_ID=gdp-smoke-$(date +%Y%m%d%H%M) \
REPO_DIR=$XTUNER_PATH \
CONFIG_PATH=$XTUNER_PATH/recipe/gdp_pdf_agentic/configs/qwen35_vl_gdp_agentic_smoke.py \
DATA_PATH=$GDP_DATA_DIR/gdp_agentic_qwen_smoke_v2_config.json \
MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/model/Qwen3.6-35B-A3B \
JUDGE_MODEL=Qwen3.5-9B \
JUDGE_MODEL_PATH=/mnt/shared-storage-user/llmrazor-share/yushan/models/Qwen3.5-9B \
JUDGE_EVIDENCE_MATCH_MODE=lenient \
CHARGED_GROUP=llmmultimodal_gpu JOB_NAMESPACE=ailab-llmmultimodal \
bash /path/to/glm53-multimodal-agentic-rl/scripts/submit_gdp_smoke_rjob.sh --submit
```

## 测试

```bash
cd $XTUNER_PATH
PYTHONPATH=. python -m pytest recipe/gdp_pdf_agentic/tests/test_gdp_agent_loop.py -q
```