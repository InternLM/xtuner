#!/bin/bash
# 验证 recompute_cfg：跑两次同一份训练，一次不选 unit、一次选 save_attn，比对结果。
#
# 判据（两条都必须成立）：
#   1. step-1 loss 逐位相同 —— checkpoint 只改 backward，前向不能被动到；
#   2. 峰值显存显著上升 —— unit 真的把激活留驻了，而不是配置被静默忽略。
#
# 用 eager（torch_compile=False）跑：编译会改变 kernel，从而改变 loss 的低位，
# 那样就无法把差异归因到 SAC 本身。
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SAC_VERIFY_OUT:-/tmp/sac_verify}"
PY="${PYTHON:-/mnt/shared-storage-user/yehaochen/miniconda3/envs/py312-pt210/bin/torchrun}"
export QWEN3_MOE_PATH="${QWEN3_MOE_PATH:-/mnt/shared-storage-user/llmrazor-share/model/Qwen3-30B-A3B}"
export ALPACA_PATH="${ALPACA_PATH:-/mnt/shared-storage-user/llmrazor-share/data/alpaca}"
export EP_DISABLE_GIN=1 EP_REUSE_NCCL_COMM=0
mkdir -p "$OUT"

for unit in none attn; do
  echo "==> 跑 recompute_cfg=$unit ..."
  (cd /tmp && PYTHONPATH="$REPO" timeout 3600 "$PY" --master-port "${MASTER_PORT:-21971}" \
      --nproc-per-node "${NPROC:-8}" -m xtuner.v1.train.cli.sft \
      --config "$REPO/ci/config/sac_verify_$unit.py") > "$OUT/$unit.log" 2>&1
  [ $? -eq 0 ] || { echo "失败，日志见 $OUT/$unit.log"; grep -m3 -iE "error|Traceback" "$OUT/$unit.log"; exit 1; }
done

"${PYTHON_BIN:-python}" - "$OUT" <<'PY'
import re, sys, pathlib

out = pathlib.Path(sys.argv[1])
res = {}
for unit in ("none", "attn"):
    log = out / f"{unit}.log"
    text = log.read_text(errors="ignore")
    # 只看 rank 0：各 rank 峰值本就不同，跨 rank 比较得到的是 rank 差异而非配置差异。
    rows = re.findall(r"RANK 0\].*?Step (\d+)/\d+.*?reduced_llm_loss: ([0-9.]+).*?max_memory: ([0-9.]+) GB", text)
    by_step = {int(s): (loss, float(mem)) for s, loss, mem in rows}
    if 1 not in by_step or len(by_step) < 2:
        print("没读到 %s 至少两步的 rank-0 指标，日志：%s" % (unit, log))
        sys.exit(1)
    # loss 取第一步（纯前向）；显存取最后一步：第一步的峰值被权重加载/优化器状态初始化这类
    # 一次性分配盖住，激活的稳态峰值要到第二步才显出来。
    res[unit] = (by_step[1][0], by_step[max(by_step)][1])

l0, m0 = res["none"]
l1, m1 = res["attn"]
print("")
print("  无 unit      step1_loss=%s  稳态峰值=%.2f GB" % (l0, m0))
print("  save_attn    step1_loss=%s  稳态峰值=%.2f GB  (+%.2f GB)" % (l1, m1, m1 - m0))
print("")

ok_loss = l0 == l1
ok_mem = (m1 - m0) > 1.0
print("  [%s] 前向未受影响：step-1 loss 逐位相同" % ("PASS" if ok_loss else "FAIL"))
print("  [%s] unit 确实生效：稳态峰值显存上升 > 1 GB" % ("PASS" if ok_mem else "FAIL"))
sys.exit(0 if (ok_loss and ok_mem) else 1)
PY
