# NOTE:
# 这里故意只保留 rollout_is 的包级导出，不再像以前那样在 __init__.py 中
# eager import `.controller` / `.worker`。
#
# 原因是 `xtuner.v1.rl.loss.base_loss` 需要导入
# `xtuner.v1.rl.trainer.rollout_is.RolloutImportanceSampling`。而 Python 在加载
# `xtuner.v1.rl.trainer.rollout_is` 之前，一定会先执行当前包的 `__init__.py`。
# 如果这里继续做下面这种包级导入：
#
#     from .controller import TrainingController
#     from .worker import WorkerConfig, TrainingWorker
#
# 那么导入链会变成：
# 1. import xtuner.v1.rl.loss
# 2. loss/base_loss.py -> import xtuner.v1.rl.trainer.rollout_is
# 3. 先执行 xtuner.v1.rl.trainer.__init__
# 4. __init__ 再 import .worker
# 5. worker.py 再反向 import xtuner.v1.rl.loss
# 6. 此时 xtuner.v1.rl.loss 仍处于 partially initialized 状态，触发循环引用
#
# 最小复现示例：
#
#     # xtuner/v1/rl/trainer/__init__.py
#     from .worker import WorkerConfig
#     from .rollout_is import RolloutImportanceSampling
#
#     # xtuner/v1/rl/loss/base_loss.py
#     from xtuner.v1.rl.trainer.rollout_is import RolloutImportanceSampling
#
#     # xtuner/v1/rl/trainer/worker.py
#     from xtuner.v1.rl.loss import BaseRLLossConfig
#
#     # 任意入口
#     from xtuner.v1.rl.loss import GRPOLossConfig
#
# 这样即使入口代码只显式导入 `xtuner.v1.rl.trainer.rollout_is`，也会因为包初始化
# 提前拉起 `.worker`，最终在 `xtuner.v1.rl.loss` 和 `xtuner.v1.rl.trainer.worker`
# 之间形成闭环。
#
# 因此，凡是会触发 `worker.py` / `controller.py` 的符号，都应显式从子模块导入：
# - `from xtuner.v1.rl.trainer.worker import WorkerConfig`
# - `from xtuner.v1.rl.trainer.worker import TrainingWorker`
# - `from xtuner.v1.rl.trainer.controller import TrainingController`
from .rollout_is import (
    RolloutImportanceSampling,
    compute_is_metrics,
    compute_mismatch_metrics,
    compute_rollout_importance_weights,
    merge_rollout_is_metrics,
)


__all__ = [
    "RolloutImportanceSampling",
    "compute_rollout_importance_weights",
    "compute_is_metrics",
    "compute_mismatch_metrics",
    "merge_rollout_is_metrics",
]
