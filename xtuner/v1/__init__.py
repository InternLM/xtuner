import os
from typing import Any, cast


def _patch_triton_autotune_for_determinism() -> None:
    # 必须放在 xtuner.v1 初始化最前面：FLA kernel 在导入时就会读取 triton.autotune 装饰器。
    # 如果等到 GatedDeltaNet 模块导入后再 patch，单独使用 xtuner 包的场景可能已经错过时机。
    import triton

    original_autotune = triton.autotune
    if getattr(original_autotune, "_xtuner_deterministic_patched", False):
        return

    def deterministic_autotune(configs, *args, **kwargs):
        # Triton autotune 会按 benchmark/cache 在多个 kernel config 中选一个实现；
        # 不同 cache 目录或计时抖动可能选到不同 tiling/num_warps/reduction 路径，
        # 从而改变浮点累加顺序。确定性模式固定第一个 config，并禁用 cache 结果。
        if configs:
            configs = configs[:1]
        kwargs["cache_results"] = False
        return original_autotune(configs, *args, **kwargs)

    patched = cast(Any, deterministic_autotune)
    patched._xtuner_deterministic_patched = True
    patched._xtuner_original_autotune = original_autotune
    triton.autotune = deterministic_autotune


if os.getenv("XTUNER_DETERMINISTIC") == "true":
    _patch_triton_autotune_for_determinism()

from . import patch  # noqa: E402,F401
