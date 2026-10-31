import os
from pathlib import Path
from typing import Any, cast


# Set cache paths before importing Triton, TileLang, or CuTe DSL: each may read
# its environment once during import. A stable shared root lets later jobs reuse
# compiled graphs and kernels across processes and nodes.
if compile_cache_dir := os.getenv("XTUNER_COMPILE_CACHE_DIR"):
    cache_root = Path(compile_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    for name, directory in (
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TILELANG_CACHE_DIR", "tilelang"),
        ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        os.environ.setdefault(name, str(cache_root / directory))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")


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
