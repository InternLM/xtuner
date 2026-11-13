import os
from pathlib import Path


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
    # 必须放在 xtuner.v1 初始化最前面：第三方 kernel 在导入时就会用 autotune 装饰器完成装饰。
    # hook 点选 Autotuner 基类而不是 triton.autotune 符号：fla>=0.5 的 kernel 改用自带的
    # fla_cache_autotune（CachedAutotuner 子类）装饰，不经过 triton.autotune，符号级替换覆盖不到；
    # 基类 __init__ 裁剪 configs 对所有下游 autotuner 生效，且不受装饰时机/导入顺序影响。
    import inspect

    from triton.runtime.autotuner import Autotuner

    original_init = Autotuner.__init__
    if getattr(original_init, "_xtuner_deterministic_patched", False):
        return

    accepts_cache_results = "cache_results" in inspect.signature(original_init).parameters

    def deterministic_init(self, fn, arg_names, configs, *args, **kwargs):
        # Triton autotune 会按 benchmark 在多个 kernel config 中选一个实现；
        # 不同进程/rank 的计时抖动可能选到不同 tiling/num_warps/reduction 路径，
        # 从而改变浮点累加顺序。确定性模式固定第一个 config，并禁用 cache 结果。
        if configs:
            configs = configs[:1]
        if accepts_cache_results:
            kwargs["cache_results"] = False
        original_init(self, fn, arg_names, configs, *args, **kwargs)

    setattr(deterministic_init, "_xtuner_deterministic_patched", True)
    setattr(deterministic_init, "_xtuner_original_init", original_init)
    Autotuner.__init__ = deterministic_init


if os.getenv("XTUNER_DETERMINISTIC") == "true":
    _patch_triton_autotune_for_determinism()

from . import patch  # noqa: E402,F401
