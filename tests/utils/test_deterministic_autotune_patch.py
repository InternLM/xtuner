"""确定性 autotune patch（xtuner.v1._patch_triton_autotune_for_determinism）的行为测试。

- TestDeterministicAutotunePatch
  - test_plain_triton_autotune_trimmed: 裸 triton.autotune 装饰路径的 configs 被裁至第一个
  - test_fla_cache_autotune_trimmed: fla fla_cache_autotune（CachedAutotuner）路径的 configs 同样被裁至第一个
  - test_idempotent: 重复执行 patch 不发生二次包装
"""

import pytest
import triton
import triton.language as tl  # noqa: F401  # @triton.jit 依赖该模块属性
from triton.runtime.autotuner import Autotuner

from xtuner.v1 import _patch_triton_autotune_for_determinism


_TWO_CONFIGS = [triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)]


@pytest.fixture()
def restore_autotuner_init():
    # 每个用例从"未 patch"的干净状态开始；结束后还原到调用前状态，避免污染其他测试
    saved = Autotuner.__init__
    if getattr(saved, "_xtuner_deterministic_patched", False):
        Autotuner.__init__ = saved._xtuner_original_init
    yield
    Autotuner.__init__ = saved


class TestDeterministicAutotunePatch:
    """XTUNER_DETERMINISTIC=true 时 autotune 固定为第一个 config 的行为"""

    def test_plain_triton_autotune_trimmed(self, restore_autotuner_init):
        # 裸 triton.autotune 装饰的 kernel，configs 被裁至第一个
        _patch_triton_autotune_for_determinism()

        @triton.autotune(configs=list(_TWO_CONFIGS), key=[])
        @triton.jit
        def k(x):
            pass

        assert len(k.configs) == 1
        assert k.configs[0].num_warps == 4

    def test_fla_cache_autotune_trimmed(self, restore_autotuner_init):
        # fla>=0.5 的 fla_cache_autotune（CachedAutotuner）绕行路径，configs 同样被裁至第一个
        fla_cache = pytest.importorskip("fla.ops.utils.cache", reason="需要 fla>=0.5")
        fla_cache_autotune = getattr(fla_cache, "fla_cache_autotune", None)
        if fla_cache_autotune is None:
            pytest.skip("当前 fla 版本无 fla_cache_autotune")
        _patch_triton_autotune_for_determinism()

        @fla_cache_autotune(configs=list(_TWO_CONFIGS), key=[])
        @triton.jit
        def k(x):
            pass

        assert len(k.configs) == 1
        assert k.configs[0].num_warps == 4

    def test_idempotent(self, restore_autotuner_init):
        # 重复执行 patch 不发生二次包装
        _patch_triton_autotune_for_determinism()
        wrapped = Autotuner.__init__
        _patch_triton_autotune_for_determinism()
        assert Autotuner.__init__ is wrapped
