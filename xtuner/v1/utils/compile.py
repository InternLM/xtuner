from types import FunctionType
from typing import Callable, Generic, cast

import torch

from .device import get_device
from .logger import get_logger
from .misc import FunctionEnum, get_function_type
from .type_helper import P, T


logger = get_logger()


TARGET_DEVICE = get_device()


def _patch_sympy_mod_eval_negative_subs() -> None:
    """Fix upstream torch/sympy integration bug in ``Mod.eval``.

    Background. ``torch.utils._sympy.functions.Mod`` is declared
    ``is_nonnegative = True`` (its precondition is non-negative integer
    arguments). Its ``eval`` enforces this with ``assert p >= 0, p``. Sympy's
    ``Expr.is_constant`` probes constness by calling ``self._random`` which
    substitutes the free symbols with random reals sampled from a hard-coded
    box of ``re ∈ [-1, 1]``, then runs ``evalf`` over the substituted
    expression. When inductor builds a ``Mod(s, q)`` term during its tile /
    coalescing analysis (e.g.
    ``torch._inductor.tiling_utils.analyze_memory_coalescing`` ->
    ``extract_normalized_read_writes`` -> ``is_constant``), the substitution
    drives ``p`` to small negative reals like ``-0.99`` and the assert blows
    up with ``InductorError: AssertionError: -<num>/<den>``.

    The collision is between two contracts. Mod's "non-negative integer
    arguments" precondition is meant to protect production callers, not
    sympy's introspection passes. ``_random`` deliberately uses a real
    sampling box that violates symbolic preconditions; sympy expects
    ``eval`` to *fall back to None* (meaning "can't simplify with these
    substituted values") rather than raise.

    Fix. Replace ``Mod.eval`` with a variant that returns ``None`` when
    both arguments are concrete Numbers but the non-negative precondition is
    violated (most commonly: ``_random`` substituted reals into integer
    symbols). All other paths are identical to upstream so behavior on
    real-call inputs is unchanged.

    Without this, compiling DeepSeek-V4's DSA forward under EP trips the
    crash because the Indexer's symbolic-shape Mod expressions get sampled
    by ``analyze_memory_coalescing``. Non-EP avoided it incidentally because
    the seq-shape symbols were backed and skipped that path.
    """
    from torch.utils._sympy import functions as _torch_sympy_functions
    from sympy.core.singleton import S

    Mod = _torch_sympy_functions.Mod
    if getattr(Mod.eval, "__xtuner_negative_subs_patch__", False):
        return

    def _safe_eval(cls, p, q):
        if q.is_zero:
            raise ZeroDivisionError("Modulo by zero")
        if p is S.Zero or p in (q, -q) or q == 1:
            return S.Zero
        if q.is_Number and p.is_Number:
            # Replace upstream's ``assert p >= 0`` / ``assert q >= 1`` with a
            # graceful ``return None`` so sympy's is_constant() float-subs
            # path falls back instead of crashing the compile.
            if p < 0 or q < 1:
                return None
            return p % q
        if q.is_Number and q == 2:
            if p.is_even:
                return S.Zero
            if p.is_odd:
                return S.One
        r = p / q
        if r.is_integer:
            return S.Zero
        less = p < q
        if less.is_Boolean and bool(less) and r.is_positive:
            return p
        return None

    _safe_eval.__xtuner_negative_subs_patch__ = True  # type: ignore[attr-defined]
    setattr(Mod, "eval", classmethod(_safe_eval))


_patch_sympy_mod_eval_negative_subs()


def _patch_triton_max_block_xblock() -> None:
    """Raise inductor's ``TRITON_MAX_BLOCK["X"]`` from 4096 to 8192.

    Background. Inductor codegen picks ``XBLOCK`` per kernel via a heuristic
    (``triton_heuristics.triton_config``) that can choose ``size_hints["X"]``
    as the block size; for pointwise kernels traced from V4's varlen path the
    contiguous X dim is ``pack_max_length`` (8192 in our config). The
    heuristic then trips its own sanity check in
    ``runtime/triton_heuristics.check_max_block`` against
    ``TRITON_MAX_BLOCK["X"] = 4096`` (set in ``runtime/hints.py``) and aborts
    backward compile with::

        AssertionError: 'XBLOCK' too large. Maximum: 4096. Actual: 8192.

    The 4096 limit is a software-side sanity cap, not a hardware constraint
    — the file comment literally says "if these fail asserts submit a PR to
    increase them". H100 / H200 run XBLOCK=8192 fine for pointwise kernels
    (block size only needs to fit shared mem / register pressure, and the
    fused-kernel signatures inductor generates here are small).

    We patch the dict in-place so any subsequent ``check_max_block`` call sees
    the relaxed limit. This is required for the DSA/compressor varlen path
    whose forward operates on a packed ``[total_tokens=pack_max_length, ...]``
    tensor — before the per-sample-loop refactor each call sliced down to a
    sample and the X-hint was < 4096 so the cap was never hit.
    """
    from torch._inductor.runtime import hints as _hints

    # 32768 covers V4-Flash at ``pack_max_length=8192`` even when the inductor
    # ``min_elem_per_thread`` heuristic doubles X past ``pack_max_length`` for
    # the compressor's per-chunk ``[total_c, ratio, coff*head_dim]`` scatter
    # backward (observed XBLOCK=16384 with my Phase 2 varlen compressor).
    # Bumping the cap is a software-side change; the hardware constraint is
    # register / shared-mem pressure of the generated kernel, which inductor's
    # ``num_warps`` heuristic adapts to alongside XBLOCK.
    if _hints.TRITON_MAX_BLOCK.get("X", 0) < 32768:
        _hints.TRITON_MAX_BLOCK["X"] = 32768


_patch_triton_max_block_xblock()


class MaybeCompile(Generic[P, T]):
    """A decorator class that can conditionally apply `torch.compile` to a
    **Top(Module) level function**.

    XTuner adopts a runtime compile strategy, which applies different compile strategies based on different model
    configurations, rather than using a decorator-based approach. However, this also makes it difficult to compile some
    module-level functions at runtime. This is because during the runtime phase, other modules have already completed
    the import of the target function, and at this point, compiling the function in the original module with in-place
    modification can no longer take effect in the target modules.

    Therefore, for these module-level functions, XTuner provides the MaybeCompile decorator class. By first decorating
    the target function with this class, it becomes possible to modify the calling behavior of all modules for that
    function at runtime.
    """

    def __init__(self, func: Callable[P, T]):
        assert isinstance(func, FunctionType), f"MaybeCompile can only be used to decorate functions, but got {func}"
        assert (function_type := get_function_type(func)) == FunctionEnum.TOP_LEVEL_FUNCTION, (
            "MaybeCompile can only be used to decorate top(module) level function, "
            f"but got type {function_type}: {func}"
        )
        func = cast(Callable[P, T], func)
        self.origin_func = func
        self.func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Apply the decorator with optional torch.compile arguments."""
        return self.func(*args, **kwargs)

    def enable_compile(self, **compile_options) -> None:
        """Enable torch.compile with the given arguments."""
        if not is_compiled_function(self.func):
            self.func = torch.compile(self.origin_func, **compile_options)

    def disable_compile(self) -> None:
        """Disable torch.compile, reverting to the original function."""
        logger.info(f"Disabling torch.compile for function {self.origin_func.__name__}")
        self.func = self.origin_func


def is_compiled_function(func: Callable) -> bool:
    """Check if a function has been compiled using torch.compile."""
    return hasattr(func, "get_compiler_config")


# Create a singleton instance
maybe_compile = MaybeCompile
