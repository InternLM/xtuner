"""Intent base classes for XTuner submodule tests.

A test's *intent* — what kind of guarantee it makes — is the single most useful
thing a reviewer wants to know before reading a line of its body. These three
base classes make that intent part of the class **type**, so it is visible at
the point of review (in the class name and in the ``pytest`` collection tree)
without opening the test:

* :class:`ParityTest` — reproduces the HuggingFace reference bit-for-bit.
* :class:`CorrectnessTest` — matches an independent oracle within a tolerance.
* :class:`SmokeTest` — runs end-to-end and stays finite; no oracle.

Each intent owns a *different* check standard, and the standards are kept in
separate methods on purpose. The point of three classes is not to save a few
lines of ``assert`` — one configurable helper would do that. The point is that
the class a test extends, and the ``assert_*`` it calls, *state its intent*.
Collapsing them into one parametrised helper would bury that intent in an
argument and defeat the reason they exist.

These are pure mixins (they do not inherit from anything), so a distributed
correctness test composes the intent with the process-spawning base:

    class TestFooCorrectness(CorrectnessTest, DeterministicDDPTestCase):
        ...
"""

import torch


__all__ = ["ParityTest", "CorrectnessTest", "SmokeTest"]


class ParityTest:
    """Bitwise parity against the HuggingFace reference.

    Extending ``ParityTest`` declares that the class proves XTuner reproduces HF
    op-for-op. The check is strict bitwise equality via :func:`torch.equal`;
    there is deliberately **no tolerance knob**. If a comparison cannot be made
    bitwise — a legitimately different reduction order (cuBLAS vs. Triton), a
    discrete tie that resolves differently, a fused kernel that only matches
    within bf16 rounding — then it is *not* parity. Classify it as
    :class:`CorrectnessTest` with a documented tolerance instead.
    """

    def assert_parity(self, actual: torch.Tensor, expected: torch.Tensor, *, msg: str = "") -> None:
        """Assert ``actual`` equals ``expected`` bit-for-bit.

        Args:
            actual (torch.Tensor): The tensor produced by XTuner.
            expected (torch.Tensor): The HuggingFace reference tensor.
            msg (str): Extra context appended to the failure message.
        """
        if actual.shape != expected.shape:
            raise AssertionError(f"parity shape mismatch: {actual.shape} vs {expected.shape}. {msg}")
        if actual.dtype != expected.dtype:
            raise AssertionError(f"parity dtype mismatch: {actual.dtype} vs {expected.dtype}. {msg}")
        if not torch.equal(actual, expected):
            # Report the worst element so a real regression is debuggable, while the
            # pass/fail decision stays exact (``torch.equal``) — parity tolerates nothing.
            max_abs = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f"parity requires bitwise equality; max|Δ| = {max_abs:.3e}. {msg}")


class CorrectnessTest:
    """Correctness against an independent oracle, within a tolerance.

    This is the home for every test that compares against a *hand-written* oracle
    or an *equivalent* computation rather than the HF reference: op math vs. a
    plain eager formula, backend agreement (Triton vs. native), padded-vs-native
    equivalence, and production settings such as distributed sharding or an
    operator swap — anything whose ground truth is real but not bit-identical.

    Tolerance is a class attribute so a reviewer sees it in the class header and
    a subclass can widen it for bf16 without touching call sites::

        class TestKVCompressorCorrectness(CorrectnessTest):
            atol = rtol = 2e-2  # bf16 accumulation, widened from the fp32 default

    ``atol`` / ``rtol`` default to ``1e-3`` (an fp32-scale default).
    """

    atol: float = 1e-3
    rtol: float = 1e-3

    def assert_correct(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float | None = None,
        rtol: float | None = None,
        msg: str = "",
    ) -> None:
        """Assert ``actual`` matches the oracle ``expected`` within tolerance.

        Args:
            actual (torch.Tensor): The tensor produced by the code under test.
            expected (torch.Tensor): The independent-oracle tensor.
            atol (float | None): Absolute tolerance; falls back to the class ``atol``.
            rtol (float | None): Relative tolerance; falls back to the class ``rtol``.
            msg (str): Extra context appended to the failure message.
        """
        torch.testing.assert_close(
            actual,
            expected,
            atol=self.atol if atol is None else atol,
            rtol=self.rtol if rtol is None else rtol,
            msg=msg or None,
        )


class SmokeTest:
    """A smoke test: the code runs end-to-end and produces finite outputs.

    A smoke test owns no oracle — correctness belongs to the ops
    (:class:`CorrectnessTest`) and to whole-model parity (:class:`ParityTest`).
    Its only guarantee is that wiring and shapes hold together: forward (and,
    where relevant, backward) run without error and every output stays finite.
    """

    def assert_smoke(self, *tensors: torch.Tensor) -> None:
        """Assert every tensor is finite (no ``NaN`` / ``Inf``).

        Args:
            *tensors (torch.Tensor): Outputs (or gradients) to check for finiteness.
        """
        for i, t in enumerate(tensors):
            if not torch.isfinite(t).all():
                raise AssertionError(f"smoke check failed: tensor #{i} has non-finite values")
