"""A missing FlashAttention install must fail loudly, not hand back None."""

import sys

import pytest

import xtuner.v1.utils
from xtuner.v1.ops.flash_attn import get_flash_attn_varlen


@pytest.fixture
def cuda_without_flash_attn(monkeypatch):
    """Pretend we are on CUDA with neither FlashAttention wheel importable."""
    monkeypatch.setattr(xtuner.v1.utils, "get_device", lambda: "cuda")
    # A None entry in sys.modules makes the matching import raise ImportError,
    # whether or not the package is installed in the test environment.
    for module in ("flash_attn.flash_attn_interface", "flash_attn_interface"):
        monkeypatch.setitem(sys.modules, module, None)


@pytest.mark.parametrize("use_fa3", ["0", "1"], ids=["fa2", "fa3"])
def test_missing_flash_attn_raises_named_error(cuda_without_flash_attn, monkeypatch, use_fa3):
    monkeypatch.setenv("XTUNER_USE_FA3", use_fa3)
    version = "3" if use_fa3 == "1" else "2"
    with pytest.raises(ImportError, match=f"Import FlashAttention {version} failed"):
        get_flash_attn_varlen()
