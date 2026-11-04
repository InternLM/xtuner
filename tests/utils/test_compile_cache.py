"""Verify the opt-in compilation cache reaches compiler runtimes at import time."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("override_triton", [False, True])
def test_shared_compile_cache_paths(tmp_path, override_triton):
    root = tmp_path / "shared"
    env = os.environ.copy()
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
    ):
        env.pop(name, None)
    env["XTUNER_COMPILE_CACHE_DIR"] = str(root)
    if override_triton:
        env["TRITON_CACHE_DIR"] = str(tmp_path / "custom_triton")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os, xtuner.v1; "
            "from torch._inductor.runtime.runtime_utils import cache_dir; "
            "from triton import knobs; "
            "names = ('TILELANG_CACHE_DIR', 'CUTE_DSL_CACHE_DIR', 'CUDA_CACHE_PATH', "
            "'TORCHINDUCTOR_FX_GRAPH_CACHE', 'TORCHINDUCTOR_AUTOGRAD_CACHE'); "
            "print(json.dumps({'env': {name: os.environ.get(name) for name in names}, "
            "'inductor': cache_dir(), 'triton': knobs.cache.dir}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(result.stdout)
    cache_env = actual["env"]
    assert root.is_dir()
    assert actual["inductor"] == str(root / "torchinductor")
    assert actual["triton"] == str(tmp_path / "custom_triton" if override_triton else root / "triton")
    assert cache_env["TILELANG_CACHE_DIR"] == str(root / "tilelang")
    assert cache_env["CUTE_DSL_CACHE_DIR"] == str(root / "cute_dsl")
    assert cache_env["CUDA_CACHE_PATH"] == str(root / "cuda")
    assert cache_env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert cache_env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"
