# Copyright (c) OpenMMLab. All rights reserved.
"""Exact Ulysses layout/VJP checks, without importing KDA or its kernel
dependencies.

Single-GPU custom-op tests: pytest tests/ops/test_ulysses_layout.py
Distributed diagnostics (P = 1, 2, or 8):
    A2A_RESULT_DIR=/tmp/a2a-results torchrun --standalone --nproc-per-node=P \
        tests/ops/test_ulysses_layout.py
Set A2A_TEST_LARGE=1 to additionally test [1, 65536, 8192]. Large cases need
substantial GPU memory; references gather bounded chunks, not P full tensors.
Both module-flag values are tested regardless of XTUNER_A2A_TOKEN_MAJOR.
JSON contains checks/metadata, not tensor dumps, and is written once per case.
"""

import gc
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from torch.utils.checkpoint import checkpoint


_DTYPES = (torch.float32, torch.bfloat16)
_LAYOUTS = ("contiguous", "offset", "noncontiguous")
_SWAP_SHAPES = ((37, 2, 17), (64, 1, 128), (65, 8, 32))
_OPCHECK_TESTS = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
_GATHER_TOKENS = 1024
_Layout = Literal["contiguous", "offset", "noncontiguous"]
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LayoutCase:
    tokens: int
    channels: int
    dtype: torch.dtype
    token_major: bool
    compiled: bool
    checkpointed: bool
    layout: _Layout
    roundtrip: bool = False
    triton_pack: bool = False


class TestUlyssesCPU:
    def test_import_without_triton(self) -> None:
        path = Path(__file__).resolve().parents[2] / "xtuner/v1/ops/comm/all_to_all.py"
        script = """
import builtins
import runpy
import sys
import torch
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'triton' or name.startswith('triton.') or name.endswith('_a2a_triton'):
        raise ModuleNotFoundError('Triton deliberately unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
module = runpy.run_path(sys.argv[1])
assert callable(module['ulysses_all_to_all'])
assert callable(module['all_to_all_single_autograd'])
"""
        subprocess.run([sys.executable, "-c", script, str(path)], check=True, capture_output=True, text=True)

    def test_cpu_fallback_with_all_flags(self) -> None:
        module = _a2a_module()
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=0, world_size=1)
            try:
                mesh = DeviceMesh("cpu", [0])
                for pack, token in product((False, True), repeat=2):
                    with (
                        patch.object(module, "_A2A_TRITON_PACK", pack),
                        patch.object(module, "_A2A_TOKEN_MAJOR", token),
                    ):
                        x = torch.randn(1, 37, 128, requires_grad=True)
                        upstream = torch.randn_like(x)
                        y = module.ulysses_all_to_all(x, 2, 1, mesh)
                        torch.testing.assert_close(y, x, atol=0, rtol=0)
                        torch.testing.assert_close(torch.autograd.grad(y, x, upstream)[0], upstream, atol=0, rtol=0)
            finally:
                dist.destroy_process_group()


class TestUlyssesDistributed(DistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @pytest.mark.gpu
    def test_fast_paths_forward_and_vjp(self, device: str = "cuda") -> None:
        self.create_pg(device)
        mesh = init_device_mesh(device, (self.world_size,), mesh_dim_names=("sp",))
        # Ordinary pytest spawns this test, so CI covers the real collective seam.
        for pack, token, compiled, checkpointed in product((False, True), repeat=4):
            case = _LayoutCase(37, 128, torch.float32, token, compiled, checkpointed, "contiguous", triton_pack=pack)
            if compiled:
                _check_layout(mesh, case)
            else:
                module = _a2a_module()
                with (
                    patch.object(module, "_transpose_pack_2d", wraps=module._transpose_pack_2d) as pack_calls,
                    patch.object(module, "_swap_groups", wraps=module._swap_groups) as swap_calls,
                ):
                    _check_layout(mesh, case)
                    assert bool(swap_calls.call_count) == token
                    assert bool(pack_calls.call_count) == (pack and not token)
        _check_layout(mesh, _LayoutCase(37, 128, torch.float64, True, False, False, "contiguous", triton_pack=True))


@pytest.mark.gpu
class TestUlyssesLayout:
    """Exercise the CUDA custom op locally and real collectives under
    torchrun."""

    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("shape", _SWAP_SHAPES)
    @pytest.mark.parametrize("layout", _LAYOUTS)
    def test_swap_groups(self, dtype: torch.dtype, shape: tuple[int, int, int], layout: _Layout) -> None:
        """Check all opcheck utilities, exact VJPs, ownership, and involution.

        Args:
            dtype (torch.dtype): FP32 or BF16 storage type.
            shape (tuple[int, int, int]): Input [tokens, ranks, channels] dimensions.
            layout (_Layout): Contiguous, offset, or noncontiguous input storage.
        """
        if not torch.cuda.is_available():
            pytest.skip("a2a_swap_groups is registered only for CUDA")
        _check_swap(shape, dtype, layout)

    def test_swap_empty_and_broadcast(self) -> None:
        """Exercise empty inputs and broadcast (zero-stride) storage."""
        if not torch.cuda.is_available():
            pytest.skip("a2a_swap_groups is registered only for CUDA")
        _check_swap_edges()

    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @pytest.mark.parametrize("shape", ((37, 65), (0, 65), (37, 0)))
    def test_transpose_pack(self, dtype: torch.dtype, shape: tuple[int, int]) -> None:
        """Check the transpose custom-op contract and exact noncontiguous
        VJP."""
        if not torch.cuda.is_available():
            pytest.skip("transpose_pack_2d is CUDA-only")
        transpose = _a2a_module()._transpose_pack_2d
        x = torch.randn(shape[::-1], device=_device(), dtype=dtype).t().detach().requires_grad_()
        torch.library.opcheck(transpose, (x,), test_utils=_OPCHECK_TESTS)
        y = transpose(x)
        _assert_exact(y, x.t(), "transpose forward", contiguous=True)
        dy = torch.randn_like(x).t()
        _assert_exact(torch.autograd.grad(y, x, dy)[0], dy.t(), "transpose VJP")

    @pytest.mark.skipif(
        os.environ.get("A2A_TEST_LARGE") != "1", reason="explicit opt-in: needs about 32 GiB of GPU memory"
    )
    def test_transpose_large_index(self) -> None:
        """Check addresses beyond INT32_MAX in both transpose directions."""
        x = torch.empty((262144, 16384), device=_device(), dtype=torch.bfloat16).random_(0, 256).requires_grad_()
        y = _a2a_module()._transpose_pack_2d(x)
        for start in range(0, x.shape[0], 4096):
            _assert_exact(y[:, start : start + 4096].t(), x[start : start + 4096], "large transpose")
        dy = torch.empty_like(y).random_(0, 256)
        dx = torch.autograd.grad(y, x, dy)[0]
        for start in range(0, x.shape[0], 4096):
            _assert_exact(dx[start : start + 4096], dy[:, start : start + 4096].t(), "large transpose VJP")

    def test_distributed_layout(self) -> None:
        """Run the layout matrix when pytest itself is launched under
        torchrun."""
        if "RANK" not in os.environ:
            pytest.skip("real NCCL collectives require torchrun; use this file's diagnostic entry point")
        if not torch.cuda.is_available():
            pytest.skip("distributed layout checks require CUDA and NCCL")
        with _distributed_mesh() as mesh:
            for case in _layout_cases():
                _check_layout(mesh, case)


def _a2a_module() -> ModuleType:
    # Delay CUDA/Triton dependencies until after pytest's CUDA skip.
    return importlib.import_module("xtuner.v1.ops.comm.all_to_all")


def _device() -> torch.device:
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", str(rank))))
    torch.cuda.set_device(device)
    return device


@contextmanager
def _distributed_mesh() -> Iterator[DeviceMesh]:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch distributed checks with torchrun --nproc-per-node=1, 2, or 8")
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise RuntimeError("CUDA and NCCL are required")
    _device()
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        if dist.get_backend() != "nccl":
            raise RuntimeError("The existing process group must use NCCL")
        if dist.get_world_size() not in (1, 2, 8):
            raise ValueError("The diagnostic matrix supports world sizes 1, 2, and 8")
        yield init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("sp",))
    finally:
        if owns_group:
            dist.destroy_process_group()


@contextmanager
def _isolated_case() -> Iterator[None]:
    # New closures/shapes/flags must not exhaust Dynamo's per-code cache limit.
    torch._dynamo.reset()
    try:
        yield
        torch.cuda.synchronize()
    finally:
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def _random_tensor(
    shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device, seed: int, layout: _Layout
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    a, b, c = shape
    if layout == "contiguous":
        value = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    elif layout == "offset":
        value = torch.randn(a * b * c + 1, dtype=dtype, device=device, generator=generator)[1:].view(shape)
    else:
        # Only one padded column, even for the optional multi-GiB cases.
        value = torch.randn((a, c, b + 1), dtype=dtype, device=device, generator=generator).transpose(1, 2)[:, 1:]
    assert tuple(value.shape) == shape
    assert value.is_contiguous() == (layout != "noncontiguous")
    assert (value.storage_offset() > 0) == (layout != "contiguous")
    return value.detach()


def _assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    group: dist.ProcessGroup | None = None,
    contiguous: bool = False,
) -> None:
    """Compare same-layout tensors and make numerical failures rank-symmetric.

    Args:
        actual (torch.Tensor): Local output or VJP, with arbitrary rank/axis order.
        expected (torch.Tensor): Independent reference with identical semantic axes and shape.
        label (str): Check and case identification.
        group (dist.ProcessGroup | None): Mesh group; every rank must enter the same check.
        contiguous (bool): Also require contiguous actual storage for API outputs.
    """
    error = ""
    try:
        if contiguous:
            assert actual.is_contiguous(), f"noncontiguous output: stride={actual.stride()}"
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    except AssertionError as exc:
        error = str(exc)
    failed = bool(error)
    if group is not None:
        flag = torch.tensor(int(failed), dtype=torch.int32, device=actual.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        failed = bool(flag.item())
    if failed:
        raise AssertionError(f"{label}: {error or 'a peer rank failed this check'}")


def _check_swap(shape: tuple[int, int, int], dtype: torch.dtype, layout: _Layout) -> dict[str, object]:
    with _isolated_case():
        swap = _a2a_module()._swap_groups
        seed = 1701 + int(os.environ.get("RANK", "0")) * 1009
        x = _random_tensor(shape, dtype, _device(), seed, layout).requires_grad_()
        # opcheck checks registration contracts, not mathematical correctness.
        checks = torch.library.opcheck(swap, (x,), test_utils=_OPCHECK_TESTS, raise_exception=True)
        assert set(checks) == set(_OPCHECK_TESTS)
        assert all(value == "SUCCESS" for value in checks.values()), checks
        y = swap(x)
        _assert_exact(y, x.transpose(0, 1).contiguous(), "swap forward", contiguous=True)
        assert y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(), "swap must not alias its input"
        restored = swap(y)
        _assert_exact(restored, x, "swap involution", contiguous=True)
        del restored
        dy = _random_tensor(tuple(y.shape), dtype, x.device, seed + 1, "noncontiguous")
        expected_dx = dy.transpose(0, 1).contiguous()
        dx = torch.autograd.grad(y, x, dy, retain_graph=True)[0]
        _assert_exact(dx, expected_dx, "swap VJP with offset/noncontiguous input and upstream gradient")
        repeated_dx = torch.autograd.grad(y, x, dy)[0]
        _assert_exact(repeated_dx, dx, "swap repeated backward")
        return {"opcheck": checks, "forward": "exact", "backward": "exact", "repeat_backward": "exact"}


def _check_swap_edges() -> dict[str, object]:
    swap = _a2a_module()._swap_groups
    for shape in ((0, 2, 17), (37, 0, 17), (37, 2, 0)):
        x = torch.empty(shape, device=_device(), requires_grad=True)
        y = swap(x)
        _assert_exact(y, x.transpose(0, 1), "empty swap")
        dx = torch.autograd.grad(y.sum(), x)[0]
        _assert_exact(dx, torch.empty_like(x), "empty swap backward")
    x = torch.randn((1, 2, 17), device=_device()).expand(37, 2, 17).detach().requires_grad_()
    y = swap(x)
    dy = torch.randn_like(y)
    _assert_exact(y, x.transpose(0, 1), "zero-stride swap")
    _assert_exact(torch.autograd.grad(y, x, dy)[0], dy.transpose(0, 1), "zero-stride backward")
    return {"empty": "exact", "zero_stride": "exact"}


@torch.no_grad()
def _expected_forward(x: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Gather original inputs, slice this rank's channels, and concatenate
    source tokens.

    Args:
        x (torch.Tensor): Per-source-rank input [1, local_tokens, channels], possibly strided.
        mesh (DeviceMesh): One-dimensional SP mesh; channels must be divisible by its size.

    Returns:
        torch.Tensor: Independent contiguous [1, local_tokens * P, channels / P],
        with token blocks ordered by source rank, where P is the mesh size.
    """
    _, tokens, channels = x.shape
    size, rank = mesh.size(), mesh.get_local_rank()
    shard = channels // size
    expected = x.new_empty((1, tokens * size, shard))
    for start in range(0, tokens, _GATHER_TOKENS):
        end = min(start + _GATHER_TOKENS, tokens)
        send = x[:, start:end].contiguous()
        gathered = [torch.empty_like(send) for _ in range(size)]
        dist.all_gather(gathered, send, group=mesh.get_group())
        # Explicit slice/cat oracle, not another call to the API under test.
        chunk = torch.cat([source[:, :, rank * shard : (rank + 1) * shard] for source in gathered], dim=1)
        expected.view(size, tokens, shard)[:, start:end].copy_(chunk.view(size, end - start, shard))
        del gathered, send, chunk
    return expected


@torch.no_grad()
def _expected_backward(dy: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Gather destination gradients, slice source tokens, and concatenate
    channels.

    Args:
        dy (torch.Tensor): Per-destination-rank upstream gradient
            [1, local_tokens * P, channels / P], possibly noncontiguous.
        mesh (DeviceMesh): One-dimensional SP mesh of size P in source-token order.

    Returns:
        torch.Tensor: Independent contiguous local-input VJP [1, local_tokens, channels].
    """
    _, global_tokens, shard = dy.shape
    size, rank = mesh.size(), mesh.get_local_rank()
    tokens = global_tokens // size
    expected = dy.new_empty((1, tokens, shard * size))
    # Every rank gathers every global-token chunk in the same order, then slices
    # its own source-token interval. Never all-gather P full large gradients.
    step = _GATHER_TOKENS * size
    for start in range(0, global_tokens, step):
        end = min(start + step, global_tokens)
        send = dy[:, start:end].contiguous()
        gathered = [torch.empty_like(send) for _ in range(size)]
        dist.all_gather(gathered, send, group=mesh.get_group())
        first, last = max(start, rank * tokens), min(end, (rank + 1) * tokens)
        if first < last:
            chunk = torch.cat([dest[:, first - start : last - start, :] for dest in gathered], dim=2)
            expected[:, first - rank * tokens : last - rank * tokens].copy_(chunk)
            del chunk
        del gathered, send
    return expected


def _layout_cases() -> Iterator[_LayoutCase]:
    # Cross all modes with each shape; rotate storage layouts to keep compilation
    # count modest while exercising both offset and arbitrary-stride inputs.
    shapes: list[tuple[int, int, _Layout]] = [
        (tokens, channels, layout)
        for (tokens, layout), channels in product(
            ((37, "contiguous"), (64, "offset"), (65, "noncontiguous")), (128, 256)
        )
    ]
    if os.environ.get("A2A_TEST_LARGE") == "1":
        shapes.append((65536, 8192, "contiguous"))
    for dtype, token_major, compiled, checkpointed in product(_DTYPES, (False, True), (False, True), (False, True)):
        pack = os.environ.get("XTUNER_A2A_TRITON_PACK") == "1"
        for tokens, channels, layout in shapes:
            yield _LayoutCase(tokens, channels, dtype, token_major, compiled, checkpointed, layout, triton_pack=pack)
        yield _LayoutCase(
            37, 256, dtype, token_major, compiled, checkpointed, "contiguous", roundtrip=True, triton_pack=pack
        )


def _check_layout(mesh: DeviceMesh, case: _LayoutCase) -> dict[str, object]:
    with (
        _isolated_case(),
        patch.object(_a2a_module(), "_A2A_TOKEN_MAJOR", case.token_major),
        patch.object(_a2a_module(), "_A2A_TRITON_PACK", case.triton_pack),
    ):
        a2a = _a2a_module().ulysses_all_to_all
        size, rank = mesh.size(), mesh.get_local_rank()
        seed = 20261023 + rank * 1009
        x = _random_tensor((1, case.tokens, case.channels), case.dtype, _device(), seed, case.layout).requires_grad_()

        def redistribute(value: torch.Tensor) -> torch.Tensor:
            """Redistribute local input, optionally returning through the
            legacy 4D API.

            Args:
                value (torch.Tensor): Local [1, tokens, channels], channels divisible by P.

            Returns:
                torch.Tensor: Contiguous [1, tokens * P, channels / P], or local
                [1, tokens, channels / 16, 16] for the head-major inverse roundtrip.
            """
            output = a2a(value, scatter_dim=2, gather_dim=1, mesh=mesh)
            if case.roundtrip:
                output = output.view(1, case.tokens * size, case.channels // size // 16, 16)
                output = a2a(output, scatter_dim=1, gather_dim=2, mesh=mesh)
            if case.checkpointed:
                # Pure permutations need no saved activations. Save a scalar for
                # multiplication so eager non-reentrant checkpoint really revisits
                # the collective, without changing either values or the VJP.
                output = output * output.new_ones(())
            return output

        def forward(value: torch.Tensor) -> torch.Tensor:
            """Run redistribution with optional non-reentrant activation
            checkpointing.

            Args:
                value (torch.Tensor): Local [1, tokens, channels] activations.

            Returns:
                torch.Tensor: redistribute's token-major output or local 4D roundtrip output.
            """
            if case.checkpointed:
                return checkpoint(redistribute, value, use_reentrant=False)
            return redistribute(value)

        runner = torch.compile(forward, fullgraph=True, dynamic=False) if case.compiled else forward
        y = runner(x)
        group = mesh.get_group()
        expected = (
            x.detach().view(1, case.tokens, case.channels // 16, 16) if case.roundtrip else _expected_forward(x, mesh)
        )
        _assert_exact(y, expected, f"{case} forward", group=group, contiguous=True)
        del expected
        output_shape, output_stride = list(y.shape), list(y.stride())
        grad_shape = (1, case.tokens, case.channels) if case.roundtrip else tuple(y.shape)
        dy = _random_tensor(grad_shape, case.dtype, x.device, seed + 1, "noncontiguous")
        if case.roundtrip:
            dy = dy.view(y.shape)
        assert not dy.is_contiguous()
        repeat = case.tokens != 65536
        dx = torch.autograd.grad(y, x, dy, retain_graph=repeat)[0]
        if repeat:
            repeated_dx = torch.autograd.grad(y, x, dy)[0]
            _assert_exact(repeated_dx, dx, f"{case} repeated backward", group=group)
            del repeated_dx
        # Release outputs/graphs before allocating the independent full-size VJP.
        del y, runner
        expected_dx = dy.reshape(x.shape) if case.roundtrip else _expected_backward(dy, mesh)
        _assert_exact(dx, expected_dx, f"{case} backward", group=group)
        return {
            "input_seed": seed,
            "gradient_seed": seed + 1,
            "output_shape": output_shape,
            "output_stride": output_stride,
            "forward": "exact",
            "backward": "exact",
            "repeat_backward": "exact" if repeat else "omitted for large-case memory",
        }


def _result_file() -> Path:
    raw = os.environ.get("A2A_RESULT_DIR")
    if not raw:
        raise ValueError("Set A2A_RESULT_DIR to an absolute directory outside the source checkout")
    directory = Path(raw).expanduser()
    if not directory.is_absolute():
        raise ValueError("A2A_RESULT_DIR must be absolute")
    directory = directory.resolve()
    repository = Path(__file__).resolve().parents[2]
    if directory.is_relative_to(repository):
        raise ValueError("A2A_RESULT_DIR must not be inside the source checkout")
    rank, size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    path = directory / f"ulysses_layout_p{size}_rank{rank}_pid{os.getpid()}.json"
    if path.resolve().is_relative_to(repository):
        raise ValueError("The result file must not resolve inside the source checkout")
    directory.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _main() -> None:
    path = _result_file()
    rank = int(os.environ.get("RANK", "0"))
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank={rank} %(message)s")
    results: list[dict[str, object]] = []
    report: dict[str, object] = {
        "status": "running",
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "atol": 0,
        "rtol": 0,
        "env": {
            name: os.environ.get(name)
            for name in ("XTUNER_A2A_TOKEN_MAJOR", "XTUNER_A2A_TRITON_PACK", "A2A_TEST_LARGE")
        },
        "cases": results,
    }

    def record(metadata: dict[str, object], check: Callable[[], dict[str, object]]) -> None:
        entry = {**metadata, "status": "running"}
        results.append(entry)
        _write_report(path, report)
        started = time.perf_counter()
        try:
            entry["checks"] = check()
            entry["status"] = "passed"
        except Exception:
            entry["status"] = "failed"
            entry["traceback"] = traceback.format_exc()
            raise
        finally:
            entry["elapsed_seconds_including_compile"] = time.perf_counter() - started
            _write_report(path, report)
        _LOG.info("%s", json.dumps(entry))

    _write_report(path, report)
    try:
        with _distributed_mesh() as mesh:
            report["device"] = torch.cuda.get_device_name(_device())
            report["implementation"] = _a2a_module().__file__
            record({"kind": "swap_edges"}, _check_swap_edges)
            for shape, dtype, layout in product(_SWAP_SHAPES, _DTYPES, _LAYOUTS):
                record(
                    {"kind": "swap", "shape": list(shape), "dtype": str(dtype), "layout": layout},
                    lambda: _check_swap(shape, dtype, layout),
                )
            for case in _layout_cases():
                record(
                    {"kind": "roundtrip" if case.roundtrip else "layout", **asdict(case), "dtype": str(case.dtype)},
                    lambda: _check_layout(mesh, case),
                )
            # No rank may report overall success while another is still testing.
            dist.barrier(group=mesh.get_group())
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        _write_report(path, report)
        _LOG.info("Diagnostic JSON: %s", path)


if __name__ == "__main__":
    # Direct file execution must select this checkout even without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _main()
