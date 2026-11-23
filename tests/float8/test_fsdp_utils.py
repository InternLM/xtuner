import os
import unittest.mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

import xtuner.v1.float8.fsdp_utils as fsdp_utils
from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTilewiseFloat8CastTensor,
    precompute_tilewise_float8_scale_for_fsdp,
)


@pytest.fixture(scope="module")
def device_mesh():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
    dist.init_process_group("gloo", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    yield mesh
    dist.destroy_process_group()


def _make_sharded_linear(in_features, out_features, mesh, seed):
    linear = TileWiseFloat8Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        torch.manual_seed(seed)
        linear.weight.copy_(torch.randn_like(linear.weight))
    local = linear.weight.detach()
    assert isinstance(local, WeightWithDynamicTilewiseFloat8CastTensor)
    spec = DTensorSpec(mesh, (Shard(0),), tensor_meta=TensorMeta(local.shape, local.stride(), local.dtype))
    linear.weight = nn.Parameter(DTensor(local, spec, requires_grad=False))
    return linear


def _precomputed_scales(module, mesh, max_stack_bytes):
    precompute_tilewise_float8_scale_for_fsdp(module, reduce_mesh_mapping={}, max_stack_bytes=max_stack_bytes)
    return [m.weight._local_tensor._precomputed_scale for m in module.modules() if isinstance(m, TileWiseFloat8Linear)]


def _assert_chunk_invariant(module_fn, seeds, max_stack_bytes, mesh, expected_calls):
    reference = _precomputed_scales(module_fn(seeds, mesh), mesh, max_stack_bytes=1 << 40)
    with unittest.mock.patch(
        "xtuner.v1.float8.fsdp_utils.tensor_to_per_block_fp8_scales",
        wraps=fsdp_utils.tensor_to_per_block_fp8_scales,
    ) as mock_scales:
        chunked = _precomputed_scales(module_fn(seeds, mesh), mesh, max_stack_bytes=max_stack_bytes)
    assert mock_scales.call_count == expected_calls
    for ref, got in zip(reference, chunked):
        assert ref is not None and got is not None
        assert torch.equal(ref, got)


def test_chunked_stack_matches_full_stack_dout_ge_128(device_mesh):
    # 3 same-shape weights of (256, 512) -> 512 KiB each in float32; the chunk budget
    # forces one weight per chunk, so the scale kernel must run 3 times instead of 1.
    def build(seeds, mesh):
        return nn.ModuleList([_make_sharded_linear(512, 256, mesh, seed) for seed in seeds])

    _assert_chunk_invariant(build, seeds=[0, 1, 2], max_stack_bytes=256 * 512 * 4, mesh=device_mesh, expected_calls=3)


def test_chunked_stack_matches_full_stack_mixed_shape_groups(device_mesh):
    # Two shape groups interleaved in module order: grouping must not confuse chunking.
    def build(seeds, mesh):
        return nn.ModuleList(
            [
                _make_sharded_linear(512, 256, mesh, seeds[0]),
                _make_sharded_linear(256, 128, mesh, seeds[1]),
                _make_sharded_linear(512, 256, mesh, seeds[2]),
            ]
        )

    _assert_chunk_invariant(
        build,
        seeds=[3, 4, 5],
        max_stack_bytes=256 * 512 * 4,  # (256, 512) group chunked per-weight, (128, 256) group untouched
        mesh=device_mesh,
        expected_calls=3,
    )


def test_chunked_scales_match_block_amax_reference(device_mesh):
    (linear,) = nn.ModuleList([_make_sharded_linear(512, 256, device_mesh, seed=6)])
    (scale,) = _precomputed_scales(linear, device_mesh, max_stack_bytes=1 << 40)
    weight = linear.weight._local_tensor.float()
    blocks = weight.view(2, 128, 4, 128).permute(0, 2, 1, 3).reshape(-1, 128 * 128)
    amax = blocks.abs().amax(-1).to(torch.float64)
    expected = (torch.clamp(amax, min=1e-12) / torch.finfo(torch.float8_e4m3fn).max).to(torch.float32).view(2, 4)
    assert torch.equal(scale, expected)
