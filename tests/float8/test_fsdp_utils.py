import os
import unittest.mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

import xtuner.v1.float8.fsdp_utils as fsdp_utils
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear
from xtuner.v1.float8.fsdp_utils import (
    WeightWithDynamicTilewiseFloat8CastTensor,
    _rl_bf16_roundtrip_peak_bytes,
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
    reference = _precomputed_scales(module_fn(seeds, mesh), mesh, max_stack_bytes=None)  # 不分 chunk 的原路径
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


def test_max_stack_bytes_passthrough_config_to_precompute():
    # 透传链:Float8Config.max_stack_bytes -> Float8Handler -> precompute 的第 4 个位置参数;
    # 覆盖默认值(1 GiB)、显式预算与显式 None(关 chunking)三种取值
    cases = [
        ({}, 1 << 30),
        ({"max_stack_bytes": 1 << 28}, 1 << 28),
        ({"max_stack_bytes": None}, None),
    ]
    for config_kwargs, expected in cases:
        handler = Float8Config(scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE, **config_kwargs).build()
        with unittest.mock.patch(
            "xtuner.v1.float8.float8_handler.precompute_tilewise_float8_scale_for_fsdp"
        ) as mock_precompute:
            handler.precompute_float8_dynamic_scale_for_fsdp(nn.Module())
        assert mock_precompute.call_count == 1
        assert mock_precompute.call_args.args[3] == expected


def test_max_stack_bytes_none_disables_chunking(device_mesh):
    # max_stack_bytes=None 走原路径:3 个同 shape 权重一次 stack,kernel 只调 1 次,
    # 且结果与逐权重分 chunk bitwise 一致
    def build(seeds, mesh):
        return nn.ModuleList([_make_sharded_linear(512, 256, mesh, seed) for seed in seeds])

    with unittest.mock.patch(
        "xtuner.v1.float8.fsdp_utils.tensor_to_per_block_fp8_scales",
        wraps=fsdp_utils.tensor_to_per_block_fp8_scales,
    ) as mock_scales:
        reference = _precomputed_scales(build([0, 1, 2], device_mesh), device_mesh, max_stack_bytes=None)
    assert mock_scales.call_count == 1
    chunked = _precomputed_scales(build([0, 1, 2], device_mesh), device_mesh, max_stack_bytes=256 * 512 * 4)
    for ref, got in zip(reference, chunked):
        assert ref is not None and got is not None
        assert torch.equal(ref, got)


def test_chunked_scales_match_block_amax_reference(device_mesh):
    (linear,) = nn.ModuleList([_make_sharded_linear(512, 256, device_mesh, seed=6)])
    (scale,) = _precomputed_scales(linear, device_mesh, max_stack_bytes=None)
    weight = linear.weight._local_tensor.float()
    blocks = weight.view(2, 128, 4, 128).permute(0, 2, 1, 3).reshape(-1, 128 * 128)
    amax = blocks.abs().amax(-1).to(torch.float64)
    expected = (torch.clamp(amax, min=1e-12) / torch.finfo(torch.float8_e4m3fn).max).to(torch.float32).view(2, 4)
    assert torch.equal(scale, expected)


def _chunked_equiv_worker(rank, world_size, init_path, dout, use_divided_64):
    # 每个 rank 独立构造相同 seed 的 padded local 权重(模拟真实 FSDP 场景下各 rank 的本地分片),
    # 先以不分 chunk 的原路径(max_stack_bytes=None)算参照,再以单权重预算(3 chunks)算被测结果。
    # 若各 rank 推导出的 chunk 边界不一致,all_reduce 次数会对不上,进程会直接挂死,
    # 因此跑通本身即验证跨 rank chunk 边界一致性。
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_path}")
    mesh = init_device_mesh("cpu", (world_size,))
    din = 256
    weights_kwargs = {"reduce_mesh_mapping": {}, "reduce_mesh_devided_64": None}
    if use_divided_64:
        weights_kwargs["reduce_mesh_devided_64"] = mesh
    else:
        weights_kwargs["reduce_mesh_mapping"] = {(dout, din): mesh}

    def scales(max_stack_bytes):
        module = nn.ModuleList([_make_sharded_linear(din, dout, mesh, rank * 100 + seed) for seed in (0, 1, 2)])
        precompute_tilewise_float8_scale_for_fsdp(module, max_stack_bytes=max_stack_bytes, **weights_kwargs)
        return [
            m.weight._local_tensor._precomputed_scale for m in module.modules() if isinstance(m, TileWiseFloat8Linear)
        ]

    reference = scales(None)  # 不分 chunk
    chunked = scales(dout * din * 4)  # 单权重预算 -> 逐权重 3 个 chunk
    for i, (ref, got) in enumerate(zip(reference, chunked)):
        assert ref is not None and got is not None
        assert torch.equal(ref, got), f"rank{rank} weight{i}: chunked scales differ from single-shot"
    dist.destroy_process_group()


def test_chunked_stack_matches_full_stack_reduce_path_multi_rank(tmp_path):
    # dout=64 < 128:amax 需要跨 reduce_mesh all_reduce 的路径,2 个 gloo rank 上验证分 chunk 与单次 stack bitwise 一致
    mp.spawn(_chunked_equiv_worker, args=(2, str(tmp_path / "gloo_init"), 64, False), nprocs=2, join=True)


def test_chunked_stack_matches_full_stack_divided_64_path_multi_rank(tmp_path):
    # dout=192(=128+64):divided_64 路径(128-block 横跨两 rank),2 个 gloo rank 上验证分 chunk 与单次 stack bitwise 一致
    mp.spawn(_chunked_equiv_worker, args=(2, str(tmp_path / "gloo_init"), 192, True), nprocs=2, join=True)


def test_rl_bf16_roundtrip_peak_bytes_matches_torch_semantics():
    # 钉住峰值系数依赖的 torch 行为:同 dtype .bfloat16() 零拷贝返回自身,.float() 总是拷贝;
    # 若 _rl_bf16_roundtrip 的实现改动,需同步更新 _rl_bf16_roundtrip_peak_bytes
    bf16_probe = torch.ones(1, dtype=torch.bfloat16)
    fp32_probe = torch.ones(1, dtype=torch.float32)
    fp16_probe = torch.ones(1, dtype=torch.float16)
    assert bf16_probe.bfloat16() is bf16_probe
    assert fp32_probe.bfloat16() is not fp32_probe
    assert fp16_probe.bfloat16() is not fp16_probe
    assert _rl_bf16_roundtrip_peak_bytes(torch.bfloat16) == 6  # stack 2 + bf16 零拷贝 0 + fp32 副本 4
    assert _rl_bf16_roundtrip_peak_bytes(torch.float32) == 10  # stack 4 + bf16 副本 2 + fp32 副本 4
    assert _rl_bf16_roundtrip_peak_bytes(torch.float16) == 8  # stack 2 + bf16 副本 2 + fp32 副本 4


def test_chunked_stack_matches_full_stack_with_rl_bf16_env(device_mesh, monkeypatch):
    # XTUNER_RL_FP8_QUANTIZE_IN_BF16=1 时预算按 10 B/elem 收账(fp32 权重),
    # 验证该路径下分 chunk 与单次 stack 仍 bitwise 一致
    monkeypatch.setenv("XTUNER_RL_FP8_QUANTIZE_IN_BF16", "1")

    def build(seeds, mesh):
        return nn.ModuleList([_make_sharded_linear(512, 256, mesh, seed) for seed in seeds])

    bytes_per_weight = 256 * 512 * _rl_bf16_roundtrip_peak_bytes(torch.float32)  # 单权重峰值预算 -> 逐权重 3 个 chunk
    _assert_chunk_invariant(
        build, seeds=[0, 1, 2], max_stack_bytes=bytes_per_weight, mesh=device_mesh, expected_calls=3
    )
