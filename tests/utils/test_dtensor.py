"""xtuner/v1/utils/dtensor.py 的 DTensor 辅助函数。

TestShardedPlacement
    test_shard_and_strided_shard_are_sharded                  Shard / _StridedShard 都算分片
    test_replicate_and_partial_are_not_sharded                Replicate / Partial 不算分片

TestMaterializeFull
    test_plain_tensor_passes_through                          普通张量原样返回
    test_replicated_dtensor_is_unwrapped_to_the_whole_tensor  Replicate 解包成完整张量
    test_sharded_dtensor_is_rejected                          Shard 直接报错而非返回分片
    test_strided_sharded_dtensor_is_rejected                  _StridedShard 同样报错

TestIsEvenlyDistributed
    test_strided_shard_is_counted                             _StridedShard 参与整除判断
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Placement, _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.utils.dtensor import is_evenly_distributed, is_sharded_placement, materialize_full


def _strided_shard() -> Placement:
    """Build a ``_StridedShard`` with a signature valid on both old and new PyTorch."""
    try:
        return _StridedShard(0, split_factor=2)
    except TypeError:
        return _StridedShard(0)


def _strided_dtensor(mesh: DeviceMesh, local: torch.Tensor, shape: tuple[int, ...]) -> DTensor:
    return DTensor.from_local(local, mesh, (_strided_shard(),), run_check=False, shape=shape, stride=(shape[1], 1))


@pytest.fixture
def fake_mesh():
    """A 2-rank CPU mesh backed by the fake process group: enough to build DTensors with real
    placements in-process, which is all these placement checks need."""
    dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
    try:
        yield init_device_mesh("cpu", (2,))
    finally:
        dist.destroy_process_group()


class TestShardedPlacement:
    def test_shard_and_strided_shard_are_sharded(self):
        # _StridedShard 是否继承 Shard 随 torch 版本变化，两者都必须判定为分片。
        assert is_sharded_placement(Shard(0))
        assert is_sharded_placement(_strided_shard())

    def test_replicate_and_partial_are_not_sharded(self):
        assert not is_sharded_placement(Replicate())
        assert not is_sharded_placement(Partial())


class TestMaterializeFull:
    def test_plain_tensor_passes_through(self):
        # 非 DTensor 原样返回，不做多余拷贝。
        t = torch.randn(4, 8)
        assert materialize_full(t) is t

    def test_replicated_dtensor_is_unwrapped_to_the_whole_tensor(self, fake_mesh):
        # Replicate 的 DTensor 解包成完整张量。
        full = torch.randn(4, 8)
        out = materialize_full(distribute_tensor(full, fake_mesh, [Replicate()]))
        assert not isinstance(out, torch.distributed.tensor.DTensor)
        assert out.shape == full.shape

    def test_sharded_dtensor_is_rejected(self, fake_mesh):
        """A local shard here is not a smaller-but-valid tensor: every call site reshapes by
        head or channel straight afterwards, so the wrong slice would be silently mis-grouped
        instead of raising a shape error."""
        # Shard 必须报错：返回分片会被后续 view/chunk 静默按错误分组使用。
        sharded = distribute_tensor(torch.randn(4, 8), fake_mesh, [Shard(0)])
        with pytest.raises(RuntimeError, match="Shard"):
            materialize_full(sharded, name="self_attn.kv_b_proj.weight")

    def test_strided_sharded_dtensor_is_rejected(self, fake_mesh):
        # FSDP2 的 _StridedShard 同样持有局部切片，不能被当成完整张量放行。
        sharded = _strided_dtensor(fake_mesh, torch.randn(2, 8), (4, 8))
        with pytest.raises(RuntimeError, match="still sharded"):
            materialize_full(sharded, name="self_attn.kv_b_proj.weight")


class TestIsEvenlyDistributed:
    def test_strided_shard_is_counted(self, fake_mesh):
        # 3 不能被 mesh 的 2 个 rank 整除；若 _StridedShard 被漏判，会错误地返回 True。
        assert not is_evenly_distributed(_strided_dtensor(fake_mesh, torch.randn(2, 8), (3, 8)))
        assert is_evenly_distributed(_strided_dtensor(fake_mesh, torch.randn(2, 8), (4, 8)))
