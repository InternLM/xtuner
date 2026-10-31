"""xtuner/v1/utils/dtensor.py 的 DTensor 辅助函数。

TestMaterializeFull
    test_plain_tensor_passes_through                          普通张量原样返回
    test_replicated_dtensor_is_unwrapped_to_the_whole_tensor  Replicate 解包成完整张量
    test_sharded_dtensor_is_rejected                          Shard 直接报错而非返回分片
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.utils.dtensor import materialize_full


@pytest.fixture
def fake_mesh():
    """A 2-rank CPU mesh backed by the fake process group: enough to build DTensors with real
    placements in-process, which is all these placement checks need."""
    dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())
    try:
        yield init_device_mesh("cpu", (2,))
    finally:
        dist.destroy_process_group()


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
