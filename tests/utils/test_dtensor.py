"""xtuner/v1/utils/dtensor.py 的 DTensor 辅助函数。

TestMaterializeFull
    test_plain_tensor_passes_through                          普通张量原样返回
    test_replicated_dtensor_is_unwrapped_to_the_whole_tensor  Replicate 解包成完整张量
    test_sharded_dtensor_is_rejected                          Shard 直接报错而非返回分片
"""

import io
import zipfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.utils.dtensor import dtensor_checkpoint_view, materialize_full


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


class TestDTensorCheckpointView:
    @pytest.mark.parametrize("shape,stride", [((2, 3), (3, 1)), ((3, 2), (1, 3)), ((0, 3), (3, 1)), ((), ())])
    def test_offset_views_roundtrip_without_copying(
        self, fake_mesh: DeviceMesh, shape: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        slab = torch.arange(20.0)
        local = slab.as_strided(shape, stride, 2)
        spec = DTensorSpec(fake_mesh, (Replicate(),), tensor_meta=TensorMeta(local.shape, stride, local.dtype))
        tensor = DTensor(local, spec, requires_grad=False)
        checkpoint = dtensor_checkpoint_view(tensor)
        assert type(checkpoint) is DTensor
        assert checkpoint._spec is tensor._spec
        assert checkpoint._local_tensor is local
        assert checkpoint.storage_offset() == 0
        assert tensor.storage_offset() == 2
        assert local.storage_offset() == 2
        assert checkpoint.stride() == stride
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        torch.testing.assert_close(restored.to_local(), local)
        assert restored.shape == tensor.shape
        assert restored.stride() == tensor.stride()
        assert restored.placements == tensor.placements
        assert restored.device_mesh == tensor.device_mesh
        assert restored._local_tensor.storage_offset() == 2
        torch.save(restored, io.BytesIO())

    def test_shared_slab_deduplicates_and_stays_aliased(self, fake_mesh: DeviceMesh) -> None:
        slab = torch.arange(20.0)
        spec = DTensorSpec(fake_mesh, (Replicate(),), tensor_meta=TensorMeta(torch.Size([3]), (1,), slab.dtype))
        tensors = [DTensor(slab[offset : offset + 3], spec, requires_grad=False) for offset in (2, 5)]
        checkpoints = [dtensor_checkpoint_view(tensor) for tensor in tensors]
        slab[2] = 99
        assert checkpoints[0].to_local()[0] == 99
        stream = io.BytesIO()
        torch.save(checkpoints, stream)
        with zipfile.ZipFile(stream) as archive:
            # DeviceMesh also serializes its rank tensor; the slab itself is written only once.
            storages = [entry for entry in archive.infolist() if entry.filename.startswith("archive/data/")]
            assert sum(entry.file_size == slab.untyped_storage().nbytes() for entry in storages) == 1
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        assert restored[0]._local_tensor.untyped_storage() is restored[1]._local_tensor.untyped_storage()
        for actual, expected in zip(restored, tensors):
            torch.testing.assert_close(actual.to_local(), expected.to_local())

    def test_unaffected_tensors_are_returned_unchanged(self, fake_mesh: DeviceMesh) -> None:
        plain = torch.arange(4.0)
        assert dtensor_checkpoint_view(plain) is plain
        dtensor = distribute_tensor(plain, fake_mesh, [Replicate()])
        assert dtensor_checkpoint_view(dtensor) is dtensor

    def test_requires_detached_input(self, fake_mesh: DeviceMesh) -> None:
        local = torch.arange(6.0)[2:4]
        spec = DTensorSpec(fake_mesh, (Replicate(),), tensor_meta=TensorMeta(local.shape, (1,), local.dtype))
        tensor = DTensor(local, spec, requires_grad=True)
        with pytest.raises(ValueError, match="detached"):
            dtensor_checkpoint_view(tensor)
        detached = tensor.detach()
        torch.save(dtensor_checkpoint_view(detached), io.BytesIO())
        assert tensor.requires_grad and tensor.storage_offset() == 2
