import pytest
import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.collator import _finalize_media_packs
from xtuner.v1.datasets.media import MediaPipelineConfig


def context():
    return SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")


def identity(seq_ctx, **overrides):
    config = MediaPipelineConfig(enabled=True, max_inflight_bytes=2**20, max_pending_tasks=1)
    item = {"seq_ctx": seq_ctx, "shifted_labels": torch.tensor([[2, 3, 4, -100]]), **overrides}
    _finalize_media_packs([item], config)
    return seq_ctx.media_pack_id


def test_identical_final_semantics_have_identical_pack_identity():
    assert identity(context()) == identity(context())


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_ids", torch.tensor([[1, 2, 3, 4]])),
        ("mm_token_type_ids", torch.tensor([[0, 1, 1, 0]])),
        ("image_grid_thw", torch.tensor([[1, 2, 2]])),
        ("video_grid_thw", torch.tensor([[1, 2, 2]])),
        ("num_img_tokens", [[4]]),
    ],
)
def test_identity_binds_late_native_semantic_fields(field, value):
    first = context()
    second = context()
    setattr(second, field, value)
    assert identity(first) != identity(second)


def test_identity_distinguishes_tensor_dtype_shape_and_presence():
    first = context()
    second = context()
    second.position_ids = second.position_ids.to(torch.int32)
    assert identity(first) != identity(second)
    second = context()
    second.position_ids = second.position_ids.reshape(4, 1)
    assert identity(first) != identity(second)


def test_identity_distinguishes_eager_payload_layout_without_heavy_pixel_hash():
    first = context()
    second = context()
    second.pixel_values = torch.ones(4, 8)
    assert identity(first) != identity(second)


def test_disabled_pipeline_does_not_add_identity():
    seq_ctx = context()
    items = [{"seq_ctx": seq_ctx, "shifted_labels": torch.tensor([[2, 3, 4, -100]])}]
    assert _finalize_media_packs(items, MediaPipelineConfig()) is items
    assert seq_ctx.media_pack_id is None
