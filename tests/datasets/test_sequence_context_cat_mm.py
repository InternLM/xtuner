"""Modality IDs must cover text and padding segments in mixed sequence packs."""

from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext


def make_context(length, token_types=None, padding=0):
    ctx = SequenceContext.from_input_ids((torch.arange(length).view(1, -1),), device="cpu")
    if token_types is not None:
        ctx.mm_token_type_ids = torch.tensor([token_types], dtype=torch.long)
    ctx.num_padding = padding
    return ctx


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2)])
def test_cat_keeps_text_and_padding_offsets(order):
    contexts = [make_context(4, [0, 1, 1, 0]), make_context(3), make_context(2, padding=2)]
    contexts[0].pixel_values = torch.ones(8, 3)
    expected_segments = [torch.tensor([[0, 1, 1, 0]]), torch.zeros(1, 3), torch.zeros(1, 2)]

    packed = SequenceContext.cat([contexts[i] for i in order])

    expected = torch.cat([expected_segments[i] for i in order], dim=1).long()
    assert torch.equal(packed.mm_token_type_ids, expected)
    assert packed.mm_token_type_ids.shape == packed.input_ids.shape == (1, 9)
    assert packed.num_padding == 2
    assert packed.mask.tolist() == [[True] * 7 + [False] * 2]
    assert contexts[1].mm_token_type_ids is None
    assert contexts[2].mm_token_type_ids is None


@pytest.mark.parametrize("rank", [0, 1])
def test_cat_then_split_keeps_odd_length_modality_alignment(rank, monkeypatch):
    # Exercise real pad/split tensor operations with mesh metadata only; no collectives.
    monkeypatch.setattr("xtuner.v1.data_proto.utils.dist.get_rank", lambda group: rank)
    mesh = Mock()
    mesh.size.return_value = 2
    mesh.get_local_rank.return_value = rank
    packed = SequenceContext.cat([make_context(4, [0, 1, 1, 0]), make_context(3), make_context(2, padding=2)])

    shard = packed.split(mesh)

    expected = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])[:, rank * 5 : (rank + 1) * 5]
    assert shard.mm_token_type_ids.shape == shard.input_ids.shape == (1, 5)
    assert torch.equal(shard.mm_token_type_ids, expected)


def test_cat_preserves_image_video_order():
    images = [make_context(3, [0, 1, 0]), make_context(4, [0, 1, 1, 0])]
    videos = [make_context(4, [0, 2, 2, 0]), make_context(5, [0, 2, 2, 2, 0])]
    for index, image in enumerate(images):
        image.pixel_values = torch.full((4 * (index + 1), 3), 10.0 + index)
        image.image_grid_thw = torch.tensor([[1, 2, 2 * (index + 1)]])
    for index, video in enumerate(videos):
        video.pixel_values_videos = torch.full((4 * (index + 2), 3), 20.0 + index)
        video.video_grid_thw = torch.tensor([[index + 2, 2, 2]])
    packed = SequenceContext.cat([images[1], videos[0], make_context(2), images[0], videos[1]])

    assert packed.mm_token_type_ids.tolist() == [[0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 0]]
    torch.testing.assert_close(packed.pixel_values, torch.cat([image.pixel_values for image in images[::-1]]))
    torch.testing.assert_close(packed.image_grid_thw, torch.cat([image.image_grid_thw for image in images[::-1]]))
    torch.testing.assert_close(packed.pixel_values_videos, torch.cat([video.pixel_values_videos for video in videos]))
    torch.testing.assert_close(packed.video_grid_thw, torch.cat([video.video_grid_thw for video in videos]))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cat_preserves_modality_dtype_device_and_inputs(dtype: torch.dtype) -> None:
    image = make_context(3, [0, 1, 0])
    image.mm_token_type_ids = image.mm_token_type_ids.to(dtype)
    original = image.mm_token_type_ids.clone()
    text = make_context(2)
    packed = SequenceContext.cat([text, image])

    assert packed.mm_token_type_ids.dtype == dtype
    assert packed.mm_token_type_ids.device == image.mm_token_type_ids.device
    torch.testing.assert_close(image.mm_token_type_ids, original)
    assert text.mm_token_type_ids is None


def test_cat_without_modality_ids_retains_existing_behavior():
    # Other multimodal models do not use GLM's modality IDs.
    image = make_context(3)
    image.pixel_values = torch.ones(4, 3)
    packed = SequenceContext.cat([image, make_context(2)])
    assert packed.mm_token_type_ids is None
    assert packed.input_ids.shape == (1, 5)


def test_cat_supports_inputs_embeds_without_input_ids():
    image = make_context(3, [0, 1, 0]).copy(input_ids=None, inputs_embeds=torch.zeros(1, 3, 4))
    text = make_context(2).copy(input_ids=None, inputs_embeds=torch.zeros(1, 2, 4))
    packed = SequenceContext.cat([image, text])
    assert packed.input_ids is None
    assert packed.inputs_embeds.shape == (1, 5, 4)
    assert packed.mm_token_type_ids.tolist() == [[0, 1, 0, 0, 0]]


def test_cat_rejects_mismatched_modality_length():
    with pytest.raises(ValueError, match="does not match token shape"):
        SequenceContext.cat([make_context(4, [0, 1, 0])])


@pytest.mark.parametrize("media_field", ["pixel_values", "pixel_values_videos"])
def test_cat_does_not_silently_fill_missing_visual_modality_ids(media_field):
    visual = make_context(3)
    setattr(visual, media_field, torch.ones(4, 3))
    with pytest.raises(ValueError, match="visual sequence without mm_token_type_ids"):
        SequenceContext.cat([make_context(2, [0, 0]), visual])
