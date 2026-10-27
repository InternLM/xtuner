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


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
def test_cat_keeps_text_and_padding_offsets(order):
    contexts = [make_context(4, [0, 1, 1, 0]), make_context(3), make_context(2, padding=2)]
    contexts[0].pixel_values = torch.ones(8, 3)
    expected_segments = [torch.tensor([[0, 1, 1, 0]]), torch.zeros(1, 3), torch.zeros(1, 2)]

    packed = SequenceContext.cat([contexts[i] for i in order])

    expected = torch.cat([expected_segments[i] for i in order], dim=1).long()
    assert torch.equal(packed.mm_token_type_ids, expected)
    assert packed.mm_token_type_ids.shape == packed.input_ids.shape == (1, 9)
    assert packed.num_padding == 2
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
    image = make_context(3, [0, 1, 0])
    image.pixel_values = torch.ones(4, 3)
    video = make_context(3, [0, 2, 0])
    video.pixel_values_videos = torch.ones(4, 3)
    packed = SequenceContext.cat([image, make_context(2), video])
    assert packed.mm_token_type_ids.tolist() == [[0, 1, 0, 0, 0, 0, 2, 0]]


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
