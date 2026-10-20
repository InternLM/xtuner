import pytest
import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.media import MediaDeliveryProof, MediaEntry, MediaPlan, MediaSource, TensorSpec
from xtuner.v1.model.compose.base import BaseComposeModel


def make_context():
    return SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu")


def make_plan():
    spec = TensorSpec("pixel_values", "float32", (4, 8))
    source = MediaSource("fixture://one", "frozen", 16, 16)
    entry = MediaEntry("one:0", (source,), (spec,), 1024)
    return MediaPlan("fixture", "1", "{}", "processor", "manifest", (entry,), (spec,))


def test_planned_copy_to_and_export_keep_same_immutable_plan():
    context = make_context()
    context.media_plan = make_plan()
    context.media_pack_id = "pack-0"
    copied = context.copy().to("cpu")
    assert copied.media_plan is context.media_plan
    assert copied.media_pack_id == "pack-0"
    assert copied.data["media_plan"] is context.media_plan


def test_native_export_has_no_new_media_keys():
    assert not any(name.startswith("media_") for name in make_context().data)


def test_direct_unresolved_forward_is_rejected_before_model():
    context = make_context()
    context.media_plan = make_plan()
    with pytest.raises(ValueError, match="unresolved"):
        BaseComposeModel._require_resolved_media(None, (), {"seq_ctx": context})


def test_consumption_copy_does_not_mutate_original_plan_and_proof_survives_copy():
    original = make_context()
    original.media_plan = make_plan()
    proof = MediaDeliveryProof("step:micro", original.media_plan.fingerprint, original.media_plan.output_specs)
    consumed = original.copy(media_plan=None, media_delivery=proof, pixel_values=torch.zeros(4, 8))
    assert original.media_plan is not None
    assert original.pixel_values is None
    BaseComposeModel._require_resolved_media(None, (), {"seq_ctx": consumed.copy()})
    with pytest.raises(ValueError, match="missing"):
        BaseComposeModel._require_resolved_media(None, (), {"seq_ctx": consumed.copy(pixel_values=None)})


def test_unsupported_context_remapping_fails_instead_of_dropping_plan():
    context = make_context()
    context.media_plan = make_plan()
    with pytest.raises(ValueError, match="retained-pack"):
        SequenceContext.cat([context])
    with pytest.raises(ValueError, match="mapping"):
        context.chunk(1)


def test_language_boundary_drops_raw_payload_only_after_visual_encoding():
    context = make_context()
    plan = make_plan()
    proof = MediaDeliveryProof("step:micro", plan.fingerprint, plan.output_specs)
    original = context.copy(media_delivery=proof, pixel_values=torch.zeros(4, 8))
    encoded = original.copy(input_ids=None, inputs_embeds=torch.zeros(1, 4, 8))
    args, kwargs = BaseComposeModel._language_media_boundary(None, (), {"seq_ctx": encoded})
    actual = kwargs["seq_ctx"]
    assert args == ()
    assert actual is not encoded
    assert actual.pixel_values is None and actual.media_delivery is None
    assert actual.inputs_embeds is encoded.inputs_embeds
    assert actual.position_ids is encoded.position_ids
    assert original.pixel_values is encoded.pixel_values
    with pytest.raises(ValueError, match="encoded"):
        BaseComposeModel._language_media_boundary(None, (), {"seq_ctx": original})


def test_native_language_boundary_keeps_original_context_unchanged():
    context = make_context()
    args, kwargs = BaseComposeModel._language_media_boundary(None, (context,), {})
    assert args[0] is context
    assert kwargs == {}
