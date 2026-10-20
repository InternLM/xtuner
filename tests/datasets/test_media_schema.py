import copy
import json
import pickle

import pytest

from xtuner.v1.datasets.media import (
    MediaDeliveryProof,
    MediaEntry,
    MediaPipelineConfig,
    MediaPlan,
    MediaSource,
    TensorSpec,
)
from xtuner.v1.datasets.media.provider import create_provider, register_provider


def make_plan():
    source = MediaSource("fixture://one", "manifest-sha", 32, 48)
    spec = TensorSpec("fixture", "float32", (6, 4))
    entry = MediaEntry("record-1:image-0", (source,), (spec,), 4096)
    return MediaPlan("fixture", "1", "{}", "processor-sha", "manifest-sha", (entry,), (spec,), ("record-1",))


def test_plan_serialization_has_no_runtime_payload():
    plan = make_plan()
    assert MediaPlan.from_json(plan.to_json()) == plan
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert copy.deepcopy(plan).fingerprint == plan.fingerprint
    assert plan.output_bytes == 96
    assert json.loads(plan.to_json())["entries"][0]["sources"][0]["uri"] == "fixture://one"


def test_source_revision_is_part_of_identity():
    plan = make_plan()
    changed = plan.to_dict()
    changed["entries"][0]["sources"][0]["revision"] = "different-version"
    assert MediaPlan.from_dict(changed).fingerprint != plan.fingerprint


def test_missing_source_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        MediaSource("fixture://one", "", 32, 48)


@pytest.mark.parametrize("shape", [(True, 3), (-1, 3), (2**62, 8)])
def test_untrusted_shapes_rejected_before_allocation(shape):
    with pytest.raises(ValueError):
        TensorSpec("fixture", "float32", shape)


def test_plan_rejects_payload_metadata():
    with pytest.raises((TypeError, ValueError)):
        MediaSource("fixture://one", "revision", 32, 48, metadata_json={"object": object()})


def test_disabled_is_native_and_explicit_requests_not_ignored():
    assert not MediaPipelineConfig().enabled
    with pytest.raises(ValueError, match="enabled"):
        MediaPipelineConfig(per_image=True)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "decode_threads": 2},
        {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "prefetch_batches": 1},
        {"max_inflight_bytes": 1000, "max_pending_tasks": 1, "materialization": "lazy"},
    ],
)
def test_invalid_enabled_combinations_rejected(options):
    with pytest.raises(ValueError):
        MediaPipelineConfig(enabled=True, **options)


def test_mock_provider_registration_needs_no_model_dispatch():
    class MockProvider:
        def __init__(self, settings):
            self.settings = settings

    register_provider("schema-fixture-v1", MockProvider)
    assert create_provider("schema-fixture-v1", '{"variant":1}').settings == '{"variant":1}'
    with pytest.raises(ValueError, match="duplicate"):
        register_provider("schema-fixture-v1", MockProvider)


def test_delivery_proof_rejects_dropped_media_field():
    proof = MediaDeliveryProof("run:step:micro", make_plan().fingerprint, make_plan().output_specs)
    with pytest.raises(ValueError, match="missing"):
        proof.validate_context(type("Context", (), {"fixture": None})())
