import copy
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from xtuner.v1.datasets.media.config import MediaPipelineConfig
from xtuner.v1.datasets.media.provider import create_provider
from xtuner.v1.datasets.media.providers import shared
from xtuner.v1.datasets.media.schema import MaterializedMedia, MediaPlan, canonical_json
from xtuner.v1.datasets.mllm_tokenize_fn import glm53_vl_tokenize_fn as glm_tokenize
from xtuner.v1.datasets.mllm_tokenize_fn import qwen3_vl_tokenize_fn as qwen_tokenize
from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import load_image
from xtuner.v1.datasets.utils import apply_exif_orientation


def _pipeline(manifest=None, **overrides):
    return MediaPipelineConfig(
        enabled=True,
        max_inflight_bytes=256 * 1024**2,
        max_pending_tasks=4,
        source_manifest=None if manifest is None else str(manifest),
        **overrides,
    )


@pytest.fixture(params=("glm", "qwen3", "qwen3.5"))
def image_fixture(request, tmp_path):
    family = request.param
    if family == "glm":
        name = "glm_image"
        processor = Glm5NextImageProcessor(min_image_tokens=4, max_image_tokens=32)
    else:
        name = "qwen_image"
        processor = Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 4096, "longest_edge": 16384})
    identity = shared.image_processor_identity(name, processor)
    profile_key = shared.image_profile_fingerprint(identity)
    certificates = []
    native_outputs = []
    for image_index, (width, height, color) in enumerate(((64, 96, (120, 30, 200)), (96, 64, (50, 170, 20)))):
        path = tmp_path / f"image-{image_index}.png"
        image = Image.new("RGB", (width, height), color)
        exif = Image.Exif()
        exif[274] = 6 if image_index == 0 else 1
        image.save(path, exif=exif)
        image.close()
        decoded = apply_exif_orientation(load_image(str(path)))
        processed = processor(images=[decoded], return_tensors="pt")
        source_bytes = path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        certificates.append(
            {
                "uri": str(path),
                "revision": f"sha256:{digest}",
                "content_sha256": digest,
                "source_bytes": len(source_bytes),
                "width": decoded.width,
                "height": decoded.height,
                "orientation": exif[274],
                "decode_verified": True,
                "eligible": True,
                "profiles": {
                    profile_key: {
                        "image_grid_thw": processed["image_grid_thw"][0].tolist(),
                        "pixel_values": {
                            "dtype": str(processed["pixel_values"].dtype).removeprefix("torch."),
                            "shape": list(processed["pixel_values"].shape),
                        },
                        "estimated_peak_bytes": 32 * 1024**2,
                    }
                },
            }
        )
        native_outputs.append(processed)
        decoded.close()
    manifest = tmp_path / "certificates.json"
    manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    config_json = shared.make_image_provider_config(name, processor, str(manifest))
    provider = create_provider(name, config_json)
    sources = provider.sources_for_uris([certificate["uri"] for certificate in certificates])
    return SimpleNamespace(
        family=family,
        name=name,
        processor=processor,
        provider=provider,
        sources=sources,
        manifest=manifest,
        certificates=certificates,
        native_outputs=native_outputs,
        profile_key=profile_key,
        config_json=config_json,
    )


def _plan(fixture, per_image=True, sources=None, occurrence_id="record"):
    return fixture.provider.describe(
        fixture.sources if sources is None else sources,
        occurrence_id=occurrence_id,
        per_image=per_image,
        source_manifest_fingerprint=fixture.provider.manifest_fingerprint,
    )


def test_describe_is_decode_free_and_keeps_exact_effective_grid(image_fixture):
    fixture = image_fixture
    with patch.object(Image, "open", side_effect=AssertionError("describe must not decode")):
        plan = _plan(fixture)
    assert fixture.provider._processor is None
    assert fixture.sources[0].orientation == 6
    assert (fixture.sources[0].width, fixture.sources[0].height) == (96, 64)
    assert [json.loads(entry.metadata_json)["image_grid_thw"][0] for entry in plan.entries] == [
        output["image_grid_thw"][0].tolist() for output in fixture.native_outputs
    ]
    assert MediaPlan.from_json(plan.to_json()) == plan
    assert plan.processor_fingerprint == hashlib.sha256(plan.processor_config_json.encode()).hexdigest()
    assert plan.source_manifest_fingerprint == hashlib.sha256(fixture.manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("per_image", (False, True))
def test_real_native_processor_matches_provider_grouped_and_per_image(image_fixture, per_image):
    fixture = image_fixture
    plan = _plan(fixture, per_image)
    outputs = [fixture.provider.process(entry) for entry in plan.entries]
    actual = fixture.provider.assemble(plan, outputs)
    if fixture.name == "glm_image":
        expected = {
            field: torch.cat([output[field] for output in fixture.native_outputs])
            for field in ("pixel_values", "image_grid_thw")
        }
        assert len(plan.entries) == 2
    else:
        images = [apply_exif_orientation(load_image(source.uri)) for source in fixture.sources]
        try:
            expected = fixture.processor.preprocess(images, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        assert len(plan.entries) == (2 if per_image else 1)
    for field in ("pixel_values", "image_grid_thw"):
        assert actual.tensors[field].dtype == expected[field].dtype
        torch.testing.assert_close(actual.tensors[field], expected[field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="occurrence order"):
        fixture.provider.assemble(
            _plan(fixture), list(reversed([fixture.provider.process(entry) for entry in _plan(fixture).entries]))
        )


def test_merge_keeps_occurrences_and_rejects_identity_change(image_fixture):
    fixture = image_fixture
    plans = [
        _plan(fixture, sources=[source], occurrence_id=f"record-{index}")
        for index, source in enumerate(fixture.sources)
    ]
    merged = fixture.provider.merge_plans(plans)
    assert merged.entries == tuple(entry for plan in plans for entry in plan.entries)
    assert merged.sample_ids == ("record-0", "record-1")
    assert merged.output_specs == _plan(fixture).output_specs
    with pytest.raises(ValueError, match="identities"):
        fixture.provider.merge_plans([plans[0], replace(plans[1], source_manifest_fingerprint="different")])
    repeated = fixture.provider.merge_plans([plans[0], plans[0]])
    assert len(repeated.entries) == 2
    assert repeated.output_specs[0].shape[0] == 2 * plans[0].output_specs[0].shape[0]


def test_source_content_is_checked_before_decode(image_fixture):
    fixture = image_fixture
    plan = _plan(fixture)
    source_path = fixture.manifest.parent / "image-0.png"
    source_bytes = source_path.read_bytes()
    source_path.write_bytes(bytes([source_bytes[0] ^ 1]) + source_bytes[1:])
    with patch.object(Image, "open", side_effect=AssertionError("changed source must fail before decode")):
        with pytest.raises(ValueError, match="bytes differ"):
            fixture.provider.process(plan.entries[0])


def test_changed_manifest_and_weak_descriptors_are_rejected(image_fixture):
    fixture = image_fixture
    with pytest.raises(ValueError, match="descriptor differs"):
        fixture.provider.describe(
            [replace(fixture.sources[0], metadata_json="{}")],
            occurrence_id="bad",
            per_image=True,
            source_manifest_fingerprint=fixture.provider.manifest_fingerprint,
        )
    fixture.manifest.write_text(fixture.manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest changed"):
        create_provider(fixture.name, fixture.config_json)


@pytest.mark.parametrize("mutation", ("decode", "profile", "exif", "spatial_grid", "revision"))
def test_certificate_counterexamples_are_not_silently_accepted(image_fixture, mutation):
    fixture = image_fixture
    certificates = copy.deepcopy(fixture.certificates)
    certificate = certificates[0]
    if mutation == "decode":
        certificate["decode_verified"] = False
    elif mutation == "profile":
        certificate["profiles"] = {}
    elif mutation == "exif":
        certificate["orientation"] = 1
    elif mutation == "revision":
        certificate["revision"] = certificate["uri"]
    else:
        grid = certificate["profiles"][fixture.profile_key]["image_grid_thw"]
        assert grid[1] != grid[2]
        grid[1], grid[2] = grid[2], grid[1]
    fixture.manifest.write_text(canonical_json({"schema_version": 1, "sources": certificates}))
    provider = create_provider(
        fixture.name, shared.make_image_provider_config(fixture.name, fixture.processor, str(fixture.manifest))
    )
    if mutation in ("decode", "profile", "revision"):
        with pytest.raises(ValueError):
            provider.sources_for_uris([certificate["uri"]])
    else:
        sources = provider.sources_for_uris([certificate["uri"]])
        plan = provider.describe(
            sources, occurrence_id="bad", per_image=True, source_manifest_fingerprint=provider.manifest_fingerprint
        )
        with pytest.raises(ValueError, match="orientation|grid"):
            provider.process(plan.entries[0])


def _tokenize_instance(fixture):
    if fixture.family == "glm":
        tokenize_fn = object.__new__(glm_tokenize.Glm53VLTokenizeFunction)
    else:
        tokenize_fn = object.__new__(qwen_tokenize.Qwen3VLTokenizeFunction)
    tokenize_fn.data_name = "synthetic"
    tokenize_fn._image_path = [source.uri for source in fixture.sources]
    tokenize_fn.image_processor = fixture.processor
    tokenize_fn._media_oss_loader_config = None
    tokenize_fn.oss_loader = None
    tokenize_fn.media_pipeline = None
    tokenize_fn.max_length = None
    return tokenize_fn


def test_eager_uses_common_materializer_and_lazy_never_reads_pixels(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    record = {"id": "record", "messages": []}
    assert shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name) is None
    tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
    with patch.object(shared.CertifiedImageProvider, "_read_source", side_effect=AssertionError("lazy decode")):
        pixels, grids, plan = shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name)
    assert pixels is None
    assert plan is not None
    assert grids.tolist() == [output["image_grid_thw"][0].tolist() for output in fixture.native_outputs]
    tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="eager", per_image=True)
    materialized = fixture.provider.assemble(plan, [fixture.provider.process(entry) for entry in plan.entries])
    with patch.object(shared, "_materialize_plans", return_value=[materialized]) as common:
        pixels, eager_grid, eager_plan = shared.prepare_image_inputs(tokenize_fn, record, "", fixture.name)
    assert common.call_count == 1
    assert pixels is materialized.tensors["pixel_values"]
    assert torch.equal(eager_grid, grids)
    assert eager_plan is None


def test_missing_certificate_and_video_choose_policy_before_collation(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    tokenize_fn.media_pipeline = _pipeline()
    with pytest.raises(shared.UnsupportedMediaSource, match="source_manifest"):
        shared.prepare_image_inputs(tokenize_fn, {}, "", fixture.name)
    with pytest.raises(shared.UnsupportedMediaSource, match="video"):
        shared.check_video_policy(tokenize_fn)
    tokenize_fn.media_pipeline = _pipeline(unsupported_policy="eager")
    assert shared.prepare_image_inputs(tokenize_fn, {}, "", fixture.name) is None
    shared.check_video_policy(tokenize_fn)


def test_mixed_pack_resolves_only_before_collation_without_mutating_items(image_fixture):
    fixture = image_fixture
    plan = _plan(fixture)
    planned = {"input_ids": [1], "media_plan": plan, "image_grid_thw": torch.tensor([[1, 4, 4]])}
    native = {"input_ids": [2], "pixel_values_videos": torch.zeros(1, 1)}
    with pytest.raises(shared.UnsupportedMediaSource, match="before collation"):
        shared.resolve_mixed_media_items([planned, native], _pipeline(fixture.manifest))
    materialized = MaterializedMedia({"pixel_values": torch.zeros(1, 1), "image_grid_thw": torch.ones(1, 3)})
    with patch.object(shared, "_materialize_plans", return_value=[materialized]) as common:
        resolved = shared.resolve_mixed_media_items(
            [planned, native], _pipeline(fixture.manifest, unsupported_policy="eager")
        )
    common.assert_called_once()
    assert "media_plan" in planned and "pixel_values" not in planned
    assert "media_plan" not in resolved[0]
    assert resolved[0]["pixel_values"] is materialized.tensors["pixel_values"]
    assert resolved[1]["pixel_values_videos"] is native["pixel_values_videos"]


def test_discarded_video_does_not_trigger_fallback(image_fixture):
    fixture = image_fixture
    planned = {"input_ids": [1], "media_plan": _plan(fixture)}
    text = {"input_ids": [2], "labels": [-100], "num_imgs": [0]}
    discarded_video = {"input_ids": [3], "pixel_values_videos": torch.zeros(1, 1)}
    candidates = [planned, text, discarded_video]
    retained = candidates[:2]
    with patch.object(shared, "_materialize_plans", side_effect=AssertionError("discarded video fallback")):
        resolved = shared.resolve_mixed_media_items(retained, _pipeline(fixture.manifest, unsupported_policy="eager"))
    assert resolved == retained
    assert resolved[0]["media_plan"] is planned["media_plan"]


def test_family_native_and_lazy_token_metadata_match(image_fixture):
    fixture = image_fixture
    tokenize_fn = _tokenize_instance(fixture)
    counts = [
        output["image_grid_thw"].prod().item() // fixture.processor.merge_size**2 for output in fixture.native_outputs
    ]
    record = {"id": "record", "messages": [{"role": "user", "content": "synthetic"}]}
    if fixture.family == "glm":
        tokenize_fn.merge_unit = fixture.processor.merge_size**2
        tokenize_fn.processor = SimpleNamespace(image_token_id=7, replace_image_token=Mock(return_value="visual"))
        tokenize_fn._render_base = Mock(return_value=("<|image|>" * 2, [False] * (len("<|image|>") * 2)))
        input_ids = [10] + [7] * sum(counts) + [11]
        tokenize_fn._finalize = Mock(return_value=(input_ids, [-100] * len(input_ids), [0] * len(input_ids)))
        native = tokenize_fn.multi_modal_get_item(record)
        tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
        lazy = tokenize_fn.multi_modal_get_item(record)
        assert torch.equal(native["mm_token_type_ids"], lazy["mm_token_type_ids"])
    else:
        is_qwen35 = fixture.family == "qwen3.5"
        image_id, start_id, end_id = (248056, 248053, 248054) if is_qwen35 else (151655, 151652, 151653)
        tokenize_fn.img_context_token_id = image_id
        tokenize_fn.video_context_token_id = 248057 if is_qwen35 else 151656
        tokenize_fn.img_start_token_id = start_id
        tokenize_fn.img_end_token_id = end_id
        tokenize_fn.merge_length = fixture.processor.merge_size**2
        tokenize_fn.chat_template_name = "qwen3.5-vl" if is_qwen35 else "qwen3-vl"
        tokenize_fn.chat_template = Mock()
        tokenize_fn.tokenizer = Mock()
        tokenize_fn.enable_3d_rope = True
        tokenize_fn.add_vision_id = False
        tokenize_fn.add_generation_prompt = False
        tokenize_fn.enable_thinking = None

        def tokenize(*args, **kwargs):
            input_ids = [10]
            for count in counts:
                input_ids.extend([start_id] + [image_id] * (1 if is_qwen35 else count) + [end_id])
            input_ids.append(11)
            return {"input_ids": input_ids, "labels": [-100] * len(input_ids)}

        messages = Mock(messages=[SimpleNamespace(role="user")])
        messages.tokenize.side_effect = tokenize
        with patch.multiple(
            qwen_tokenize,
            ChatMessages=Mock(return_value=messages),
            Qwen35ChatMessages=Mock(return_value=messages),
            replace_image_token=Mock(),
        ):
            native = tokenize_fn.multi_modal_get_item(record)
            tokenize_fn.media_pipeline = _pipeline(fixture.manifest, materialization="lazy", per_image=True)
            lazy = tokenize_fn.multi_modal_get_item(record)
        assert torch.equal(native["position_ids"], lazy["position_ids"])
    assert "pixel_values" in native and "media_plan" not in native
    assert "pixel_values" not in lazy and lazy["media_plan"] is not None
    assert torch.equal(native["image_grid_thw"], lazy["image_grid_thw"])
    for field in ("input_ids", "labels", "num_tokens", "num_img_tokens", "num_imgs"):
        assert native[field] == lazy[field]


@pytest.mark.parametrize(
    "family_module,config_class",
    (
        (glm_tokenize, glm_tokenize.Glm53VLTokenizeFnConfig),
        (qwen_tokenize, qwen_tokenize.Qwen3VLTokenizeFnConfig),
    ),
)
def test_family_build_forwards_validated_pipeline(family_module, config_class):
    pipeline = _pipeline(unsupported_policy="eager")
    config = config_class(processor_path="unused", media_pipeline=pipeline)
    constructor = "Glm53VLTokenizeFunction" if family_module is glm_tokenize else "Qwen3VLTokenizeFunction"
    assert config.supports_media_pipeline is True
    with patch.object(family_module, constructor) as build:
        config.build(Mock())
    assert build.call_args.kwargs["media_pipeline"] is pipeline


@pytest.mark.parametrize("fail", (False, True))
def test_common_materializer_is_closed_even_on_failure(monkeypatch, fail):
    instance = Mock()
    factory = Mock(return_value=instance)
    monkeypatch.setitem(
        sys.modules, "xtuner.v1.datasets.media.materializer", SimpleNamespace(MediaMaterializer=factory)
    )
    plan = object()
    config = object()
    if fail:
        instance.materialize.side_effect = ValueError("native processing failure")
        with pytest.raises(ValueError, match="native processing failure"):
            shared._materialize_plans([plan], config)
    else:
        assert shared._materialize_plans([plan], config) == [instance.materialize.return_value]
    factory.assert_called_once_with(config)
    instance.materialize.assert_called_once_with(plan)
    instance.close.assert_called_once_with()
