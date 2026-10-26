"""CPU tests of the shared image adapter, without model weights or CUDA."""

import copy
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.datasets import lazy_media as media
from xtuner.v1.train import lazy_media_prefetch as prefetch


@pytest.fixture(params=["glm", "qwen"])
def image_plan(request, tmp_path):
    processor = (
        Glm5NextImageProcessor(min_image_tokens=4, max_image_tokens=32)
        if request.param == "glm"
        else Qwen2VLImageProcessor(patch_size=16, size={"shortest_edge": 4096, "longest_edge": 16384})
    )
    paths, images = [], []
    for i, (width, height) in enumerate([(64, 96), (96, 64)]):
        path = tmp_path / f"{i}.png"
        image = Image.new("RGB", (width, height), (30 + i * 70, 50, 100))
        image.save(path)
        paths.append(str(path))
        images.append(image)
    expected = processor(images=images, return_tensors="pt")
    for image in images:
        image.close()
    plan = media.make_image_plan(processor, paths, expected["image_grid_thw"].tolist())
    return plan, expected


@pytest.mark.parametrize("threads", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_processor_pixels_grid_and_order(image_plan, monkeypatch, threads, dtype):
    plan, expected = image_plan
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_THREADS", str(threads))
    pixels, grid = media.materialize_image_plan(dict(plan, media_dtype=dtype))
    assert torch.equal(pixels, expected["pixel_values"].to(dtype))
    assert torch.equal(grid, expected["image_grid_thw"])
    assert pixels.device.type == grid.device.type == "cpu"


def test_merge_checks_settings_and_preserves_duplicates(image_plan):
    plan, _ = image_plan
    merged = media.merge_image_plans([plan, plan])
    assert merged["images"] == plan["images"] * 2
    for other in [dict(plan, media_dtype=torch.bfloat16), dict(plan, processor_config={"changed": True})]:
        with pytest.raises(ValueError, match="share processor"):
            media.merge_image_plans([plan, other])
    with pytest.raises(ValueError, match="before scheduling"):
        media.merge_image_plans([dict(plan, _future=Future())])


def test_mixed_eager_lazy_records_keep_native_order(image_plan):
    plan, expected = image_plan
    eager = {"pixel_values": expected["pixel_values"], "image_grid_thw": expected["image_grid_thw"]}
    items = [copy.deepcopy(eager), {"media_plan": plan}]
    assert media.pack_image_plans(items) is None
    assert torch.equal(torch.cat([item["pixel_values"] for item in items]), expected["pixel_values"].repeat(2, 1))


def test_bad_geometry_fails_and_closes_original(image_plan, monkeypatch):
    from xtuner.v1.datasets.mllm_tokenize_fn import base_mllm_tokenize_fn as module

    plan, _ = image_plan
    original_load = module.load_image
    opened = []

    def load(path):
        image = original_load(path)
        opened.append(image)
        return image

    monkeypatch.setattr(module, "load_image", load)
    broken = copy.deepcopy(plan)
    broken["images"][0]["grid"] = [1, 2, 2]
    with pytest.raises(ValueError, match="predicted grid"):
        media.materialize_image_plan(broken)
    with pytest.raises(ValueError):
        opened[0].getpixel((0, 0))


def context(plan):
    return SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu").copy(media_plan=plan)


def test_prefetched_resolution_keeps_original_plan_without_pixel_aliases(image_plan, monkeypatch):
    plan, expected = image_plan
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_SHARED", "0")
    future = Future()
    future.set_result((expected["pixel_values"], expected["image_grid_thw"]))
    plan["_future"] = future
    original = context(plan)
    resolved = media.resolve_lazy_media(original)
    assert resolved is not original
    assert resolved.media_plan is None
    assert original.pixel_values is None
    assert "_future" not in original.media_plan
    assert torch.equal(resolved.pixel_values, expected["pixel_values"])
    assert media.resolve_lazy_media(context(None)).media_plan is None


def test_context_copy_and_cat_preserve_adapter(image_plan):
    plan, _ = image_plan
    first = context(plan)
    assert first.copy().media_plan is plan
    combined = SequenceContext.cat([first, context(copy.deepcopy(plan))])
    assert combined.media_plan == media.merge_image_plans([plan, plan])


@pytest.mark.parametrize("failure", [False, True])
def test_sp_status_and_payload_cpu_mock(monkeypatch, failure):
    import torch.distributed as dist

    group = object()
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda selected: [4, 5])
    monkeypatch.setattr(dist, "get_rank", lambda selected: 0)
    broadcasts = []
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src, group: broadcasts.append((src, tensor.clone())))
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    plan = {"images": [{"grid": [1, 2, 2]}], "media_dtype": torch.float32}
    decoder = (
        Mock(side_effect=ValueError("decode failed"))
        if failure
        else Mock(return_value=(torch.arange(12).reshape(4, 3).float(), torch.tensor([[1, 2, 2]])))
    )
    monkeypatch.setattr(prefetch, "materialize_image_plan", decoder)
    if failure:
        with pytest.raises(RuntimeError, match="decoder rank 4 failed"):
            prefetch.shared_materialize(plan, group, "cpu", 3)
        assert len(broadcasts) == 1
    else:
        pixels, grid = prefetch.shared_materialize(plan, group, "cpu", 3)
        assert pixels.shape == (4, 3) and grid.tolist() == [[1, 2, 2]]
        assert len(broadcasts) == 3
    assert all(src == 4 for src, _ in broadcasts)


@pytest.mark.parametrize("has_plan", [False, True])
def test_sp_plan_disagreement_fails_before_payload(monkeypatch, has_plan):
    import torch.distributed as dist

    monkeypatch.setenv("XTUNER_LAZY_MEDIA_SHARED", "1")
    original = context({"images": []} if has_plan else None)
    original.sequence_parallel_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: object())
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)

    def disagree(output, signature, group):
        output[0].copy_(signature)
        output[1].fill_(0)

    monkeypatch.setattr(dist, "all_gather", disagree)
    payload = Mock()
    monkeypatch.setattr(prefetch, "shared_materialize", payload)
    with pytest.raises(ValueError, match="different lazy media plans"):
        media.resolve_lazy_media(original)
    payload.assert_not_called()


@pytest.mark.parametrize("family", ["glm", "qwen"])
def test_model_visual_entry_calls_shared_resolver(image_plan, monkeypatch, family):
    from xtuner.v1.model.compose.glm53.modeling_glm53 import Glm53ForConditionalGeneration
    from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    plan, expected = image_plan
    original = context(plan)
    original.mm_token_type_ids = torch.ones_like(original.input_ids)
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_SHARED", "0")
    decoder = Mock(return_value=(expected["pixel_values"], expected["image_grid_thw"]))
    monkeypatch.setattr(media, "materialize_image_plan", decoder)
    embedding = SimpleNamespace(embed_tokens=lambda ids: torch.zeros((*ids.shape, 3)))
    fake = SimpleNamespace(language_model=embedding, only_llm_forward=False)
    if family == "glm":
        fake.vision_tower = SimpleNamespace(spatial_merge_size=2)
        fake.get_visual_features = Mock(return_value=torch.ones(1, 3))
        fake._splice = Mock(side_effect=lambda embeds, *args: embeds)
        Glm53ForConditionalGeneration._prepare_llm_inputs(fake, original)
    else:
        fake.get_visual_features = Mock(return_value=(torch.ones(1, 3), None))
        fake.get_placeholder_mask = Mock(return_value=(torch.tensor([[True, False]]), torch.ones(1, 3), None))
        Qwen3VLForConditionalGeneration._prepare_llm_inputs(fake, original)
    decoder.assert_called_once_with(plan)
    assert torch.equal(fake.get_visual_features.call_args.args[0], expected["pixel_values"])
    assert original.pixel_values is None  # The language context still starts from the plan-only batch.


def test_context_cat_rejects_unresolved_mixed_order(image_plan):
    plan, expected = image_plan
    eager = context(None).copy(pixel_values=expected["pixel_values"])
    with pytest.raises(ValueError, match="mixed eager/lazy"):
        SequenceContext.cat([eager, context(plan)])
