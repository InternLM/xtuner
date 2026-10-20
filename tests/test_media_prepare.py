import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, UnidentifiedImageError

from xtuner.v1.datasets.media import create_provider, prepare_image_manifest
from xtuner.v1.datasets.media.providers import shared
from xtuner.v1.datasets.media.providers.glm_image import GlmImageProvider


class MockImageProcessor:
    def __init__(self):
        self.config = {"patch_size": 2, "temporal_patch_size": 1, "merge_size": 1, "revision": 0}
        self.calls = 0
        self.predictions = 0
        self.error = None
        self.grid = [1, 2, 2]
        self.mutate_identity = False

    def to_json_string(self):
        return json.dumps(self.config)

    def get_number_of_image_patches(self, *, height, width):
        self.predictions += 1
        return 4

    def __call__(self, *, images, return_tensors):
        return self.preprocess(images, return_tensors=return_tensors)

    def preprocess(self, images, *, return_tensors):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.mutate_identity:
            self.config["revision"] += 1
        return {
            "pixel_values": torch.arange(48, dtype=torch.float32).reshape(4, 12),
            "image_grid_thw": torch.tensor([self.grid], dtype=torch.int64),
        }


class TestMediaPrepare(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "first.png"
        with Image.new("RGB", (4, 2), "red") as image:
            image.save(self.source)
        self.cache = self.root / "cache"
        self.processor = MockImageProcessor()
        identity_patch = patch.object(shared, "image_processor_identity", side_effect=self.identity)
        identity_patch.start()
        self.addCleanup(identity_patch.stop)

    @staticmethod
    def identity(provider, processor):
        module = (
            "transformers.models.glm5_next.fixture" if provider == "glm_image" else "transformers.models.qwen.fixture"
        )
        return {
            "provider": provider,
            "processor_module": module,
            "processor_class": "MockImageProcessor",
            "processor_config": json.loads(processor.to_json_string()),
            "transformers_version": "mock-version",
        }

    def prepare(self, sources=None, **overrides):
        arguments = {
            "provider": "glm_image",
            "processor": self.processor,
            "cache_dir": self.cache,
            "max_source_bytes": 1024**2,
            "max_inflight_bytes": 64 * 1024**2,
        }
        arguments.update(overrides)
        return prepare_image_manifest([self.source] if sources is None else sources, **arguments)

    def test_manifest_is_accepted_by_existing_provider_and_keeps_duplicate_occurrences(self):
        second = self.root / "same-content.png"
        shutil.copyfile(self.source, second)
        order = [self.source, second, self.source]
        output = self.prepare(order)
        manifest = json.loads(output.read_text())
        self.assertEqual(manifest["source_order"], [str(source) for source in order])
        self.assertEqual([source["uri"] for source in manifest["sources"]], [str(self.source), str(second)])
        self.assertEqual(self.processor.calls, 1)
        config = shared.make_image_provider_config("glm_image", self.processor, str(output))
        provider = create_provider("glm_image", config)
        sources = provider.sources_for_uris(manifest["source_order"])
        plan = provider.describe(
            sources,
            occurrence_id="record",
            per_image=True,
            source_manifest_fingerprint=provider.manifest_fingerprint,
        )
        self.assertEqual(len(plan.entries), 3)
        self.assertEqual(plan.output_specs[0].shape, (12, 12))

    def test_cache_hit_does_not_decode_predict_or_process(self):
        first = self.prepare()
        with patch.object(Image, "open", side_effect=AssertionError("cache hit decoded")):
            second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(self.processor.calls, 1)
        self.assertEqual(self.processor.predictions, 1)
        for entry in self.cache.rglob("*.json"):
            self.assertNotIn("tensor", entry.read_text())

    def test_processor_config_change_uses_a_new_cache_entry(self):
        first = self.prepare()
        self.processor.config["revision"] = 1
        second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual(self.processor.calls, 2)
        self.assertEqual(len(list((self.cache / "entries").glob("*.json"))), 2)

    def test_provider_version_change_uses_a_new_cache_entry(self):
        first = self.prepare()
        with patch.object(GlmImageProvider, "version", "next-version"):
            second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual(self.processor.calls, 2)

    def test_source_content_change_does_not_reuse_metadata(self):
        first = self.prepare()
        original = first.read_bytes()
        with Image.new("RGB", (4, 2), "blue") as image:
            image.save(self.source)
        second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(self.processor.calls, 2)

    def test_changed_uri_between_occurrences_is_an_error_not_deduplication(self):
        def sources():
            yield self.source
            with Image.new("RGB", (4, 2), "blue") as image:
                image.save(self.source)
            yield self.source

        with self.assertRaisesRegex(ValueError, "one URI changed"):
            self.prepare(sources())
        self.assertFalse((self.cache / "manifests").exists())

    def test_corrupt_cache_is_not_silently_reprocessed(self):
        output = self.prepare()
        before = output.read_bytes()
        entry = next((self.cache / "entries").glob("*.json"))
        cached = json.loads(entry.read_text())
        cached["metadata"]["width"] += 1
        entry.write_text(json.dumps(cached))
        with patch.object(Image, "open", side_effect=AssertionError("corrupt cache decoded")):
            with self.assertRaisesRegex(ValueError, "corrupt"):
                self.prepare()
        self.assertEqual(self.processor.calls, 1)
        self.assertEqual(output.read_bytes(), before)

    def test_invalid_cache_json_fails_without_decode(self):
        self.prepare()
        entry = next((self.cache / "entries").glob("*.json"))
        entry.write_text("not json")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.processor.calls, 1)

    def test_bad_image_does_not_publish_verified_metadata(self):
        self.source.write_bytes(b"not an image")
        with self.assertRaises(UnidentifiedImageError):
            self.prepare()
        self.assertEqual(self.processor.calls, 0)
        self.assertFalse(self.cache.exists())

    def test_native_processor_error_propagates_without_cache_or_manifest(self):
        failure = RuntimeError("native processor failed")
        self.processor.error = failure
        with self.assertRaises(RuntimeError) as caught:
            self.prepare()
        self.assertIs(caught.exception, failure)
        self.assertFalse(self.cache.exists())

    def test_actual_grid_schema_is_checked_before_cache_publication(self):
        self.processor.grid = [1, 1, 3]
        with self.assertRaisesRegex(ValueError, "pixel schema"):
            self.prepare()
        self.assertFalse(self.cache.exists())

    def test_processor_mutation_cannot_publish_under_the_old_identity(self):
        self.processor.mutate_identity = True
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.prepare()
        self.assertFalse(self.cache.exists())

    def test_admission_failure_precedes_decode_and_process(self):
        for limits in ({"max_source_bytes": 1}, {"max_inflight_bytes": 32 * 1024**2}):
            with self.subTest(limits=limits):
                with patch.object(Image, "open", side_effect=AssertionError("admission decoded")):
                    with self.assertRaisesRegex(ValueError, "admission"):
                        self.prepare(**limits)
        self.assertEqual(self.processor.calls, 0)

    def test_cached_reservation_still_respects_the_callers_budget(self):
        output = self.prepare()
        peak = json.loads(output.read_text())["sources"][0]["profiles"]
        limit = next(iter(peak.values()))["estimated_peak_bytes"] - 1
        with patch.object(Image, "open", side_effect=AssertionError("cache budget decoded")):
            with self.assertRaisesRegex(ValueError, "admission"):
                self.prepare(max_inflight_bytes=limit)
        self.assertEqual(self.processor.calls, 1)

    def test_unsupported_sources_providers_and_invalid_limits_are_explicit(self):
        for sources, options in (
            (["s3://bucket/image.png"], {}),
            ([self.source], {"provider": "new_provider"}),
            ([self.source], {"max_inflight_bytes": True}),
            ([self.source], {"max_source_bytes": 0}),
        ):
            with self.subTest(sources=sources, options=options):
                with self.assertRaises(ValueError):
                    self.prepare(sources, **options)
        with self.assertRaises(TypeError):
            self.prepare(str(self.source))
        self.assertEqual(self.processor.calls, 0)

    def test_qwen_uses_native_preprocess_and_produces_the_same_manifest_schema(self):
        output = self.prepare(provider="qwen_image")
        source = json.loads(output.read_text())["sources"][0]
        self.assertTrue(source["decode_verified"])
        self.assertTrue(source["eligible"])
        self.assertEqual(self.processor.calls, 1)

    def test_atomic_write_failure_cleans_temporary_metadata(self):
        with patch("xtuner.v1.datasets.media.prepare.os.replace", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.prepare()
        self.assertFalse(list(self.cache.rglob("*.json")))
        self.assertFalse(list(self.cache.rglob(".*.json.*")))


if __name__ == "__main__":
    unittest.main()
