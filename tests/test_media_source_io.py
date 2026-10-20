import ast
import builtins
import hashlib
import importlib.util
import io
import json
import math
import stat
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock, Mock, call, patch


PROVIDERS = Path(__file__).parents[1] / "xtuner/v1/datasets/media/providers"
SPEC = importlib.util.spec_from_file_location("bounded_media_source_io", PROVIDERS / "source_io.py")
SOURCE_IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE_IO)
CONTENT = b"certified-image-source"
DIGEST = hashlib.sha256(CONTENT).hexdigest()
REMOTE_URIS = (
    "s3://bucket/image.png",
    "cluster:s3://bucket/image.png",
    "petrel://bucket/image.png",
    "cluster:petrel://bucket/image.png",
    "http://server/image.png",
    "https://server/image.png",
    "file:///images/image.png",
    "S3://bucket/image.png",
    "s3:bucket/image.png",
)


def fake_file(content=CONTENT, size=None, mode=stat.S_IFREG):
    stream = Mock()
    stream.read.side_effect = lambda maximum: content[:maximum]
    context = MagicMock()
    context.__enter__.return_value = stream
    operations = SimpleNamespace(
        O_RDONLY=0,
        O_NONBLOCK=2048,
        open=Mock(return_value=17),
        fstat=Mock(return_value=SimpleNamespace(st_mode=mode, st_size=len(CONTENT) if size is None else size)),
        fdopen=Mock(return_value=context),
        close=Mock(),
    )
    trace = Mock()
    for name in ("open", "fstat", "fdopen", "close"):
        trace.attach_mock(getattr(operations, name), name)
    trace.attach_mock(stream.read, "read")
    return operations, stream, context, trace


def load_provider_helpers(importer):
    tree = ast.parse((PROVIDERS / "shared.py").read_text())
    provider = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CertifiedImageProvider"
    )
    selected = {"sources_for_certificate", "_profile", "_source_specs", "sources_for_uris", "describe", "_read_source"}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnsupportedMediaSource"]
    nodes.extend(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name in selected)
    nodes.extend(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_image_inputs"
    )
    namespace = {
        "MediaSource": SimpleNamespace,
        "MediaPlan": object,
        "TensorSpec": object,
        "Sequence": Sequence,
        "UnsupportedSourceIO": SOURCE_IO.UnsupportedSourceIO,
        "require_local_source": SOURCE_IO.require_local_source,
        "read_verified_local_source": SOURCE_IO.read_verified_local_source,
        "canonical_json": lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
        "hashlib": hashlib,
        "io": io,
        "json": json,
        "math": math,
        "_combined_specs": lambda groups: (),
        "image_processor_identity": Mock(return_value={"frozen": True}),
        "_materialize_plans": Mock(side_effect=AssertionError("unsupported source reached materializer")),
        "__builtins__": {**vars(builtins), "__import__": importer},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PROVIDERS / "shared.py"), "exec"), namespace)
    return namespace


class BoundedSourceIOTests(unittest.TestCase):
    def read(self, operations, **options):
        with patch.object(SOURCE_IO, "os", operations):
            return SOURCE_IO.read_verified_local_source(
                options.get("uri", "/images/source.png"),
                options.get("source_bytes", len(CONTENT)),
                options.get("content_sha256", DIGEST),
            )

    def test_exact_file_uses_one_same_descriptor_and_one_unbuffered_bounded_read(self):
        operations, stream, context, trace = fake_file()
        self.assertEqual(self.read(operations), CONTENT)
        self.assertEqual(
            trace.mock_calls,
            [
                call.open("/images/source.png", operations.O_RDONLY | operations.O_NONBLOCK),
                call.fstat(17),
                call.fdopen(17, "rb", buffering=0, closefd=False),
                call.read(len(CONTENT) + 1),
                call.close(17),
            ],
        )
        context.__exit__.assert_called_once()

    def test_initial_size_mismatch_rejects_before_any_source_read(self):
        for size in (0, len(CONTENT) - 1, len(CONTENT) + 1, 2**50):
            with self.subTest(size=size):
                operations, stream, context, trace = fake_file(size=size)
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    self.read(operations)
                operations.fdopen.assert_not_called()
                stream.read.assert_not_called()
                operations.close.assert_called_once_with(17)

    def test_growth_after_fstat_reads_only_certified_size_plus_one_and_skips_hash(self):
        operations, stream, context, trace = fake_file(content=CONTENT + b"extra-growth")
        digest = Mock(side_effect=AssertionError("overrun must fail before hashing"))
        with patch.object(SOURCE_IO, "hashlib", SimpleNamespace(sha256=digest)):
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                self.read(operations)
        stream.read.assert_called_once_with(len(CONTENT) + 1)
        digest.assert_not_called()
        operations.close.assert_called_once_with(17)

    def test_shrink_after_fstat_is_not_retried(self):
        operations, stream, context, trace = fake_file(content=CONTENT[:-1])
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self.read(operations)
        stream.read.assert_called_once_with(len(CONTENT) + 1)
        operations.close.assert_called_once_with(17)

    def test_same_size_replacement_fails_sha(self):
        operations, stream, context, trace = fake_file(content=b"X" + CONTENT[1:])
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self.read(operations)
        operations.close.assert_called_once_with(17)

    def test_nonregular_sources_reject_after_nonblocking_open_without_read(self):
        for mode in (stat.S_IFIFO, stat.S_IFDIR, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK):
            with self.subTest(mode=mode):
                operations, stream, context, trace = fake_file(mode=mode)
                with self.assertRaisesRegex(SOURCE_IO.UnsupportedSourceIO, "regular file"):
                    self.read(operations)
                self.assertTrue(operations.open.call_args.args[1] & operations.O_NONBLOCK)
                operations.fdopen.assert_not_called()
                stream.read.assert_not_called()
                operations.close.assert_called_once_with(17)

    def test_fstat_failure_closes_descriptor_without_read(self):
        operations, stream, context, trace = fake_file()
        operations.fstat.side_effect = OSError("synthetic fstat failure")
        with self.assertRaises(OSError):
            self.read(operations)
        operations.fdopen.assert_not_called()
        operations.close.assert_called_once_with(17)

    def test_fdopen_failure_closes_descriptor(self):
        operations, stream, context, trace = fake_file()
        operations.fdopen.side_effect = OSError("synthetic fdopen failure")
        with self.assertRaises(OSError):
            self.read(operations)
        operations.close.assert_called_once_with(17)

    def test_read_failure_exits_stream_and_closes_descriptor(self):
        operations, stream, context, trace = fake_file()
        stream.read.side_effect = OSError("synthetic read failure")
        with self.assertRaises(OSError):
            self.read(operations)
        context.__exit__.assert_called_once()
        operations.close.assert_called_once_with(17)

    def test_open_failure_does_not_close_unowned_descriptor(self):
        operations, stream, context, trace = fake_file()
        operations.open.side_effect = OSError("synthetic open failure")
        with self.assertRaises(OSError):
            self.read(operations)
        operations.fstat.assert_not_called()
        operations.close.assert_not_called()

    def test_invalid_certified_size_rejects_before_open(self):
        for size in (True, None, 0, -1, 1.0, "1", sys.maxsize, sys.maxsize + 1):
            with self.subTest(size=size):
                operations, stream, context, trace = fake_file()
                with self.assertRaisesRegex(ValueError, "byte count"):
                    self.read(operations, source_bytes=size)
                operations.open.assert_not_called()

    def test_invalid_digest_rejects_before_open(self):
        for digest in (None, 1, "", "a" * 63, "a" * 65, "A" * 64, "z" * 64):
            with self.subTest(digest=digest):
                operations, stream, context, trace = fake_file()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.read(operations, content_sha256=digest)
                operations.open.assert_not_called()

    def test_remote_and_malformed_sources_reject_before_open(self):
        for uri in (*REMOTE_URIS, "", "bad\x00path", "http://[invalid", None):
            with self.subTest(uri=uri):
                operations, stream, context, trace = fake_file()
                with self.assertRaises(SOURCE_IO.UnsupportedSourceIO):
                    self.read(operations, uri=uri)
                operations.open.assert_not_called()

    def test_local_path_check_has_no_io(self):
        operations, stream, context, trace = fake_file()
        with patch.object(SOURCE_IO, "os", operations):
            for uri in ("/images/source.png", "images/source.png", "./image.png", "/images/with space.png"):
                SOURCE_IO.require_local_source(uri)
        operations.open.assert_not_called()

    def test_helper_imports_only_stdlib_and_has_no_client_or_decoder(self):
        tree = ast.parse((PROVIDERS / "source_io.py").read_text())
        imports = {
            alias.name.split(".")[0] if isinstance(node, ast.Import) else node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertEqual(imports, {"hashlib", "os", "stat", "sys", "urllib"})


class ProviderSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.imports = []

        def importer(name, globals=None, locals=None, fromlist=(), level=0):
            self.imports.append(name)
            if name == "mllm_tokenize_fn.base_mllm_tokenize_fn":
                return SimpleNamespace(get_image_path=lambda uri, root: uri)
            raise AssertionError(f"unsupported source must reject before importing {name}")

        self.helpers = load_provider_helpers(importer)
        self.provider = SimpleNamespace(
            certificates={},
            identity={"frozen": True},
            profile_fingerprint="profile",
            manifest_fingerprint="manifest",
            _oss_loader=SimpleNamespace(client=SimpleNamespace(get=Mock())),
        )
        for name in (
            "sources_for_certificate",
            "_profile",
            "_source_specs",
            "sources_for_uris",
            "describe",
            "_read_source",
        ):
            setattr(self.provider, name, MethodType(self.helpers[name], self.provider))

    def certificate(self, uri):
        certificate = {
            "uri": uri,
            "revision": f"sha256:{DIGEST}",
            "width": 2,
            "height": 2,
            "orientation": 1,
            "source_bytes": len(CONTENT),
            "content_sha256": DIGEST,
            "decode_verified": True,
            "eligible": True,
            "profiles": {"profile": {}},
        }
        self.provider.certificates[uri] = certificate
        return SimpleNamespace(**certificate, metadata_json=self.helpers["canonical_json"](certificate))

    def test_certified_oss_is_rejected_before_descriptor_plan_or_client_get(self):
        for uri in REMOTE_URIS:
            with self.subTest(uri=uri):
                source = self.certificate(uri)
                with self.assertRaises(self.helpers["UnsupportedMediaSource"]):
                    self.provider.sources_for_uris([uri])
                with self.assertRaises(self.helpers["UnsupportedMediaSource"]):
                    self.provider.describe(
                        [source], occurrence_id="record", per_image=True, source_manifest_fingerprint="manifest"
                    )
        self.provider._oss_loader.client.get.assert_not_called()
        self.assertEqual(self.imports, [])

    def test_optimized_oss_eager_policy_returns_to_native_before_torch_or_materializer(self):
        uri = REMOTE_URIS[0]
        self.certificate(uri)
        tokenizer = SimpleNamespace(
            media_pipeline=SimpleNamespace(enabled=True, unsupported_policy="eager"),
            _media_provider=self.provider,
            image_processor=object(),
            _image_path=[uri],
        )
        actual = self.helpers["prepare_image_inputs"](tokenizer, {}, "", "qwen_image")
        self.assertIsNone(actual)
        self.assertEqual(self.imports, ["mllm_tokenize_fn.base_mllm_tokenize_fn"])
        self.helpers["_materialize_plans"].assert_not_called()
        self.provider._oss_loader.client.get.assert_not_called()

    def test_oss_rejection_does_not_require_constructing_even_a_malformed_certificate(self):
        for uri in REMOTE_URIS:
            with self.subTest(uri=uri):
                self.provider.certificates[uri] = {"decode_verified": True}
                with self.assertRaisesRegex(self.helpers["UnsupportedMediaSource"], "no qualified bounded reader"):
                    self.provider.sources_for_uris([uri])
        self.assertEqual(self.imports, [])
        self.provider._oss_loader.client.get.assert_not_called()

    def test_optimized_oss_error_policy_fails_before_torch_or_materializer(self):
        uri = REMOTE_URIS[0]
        self.certificate(uri)
        tokenizer = SimpleNamespace(
            media_pipeline=SimpleNamespace(enabled=True, unsupported_policy="error"),
            _media_provider=self.provider,
            image_processor=object(),
            _image_path=[uri],
        )
        with self.assertRaisesRegex(self.helpers["UnsupportedMediaSource"], "no qualified bounded reader"):
            self.helpers["prepare_image_inputs"](tokenizer, {}, "", "glm_image")
        self.assertEqual(self.imports, ["mllm_tokenize_fn.base_mllm_tokenize_fn"])
        self.helpers["_materialize_plans"].assert_not_called()

    def test_disabled_and_absent_pipeline_do_not_enter_new_reader(self):
        for config in (None, SimpleNamespace(enabled=False)):
            with self.subTest(config=config):
                tokenizer = SimpleNamespace(media_pipeline=config)
                self.assertIsNone(self.helpers["prepare_image_inputs"](tokenizer, {}, "", "qwen_image"))
        self.assertEqual(self.imports, [])
        self.helpers["_materialize_plans"].assert_not_called()

    def test_changed_local_source_fails_before_any_decoder_import_or_call(self):
        source = SimpleNamespace(uri="/images/source.png", source_bytes=len(CONTENT), content_sha256=DIGEST)
        self.provider._source_specs = Mock()
        for payload in (CONTENT + b"growth", CONTENT[:-1], b"X" + CONTENT[1:]):
            with self.subTest(payload=payload):
                operations, stream, context, trace = fake_file(content=payload)
                with patch.object(SOURCE_IO, "os", operations):
                    with self.assertRaisesRegex(ValueError, "bytes differ"):
                        self.provider._read_source(source)
                operations.close.assert_called_once_with(17)
        self.assertEqual(self.imports, [])

    def test_direct_oss_read_is_defensively_rejected_even_if_specs_are_stubbed(self):
        self.provider._source_specs = Mock()
        source = SimpleNamespace(uri=REMOTE_URIS[0], source_bytes=len(CONTENT), content_sha256=DIGEST)
        with self.assertRaises(self.helpers["UnsupportedMediaSource"]):
            self.provider._read_source(source)
        self.provider._oss_loader.client.get.assert_not_called()
        self.assertEqual(self.imports, [])

    def test_verified_bytes_precede_fake_decoder_without_real_imports(self):
        source = SimpleNamespace(
            uri="/images/source.png",
            source_bytes=len(CONTENT),
            content_sha256=DIGEST,
            orientation=1,
            width=2,
            height=2,
        )
        events = []
        operations, stream, context, trace = fake_file()
        self.provider._source_specs = Mock(side_effect=lambda source: events.append("specs"))
        converted = Mock(size=(2, 2))
        converted.getexif.return_value = {274: 1}
        original = Mock()
        original.convert.return_value = converted
        opened = MagicMock()
        opened.__enter__.return_value = original

        def open_image(buffer):
            self.assertEqual(buffer.getvalue(), CONTENT)
            operations.close.assert_called_once_with(17)
            events.append("decode")
            return opened

        def importer(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "PIL":
                operations.close.assert_called_once_with(17)
                events.append("verified")
                return SimpleNamespace(Image=SimpleNamespace(open=open_image))
            if name == "utils":
                return SimpleNamespace(apply_exif_orientation=lambda image: image)
            raise AssertionError(f"unexpected import {name}")

        self.helpers["__builtins__"]["__import__"] = importer
        with patch.object(SOURCE_IO, "os", operations):
            self.assertIs(self.provider._read_source(source), converted)
        self.assertEqual(events, ["specs", "verified", "decode"])
        stream.read.assert_called_once_with(len(CONTENT) + 1)
        self.provider._oss_loader.client.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
