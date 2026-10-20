import gc
import json
import os
import threading
import unittest
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock, patch

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from xtuner.v1.datasets.media import (
    MaterializedMedia,
    MediaEntry,
    MediaPipelineConfig,
    MediaPlan,
    MediaSource,
    ProviderCapabilities,
    TensorSpec,
    register_provider,
)
from xtuner.v1.datasets.media import materializer as materializer_module
from xtuner.v1.datasets.media import transport as transport_module
from xtuner.v1.datasets.media.materializer import (
    MediaBudgetError,
    MediaCancelledError,
    MediaMaterializer,
    MediaPipelineError,
    MediaShutdownTimeout,
)
from xtuner.v1.datasets.media.providers.shared import CertifiedImageProvider
from xtuner.v1.datasets.media.runtime import MediaRuntime
from xtuner.v1.datasets.media.transport import MediaTransport, MediaTransportError


_PROVIDER = "runtime_contract_test_provider"
_LOCK = threading.Lock()
_INSTANCES = []
_PROCESSED = []
_STARTED = threading.Event()
_RELEASE = threading.Event()
_SECOND_FINISHED = threading.Event()
_THIRD_STARTED = threading.Event()
_REFILL_CANCEL = threading.Event()
_REGISTERED = False


def configuration(**overrides):
    values = {
        "enabled": True,
        "materialization": "lazy",
        "source_manifest": "fixture-manifest",
        "per_image": True,
        "decode_threads": 2,
        "max_pending_tasks": 2,
        "max_inflight_bytes": 2**20,
        "max_header_bytes": 1024,
        "materialization_timeout_seconds": 3,
        "shutdown_timeout_seconds": 1,
    }
    values.update(overrides)
    return MediaPipelineConfig(**values)


def make_plan(count=3, behavior="normal", offset=0):
    entries = tuple(
        MediaEntry(
            occurrence_id=f"occurrence-{ordinal + offset}",
            sources=(MediaSource(f"fixture://{ordinal + offset}", "immutable-v1", 16, 16),),
            output_specs=(
                TensorSpec("pixel_values", "float32", (2, 2)),
                TensorSpec("image_grid_thw", "int64", (1, 3)),
            ),
            estimated_peak_bytes=512,
            metadata_json=json.dumps({"ordinal": ordinal + offset}),
        )
        for ordinal in range(count)
    )
    return MediaPlan(
        provider=_PROVIDER,
        provider_version="1",
        processor_config_json=json.dumps({"behavior": behavior}),
        processor_fingerprint="fixture-processor-v1",
        source_manifest_fingerprint="fixture-manifest-v1",
        entries=entries,
        output_specs=(
            TensorSpec("pixel_values", "float32", (2 * count, 2)),
            TensorSpec("image_grid_thw", "int64", (count, 3)),
        ),
        sample_ids=("fixture-record",),
    )


class _Provider:
    version = "1"
    capabilities = ProviderCapabilities(metadata_exact=True, independent_images=True)

    def __init__(self, config_json):
        self.behavior = json.loads(config_json)["behavior"]
        self.owner = threading.get_ident()
        if self.behavior == "bad_version":
            self.version = "2"
        with _LOCK:
            _INSTANCES.append((id(self), self.owner))

    def process(self, entry):
        if self.owner != threading.get_ident():
            raise AssertionError("processor shared across workers")
        ordinal = json.loads(entry.metadata_json)["ordinal"]
        if self.behavior == "blocked":
            _STARTED.set()
            if not _RELEASE.wait(5):
                raise TimeoutError("test fixture release deadline exceeded")
        if self.behavior == "reorder":
            if ordinal == 0 and not _SECOND_FINISHED.wait(2):
                raise TimeoutError("test fixture second unit never completed")
            if ordinal == 1:
                _SECOND_FINISHED.set()
        if self.behavior.startswith("completion_refill"):
            if ordinal == 0:
                _STARTED.set()
                if not _THIRD_STARTED.wait(2):
                    raise TimeoutError("idle worker did not start ordinal two before ordinal zero completed")
            if ordinal == 1 and not _STARTED.wait(2):
                raise TimeoutError("fixture ordinal zero never started")
            if ordinal == 2:
                _THIRD_STARTED.set()
                if self.behavior == "completion_refill_failure":
                    raise ValueError("fixture completion refill failure")
                if self.behavior == "completion_refill_cancel":
                    _REFILL_CANCEL.set()
        if self.behavior == "failure":
            raise ValueError("fixture provider failure")
        pixels = torch.full((2, 2), float(ordinal), dtype=torch.float32)
        if self.behavior == "bad_shape":
            pixels = torch.empty((3, 2), dtype=torch.float32)
        if self.behavior == "large_storage":
            pixels = torch.empty((128, 2), dtype=torch.float32)[:2].contiguous()
        if self.behavior == "non_cpu":
            pixels = torch.empty((2, 2), dtype=torch.float32, device="meta")
        with _LOCK:
            _PROCESSED.append((ordinal, self.owner))
        return MaterializedMedia(
            {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.int64)}
        )

    def assemble(self, plan, results):
        if self.owner != threading.get_ident():
            raise AssertionError("assembler shared across workers")
        return MaterializedMedia(
            {
                spec.name: torch.cat([result.tensors[spec.name] for result in results], dim=spec.concat_dim)
                if results
                else torch.empty(spec.shape, dtype=getattr(torch, spec.dtype))
                for spec in plan.output_specs
            },
            metadata={"fixture": "ordered"},
        )


class _AssemblyProvider(_Provider):
    name = _PROVIDER
    _validate_plan = CertifiedImageProvider._validate_plan
    _validate_assembly = CertifiedImageProvider._validate_assembly
    allocate_assembly = CertifiedImageProvider.allocate_assembly

    def __init__(self, config_json):
        super().__init__(config_json)
        self.config_json = config_json
        self.processor_fingerprint = "fixture-processor-v1"
        self.manifest_fingerprint = "fixture-manifest-v1"

    def process(self, entry):
        media = super().process(entry)
        media.metadata["occurrence_id"] = entry.occurrence_id
        return media


def assembly_plan(count=3):
    plan = make_plan(count=count)
    return replace(
        plan,
        entries=tuple(
            replace(
                entry,
                metadata_json=json.dumps(
                    {
                        **json.loads(entry.metadata_json),
                        "image_grid_thw": [[1, 1, 2]],
                    }
                ),
            )
            for entry in plan.entries
        ),
        output_specs=plan.output_specs if count else (),
    )


def assembly_provider_factory(provider_name, processor_config_json):
    return _AssemblyProvider(processor_config_json)


class _Mesh:
    def __init__(self):
        self.group = object()

    def size(self):
        return 2

    def get_group(self):
        return self.group


class _Collectives:
    class ReduceOp:
        MIN = "min"

    def __init__(
        self, rank=0, leader=8, wire=None, reject=False, mismatch=False, control_error_rank=None, work_error=None
    ):
        self.rank = rank
        self.leader = leader
        self.wire = wire if wire is not None else []
        self.cursor = 0
        self.reject = reject
        self.mismatch = mismatch
        self.control_error_rank = control_error_rank
        self.events = []
        self.works = []
        self.work_error = work_error

    def get_world_size(self, group):
        return 2

    def get_rank(self, group):
        return self.rank

    def get_global_rank(self, group, group_rank):
        assert group_rank == 0
        return self.leader

    def all_gather(self, outputs, tensor, group):
        self.events.append("fingerprint")
        for output in outputs:
            output.copy_(tensor)
        if self.control_error_rank is not None:
            outputs[self.control_error_rank][0] = 1
        if self.mismatch:
            outputs[1][1] ^= 1

    def broadcast(self, tensor, src, group, async_op=False):
        assert src == self.leader
        self.events.append(f"broadcast:{tensor.dtype}")
        if self.rank == 0:
            self.wire.append(tensor.clone())
        else:
            tensor.copy_(self.wire[self.cursor])
            self.cursor += 1
        if async_op:
            work = Mock()
            work.wait.return_value = True
            work.wait.side_effect = self.work_error
            work.payload_size = tensor.numel()
            self.works.append(work)
            return work

    def all_reduce(self, tensor, op, group):
        self.events.append("admission")
        if self.reject:
            tensor.zero_()


class _ProviderFixture:
    @classmethod
    def setUpClass(cls):
        global _REGISTERED
        if not _REGISTERED:
            register_provider(_PROVIDER, _Provider)
            _REGISTERED = True

    def setUp(self):
        with _LOCK:
            _INSTANCES.clear()
            _PROCESSED.clear()
        _STARTED.clear()
        _RELEASE.clear()
        _SECOND_FINISHED.clear()
        _THIRD_STARTED.clear()
        _REFILL_CANCEL.clear()


class TestMediaMaterializer(_ProviderFixture, unittest.TestCase):
    def test_parallel_assembly_is_bounded_same_pool_bit_exact_and_ordered(self):
        plan = assembly_plan(count=7)
        config = configuration(decode_threads=4, max_pending_tasks=4)
        probe = MediaMaterializer(config)
        required = probe.reservation_bytes(plan)
        probe.close()
        materializer = MediaMaterializer(config.model_copy(update={"max_inflight_bytes": required}))
        original_copy = materializer._copy_assembly
        original_submit = materializer._submit
        original_process = _AssemblyProvider.process
        active = set()
        workers = set()
        overlap = threading.Event()
        gate_lock = threading.Lock()
        copying = []

        def process(provider, entry):
            result = original_process(provider, entry)
            ordinal = json.loads(entry.metadata_json)["ordinal"]
            result.tensors["pixel_values"].view(torch.int32).copy_(
                torch.tensor([[0, -(2**31)], [0x7FC00000 + ordinal, ordinal]], dtype=torch.int32)
            )
            return result

        def copy(media, result, offsets, stop, deadline):
            self.assertEqual(media.storage_bytes, plan.output_bytes)
            self.assertEqual(materializer.reservation_bytes(plan), required)
            with gate_lock:
                workers.add(threading.get_ident())
                active.add(offsets["pixel_values"])
                if len(active) == 4:
                    overlap.set()
            try:
                if offsets["pixel_values"] < 8 and not overlap.wait(2):
                    raise TimeoutError("first four assembly copies did not overlap")
                return original_copy(media, result, offsets, stop, deadline)
            finally:
                with gate_lock:
                    active.remove(offsets["pixel_values"])

        def submit(operation, arguments, stop, deadline):
            if operation == materializer._copy_assembly:
                self.assertLess(sum(not future.done() for future in copying), 4)
                future = original_submit(operation, arguments, stop, deadline)
                copying.append(future)
                return future
            return original_submit(operation, arguments, stop, deadline)

        try:
            with (
                patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                patch.object(_AssemblyProvider, "process", process),
                patch.object(_AssemblyProvider, "assemble", side_effect=AssertionError("legacy concat used")),
                patch.object(materializer, "_copy_assembly", side_effect=copy),
                patch.object(materializer, "_submit", side_effect=submit),
            ):
                result = materializer.materialize(plan)
            expected = (
                torch.tensor([[0, -(2**31), 0x7FC00000 + ordinal, ordinal] for ordinal in range(7)], dtype=torch.int32)
                .reshape(14, 2)
                .view(torch.float32)
            )
            self.assertTrue(torch.equal(result.tensors["pixel_values"].view(torch.uint8), expected.view(torch.uint8)))
            self.assertEqual(result.tensors["image_grid_thw"].tolist(), [[1, 1, 2]] * 7)
            self.assertEqual(result.metadata, {})
            self.assertEqual(result.metrics["materializer_reserved_bytes"], required)
            self.assertTrue(
                all(tensor.is_contiguous() and tensor.device.type == "cpu" for tensor in result.tensors.values())
            )
            self.assertEqual(len(workers), 4)
            self.assertTrue(workers.issubset({thread.ident for thread in materializer._executor._threads}))
            self.assertNotIn(threading.get_ident(), workers)
            self.assertTrue(all(future.done() for future in copying))
        finally:
            overlap.set()
            materializer.close()

    def test_parallel_assembly_single_worker_empty_and_metadata_preserved(self):
        original_allocate = _AssemblyProvider.allocate_assembly

        def allocate(provider, plan, results):
            media = original_allocate(provider, plan, results)
            media.metadata["fixture"] = "preserved"
            media.metrics["provider_metric"] = 7
            return media

        for count in (0, 1, 5):
            with self.subTest(count=count):
                materializer = MediaMaterializer(configuration(decode_threads=1, max_pending_tasks=1))
                try:
                    with (
                        patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                        patch.object(_AssemblyProvider, "allocate_assembly", allocate),
                    ):
                        if count == 0:
                            with self.assertRaisesRegex(ValueError, "cannot assemble an empty image plan"):
                                materializer.materialize(assembly_plan(count=count))
                            continue
                        result = materializer.materialize(assembly_plan(count=count))
                    self.assertEqual(result.metadata, {"fixture": "preserved"})
                    self.assertEqual(result.metrics["provider_metric"], 7)
                    self.assertEqual(result.metrics["processed_entries"], count)
                    self.assertEqual(
                        result.tensors["pixel_values"][:, 0].tolist(),
                        [float(ordinal) for ordinal in range(count) for repeat in range(2)],
                    )
                finally:
                    materializer.close()

    def test_parallel_assembly_validates_order_grid_and_fresh_storage_before_copy(self):
        original_process = _AssemblyProvider.process
        for defect in ("order", "grid", "alias", "schema"):
            with self.subTest(defect=defect):
                materializer = MediaMaterializer(configuration())

                def process(provider, entry):
                    media = original_process(provider, entry)
                    if defect == "order":
                        media.metadata["occurrence_id"] = "wrong"
                    if defect == "grid":
                        media.tensors["image_grid_thw"][0, 0] = 2
                    return media

                allocate = _AssemblyProvider.allocate_assembly
                if defect == "alias":

                    def allocate(provider, plan, results):
                        return MaterializedMedia(dict(results[0].tensors))

                plan = assembly_plan(count=1)
                if defect == "schema":
                    plan = replace(
                        plan, output_specs=(replace(plan.output_specs[0], shape=(3, 2)), plan.output_specs[1])
                    )
                try:
                    with (
                        patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                        patch.object(_AssemblyProvider, "process", process),
                        patch.object(_AssemblyProvider, "allocate_assembly", allocate),
                        patch.object(materializer, "_copy_assembly", wraps=materializer._copy_assembly) as copy,
                    ):
                        with self.assertRaises(ValueError):
                            materializer.materialize(plan)
                        copy.assert_not_called()
                finally:
                    materializer.close()

    def test_parallel_assembly_variable_empty_slices_and_legacy_fallback(self):
        plan = assembly_plan(count=3)
        entries = tuple(
            replace(entry, output_specs=(replace(entry.output_specs[0], shape=(rows, 2)), entry.output_specs[1]))
            for entry, rows in zip(plan.entries, (1, 0, 3))
        )
        plan = replace(
            plan, entries=entries, output_specs=(replace(plan.output_specs[0], shape=(4, 2)), plan.output_specs[1])
        )
        original_process = _AssemblyProvider.process

        def process(provider, entry):
            media = original_process(provider, entry)
            ordinal = json.loads(entry.metadata_json)["ordinal"]
            media.tensors["pixel_values"] = torch.full(entry.output_specs[0].shape, float(ordinal))
            return media

        materializer = MediaMaterializer(configuration())
        try:
            with (
                patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                patch.object(_AssemblyProvider, "process", process),
            ):
                result = materializer.materialize(plan)
            self.assertEqual(result.tensors["pixel_values"][:, 0].tolist(), [0.0, 2.0, 2.0, 2.0])
            self.assertEqual(result.tensors["image_grid_thw"].tolist(), [[1, 1, 2]] * 3)
        finally:
            materializer.close()
        materializer = MediaMaterializer(configuration())
        try:
            with patch.object(materializer, "_copy_assembly", side_effect=AssertionError("legacy provider copied")):
                result = materializer.materialize(make_plan())
            self.assertEqual(result.metadata, {"fixture": "ordered"})
        finally:
            materializer.close()

    def test_parallel_assembly_failure_joins_other_copy_and_balances_runtime_lease(self):
        plan = assembly_plan(count=2)
        runtime = MediaRuntime(configuration())
        original_copy = runtime._materializer._copy_assembly
        started = threading.Event()
        cancelled = threading.Event()

        def copy(media, result, offsets, stop, deadline):
            if offsets["pixel_values"] == 0:
                started.set()
                if not stop.wait(2):
                    raise TimeoutError("assembly error did not cancel the other copy")
                cancelled.set()
                return original_copy(media, result, offsets, stop, deadline)
            if not started.wait(2):
                raise TimeoutError("other assembly copy never started")
            raise ValueError("fixture assembly copy failure")

        try:
            with (
                patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                patch.object(runtime._materializer, "_copy_assembly", side_effect=copy),
            ):
                with self.assertRaisesRegex(ValueError, "fixture assembly copy failure"):
                    with runtime.consume(plan, "assembly-error"):
                        self.fail("failed assembly was delivered")
            self.assertTrue(cancelled.is_set())
            self.assertFalse(runtime._materializer.poisoned)
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            with runtime._materializer._lock:
                self.assertTrue(all(future.done() for future in runtime._materializer._futures))
        finally:
            runtime.close()

    def test_parallel_assembly_cancel_and_expired_copy_never_return_output(self):
        for behavior in ("cancel", "deadline"):
            with self.subTest(behavior=behavior):
                materializer = MediaMaterializer(configuration(decode_threads=1, max_pending_tasks=1))
                stop = threading.Event()
                clock = {"now": 0}
                original_copy = materializer._copy_assembly

                def copy(media, result, offsets, stopping, deadline):
                    if behavior == "cancel":
                        stopping.set()
                    else:
                        clock["now"] = 4
                    return original_copy(media, result, offsets, stopping, deadline)

                try:
                    with (
                        patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                        patch.object(materializer, "_copy_assembly", side_effect=copy),
                        patch.object(materializer_module.time, "monotonic", side_effect=lambda: clock["now"]),
                    ):
                        expected = MediaCancelledError if behavior == "cancel" else TimeoutError
                        with self.assertRaises(expected):
                            materializer.materialize(assembly_plan(count=1), cancel_event=stop)
                    self.assertFalse(materializer.poisoned)
                    with materializer._lock:
                        self.assertTrue(all(future.done() for future in materializer._futures))
                finally:
                    materializer.close()

    def test_parallel_assembly_unjoined_writer_poison_retains_output_and_host_lease(self):
        plan = assembly_plan(count=2)
        runtime = MediaRuntime(configuration(shutdown_timeout_seconds=0.01))
        started = threading.Event()
        release = threading.Event()
        retained = []

        def copy(media, result, offsets, stop, deadline):
            if offsets["pixel_values"] == 0:
                retained.append(weakref.ref(media.tensors["pixel_values"]))
                started.set()
                if not release.wait(3):
                    raise TimeoutError("unjoined assembly fixture was not released")
                if stop.is_set():
                    raise MediaCancelledError("fixture stopped after native write")
            else:
                if not started.wait(2):
                    raise TimeoutError("unjoined assembly copy never started")
                raise ValueError("fixture copy failed beside blocked writer")

        try:
            with (
                patch.object(materializer_module, "create_provider", side_effect=assembly_provider_factory),
                patch.object(runtime._materializer, "_copy_assembly", side_effect=copy),
            ):
                with self.assertRaises(MediaShutdownTimeout):
                    with runtime.consume(plan, "assembly-unjoined"):
                        self.fail("unjoined writer was delivered")
            self.assertTrue(runtime._materializer.poisoned)
            self.assertEqual(runtime.metrics["poisoned"], 1)
            self.assertEqual(runtime.metrics["host_reserved_bytes"], runtime._materializer.reservation_bytes(plan))
            self.assertIsNotNone(retained[0]())
        finally:
            release.set()
            runtime._materializer._close_until(materializer_module.time.monotonic() + 2)
            runtime.close()

    def test_completed_later_ordinal_refills_idle_worker_before_first_result(self):
        materializer = MediaMaterializer(configuration())
        original_submit = materializer._submit
        process_futures = []

        def submit(operation, arguments, stop, deadline):
            if operation == materializer._process:
                self.assertLess(sum(not future.done() for future in process_futures), 2)
                future = original_submit(operation, arguments, stop, deadline)
                process_futures.append(future)
                return future
            return original_submit(operation, arguments, stop, deadline)

        try:
            plan = make_plan(count=4, behavior="completion_refill")
            reserved = materializer.reservation_bytes(plan)
            with patch.object(materializer, "_submit", side_effect=submit):
                result = materializer.materialize(plan)
            self.assertTrue(_THIRD_STARTED.is_set())
            self.assertEqual(len(process_futures), 4)
            self.assertEqual(result.tensors["pixel_values"][:, 0].tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
            self.assertEqual(result.tensors["image_grid_thw"].tolist(), [[1, 1, 2]] * 4)
            self.assertEqual(result.metrics["processed_entries"], 4)
            self.assertEqual(result.metrics["materializer_reserved_bytes"], reserved)
            self.assertEqual(sorted(ordinal for ordinal, _ in _PROCESSED), [0, 1, 2, 3])
            self.assertFalse(materializer.poisoned)
        finally:
            materializer.close()
        self.assertFalse(materializer._futures)
        self.assertFalse(any(thread.name.startswith(materializer._thread_prefix) for thread in threading.enumerate()))

    def test_completion_refill_failure_and_cancel_observe_tasks_without_assembly(self):
        for behavior, expected in (
            ("completion_refill_failure", ValueError),
            ("completion_refill_cancel", MediaCancelledError),
        ):
            with self.subTest(behavior=behavior):
                _STARTED.clear()
                _THIRD_STARTED.clear()
                _REFILL_CANCEL.clear()
                materializer = MediaMaterializer(configuration())
                try:
                    with patch.object(materializer, "_assemble", wraps=materializer._assemble) as assemble:
                        with self.assertRaises(expected) as raised:
                            materializer.materialize(
                                make_plan(count=3, behavior=behavior), cancel_event=_REFILL_CANCEL
                            )
                        assemble.assert_not_called()
                    self.assertTrue(_THIRD_STARTED.is_set())
                    self.assertTrue(_REFILL_CANCEL.is_set())
                    if behavior == "completion_refill_failure":
                        self.assertEqual(str(raised.exception), "fixture completion refill failure")
                    self.assertFalse(materializer.poisoned)
                    materializer.close()
                    self.assertTrue(materializer._slots.acquire(blocking=False))
                    self.assertTrue(materializer._slots.acquire(blocking=False))
                    self.assertFalse(materializer._slots.acquire(blocking=False))
                    materializer._slots.release()
                    materializer._slots.release()
                finally:
                    materializer.close()
                self.assertFalse(materializer._futures)
                self.assertFalse(
                    any(thread.name.startswith(materializer._thread_prefix) for thread in threading.enumerate())
                )

    def test_ordered_results_and_worker_local_factories(self):
        materializer = MediaMaterializer(configuration())
        try:
            result = materializer.materialize(make_plan(behavior="reorder"))
            self.assertEqual(result.tensors["pixel_values"][:, 0].tolist(), [0, 0, 1, 1, 2, 2])
            self.assertEqual(result.metrics["processed_entries"], 3)
            self.assertLessEqual(len({owner for _, owner in _INSTANCES}), 2)
            self.assertNotIn(threading.get_ident(), {owner for _, owner in _INSTANCES})
        finally:
            materializer.close()

    def test_single_thread_keeps_same_semantic_units(self):
        materializer = MediaMaterializer(configuration(decode_threads=1, max_pending_tasks=1))
        try:
            result = materializer.materialize(make_plan())
            self.assertEqual(result.tensors["pixel_values"][:, 0].tolist(), [0, 0, 1, 1, 2, 2])
            self.assertEqual([ordinal for ordinal, _ in _PROCESSED], [0, 1, 2])
        finally:
            materializer.close()

    def test_expired_deadline_never_submits_even_with_available_slot(self):
        materializer = MediaMaterializer(configuration(decode_threads=1, max_pending_tasks=1))
        try:
            with patch.object(materializer_module.time, "monotonic", return_value=2):
                with self.assertRaises(TimeoutError):
                    materializer._submit(lambda: None, (), threading.Event(), 1)
            self.assertIsNone(materializer._executor)
            with patch.object(materializer_module.time, "monotonic", side_effect=[0, 0, 2]):
                with self.assertRaises(TimeoutError):
                    materializer._submit(lambda: None, (), threading.Event(), 1)
            self.assertIsNone(materializer._executor)
            self.assertTrue(materializer._slots.acquire(blocking=False))
            self.assertFalse(materializer._slots.acquire(blocking=False))
            materializer._slots.release()
        finally:
            materializer.close()

    def test_completed_process_or_assemble_future_after_deadline_is_not_success(self):
        for count in (0, 1):
            with self.subTest(count=count):
                materializer = MediaMaterializer(configuration())
                clock = {"now": 0}
                completed = Future()
                completed.set_result(MaterializedMedia({}))

                def submit(*arguments):
                    clock["now"] = 4
                    return completed

                try:
                    with (
                        patch.object(materializer_module.time, "monotonic", side_effect=lambda: clock["now"]),
                        patch.object(materializer, "_submit", side_effect=submit) as submitted,
                    ):
                        with self.assertRaises(TimeoutError):
                            materializer.materialize(make_plan(count=count))
                    self.assertEqual(submitted.call_count, 1)
                    self.assertIsNone(materializer._executor)
                finally:
                    materializer.close()

    def test_version_shape_storage_device_and_provider_errors(self):
        for behavior, expected in (
            ("bad_version", ValueError),
            ("bad_shape", ValueError),
            ("large_storage", MediaBudgetError),
            ("non_cpu", ValueError),
            ("failure", ValueError),
        ):
            with self.subTest(behavior=behavior):
                materializer = MediaMaterializer(configuration())
                try:
                    with self.assertRaises(expected):
                        materializer.materialize(make_plan(behavior=behavior))
                finally:
                    materializer.close()

    def test_budget_rejected_before_factory_and_cancel_before_processing(self):
        plan = make_plan()
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(plan)
        probe.close()
        materializer = MediaMaterializer(configuration(max_inflight_bytes=required - 1))
        try:
            with self.assertRaises(MediaBudgetError):
                materializer.materialize(plan)
            self.assertEqual(_INSTANCES, [])
        finally:
            materializer.close()
        stop = threading.Event()
        stop.set()
        materializer = MediaMaterializer(configuration())
        try:
            with self.assertRaises(MediaCancelledError):
                materializer.materialize(plan, cancel_event=stop)
            self.assertEqual(_PROCESSED, [])
        finally:
            materializer.close()

    def test_empty_plan_and_closed_materializer(self):
        materializer = MediaMaterializer(configuration())
        empty = materializer.materialize(make_plan(count=0))
        self.assertEqual(empty.nbytes, 0)
        materializer.close()
        materializer.close()
        with self.assertRaises(MediaCancelledError):
            materializer.materialize(make_plan())


class TestMediaRuntime(_ProviderFixture, unittest.TestCase):
    def test_local_nonhost_transfer_preserves_host_lease_and_unwinds_failure(self):
        plan = make_plan(count=1)
        for fail_transfer in (False, True):
            with self.subTest(fail_transfer=fail_transfer):
                transfers = []
                transfer_error = RuntimeError("local grid transfer fixture")

                def move(tensor, device):
                    transfers.append((tuple(tensor.shape), str(tensor.dtype), torch.device(device)))
                    if fail_transfer:
                        raise transfer_error
                    return tensor.clone()

                fake = _Collectives()
                with (
                    MediaRuntime(configuration()) as runtime,
                    patch.object(transport_module, "dist", fake),
                    patch.object(torch.Tensor, "to", new=move),
                ):
                    required = runtime._materializer.reservation_bytes(plan)
                    self.assertIsNone(runtime.config.max_transport_bytes)
                    if fail_transfer:
                        with self.assertRaises(RuntimeError) as raised:
                            with runtime.consume(plan, "local", _Mesh(), device="cuda:0"):
                                self.fail("local device failure was hidden")
                        self.assertIs(raised.exception, transfer_error)
                    else:
                        with runtime.consume(plan, "local", _Mesh(), device="cuda:0") as media:
                            media.validate(plan.output_specs)
                            self.assertEqual(media.tensors["pixel_values"].device.type, "cpu")
                            self.assertEqual(runtime.metrics["host_reserved_bytes"], required)
                            self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
                    self.assertEqual(transfers, [((1, 3), "torch.int64", torch.device("cuda:0"))])
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["active_consumptions"], 0)
                    self.assertEqual(runtime.metrics["poisoned"], 0)
                    self.assertEqual(fake.events, [])

    def test_oversized_prefetch_declines_before_current_model_then_consume_fails_shared(self):
        current = make_plan(count=1)
        oversized = make_plan(count=3)
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(current)
        self.assertGreater(probe.control_bytes(oversized), required)
        probe.close()
        config = configuration(
            distribution="sp_leader",
            max_transport_bytes=2**20,
            max_inflight_bytes=required,
            prefetch_batches=1,
        )
        fake = _Collectives()
        with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", fake):
            with runtime.consume(current, "current", _Mesh()) as media:
                before = list(fake.events)
                self.assertFalse(runtime.prefetch(oversized, "future", True))
                self.assertEqual(runtime.metrics["prefetch_rejected"], 1)
                self.assertEqual(runtime.metrics["lookahead_groups"], 0)
                self.assertEqual(runtime.metrics["active_consumptions"], 1)
                self.assertEqual(fake.events, before)
                self.assertEqual(media.nbytes, current.output_bytes)
            fake.events.clear()
            with self.assertRaises(MediaTransportError) as raised:
                with runtime.consume(oversized, "future", _Mesh()):
                    self.fail("oversized current plan accepted")
            self.assertFalse(raised.exception.fail_stop)
            self.assertIn("plan/control validation failed before producer execution", str(raised.exception))
            self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
            self.assertEqual(fake.events, ["fingerprint"])
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["active_consumptions"], 0)
            self.assertEqual(len(_PROCESSED), 1)

    def test_materialization_budget_refusal_after_control_is_shared_by_status(self):
        oversized = make_plan(count=3)
        probe = MediaMaterializer(configuration())
        required = probe.control_bytes(oversized)
        self.assertGreater(probe.reservation_bytes(oversized), required)
        probe.close()
        config = configuration(distribution="sp_leader", max_transport_bytes=2**20, max_inflight_bytes=required)
        leader = _Collectives(rank=0, leader=8)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        for fake in (leader, follower):
            with self.subTest(rank=fake.rank):
                with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", fake):
                    with self.assertRaises(MediaTransportError) as raised:
                        with runtime.consume(oversized, "future", _Mesh()):
                            self.fail("oversized materialization accepted")
                    self.assertFalse(raised.exception.fail_stop)
                    self.assertIn("leader CPU materialization or schema validation failed", str(raised.exception))
                    if fake.rank == 0:
                        self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
                    self.assertEqual(fake.events, ["fingerprint", "broadcast:torch.int64"])
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["active_consumptions"], 0)
                    self.assertIsNone(runtime._materializer._executor)
        self.assertEqual(leader.wire[0].tolist(), [2, 0, 0])
        self.assertEqual(len(_PROCESSED), 0)
        self.assertEqual(len(_INSTANCES), 0)

    def test_invalid_prefetch_does_not_submit_and_still_requires_open_owner(self):
        config = configuration(prefetch_batches=1, max_tensor_rank=1)
        runtime = MediaRuntime(config)
        try:
            self.assertFalse(runtime.prefetch(make_plan(), "invalid-rank", True))
            self.assertFalse(runtime.prefetch(make_plan(), "", True))
            self.assertEqual(runtime.metrics["prefetch_rejected"], 2)
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertIsNone(runtime._materializer._executor)
        finally:
            runtime.close()
        with self.assertRaises(MediaPipelineError):
            runtime.prefetch(make_plan(), "closed", True)

    def test_leader_and_follower_current_protocol_ignore_rejected_future_plan(self):
        current = make_plan(count=1)
        oversized = make_plan(count=3)
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(current)
        probe.close()
        config = configuration(
            distribution="sp_leader",
            max_transport_bytes=2**20,
            max_inflight_bytes=required,
            prefetch_batches=1,
        )
        leader = _Collectives(rank=0, leader=8)
        with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", leader):
            with runtime.consume(current, "current", _Mesh()) as media:
                expected = {name: tensor.clone() for name, tensor in media.tensors.items()}
                self.assertFalse(runtime.prefetch(oversized, "future", True))
                self.assertEqual(runtime.metrics["prefetch_rejected"], 1)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", follower):
            with runtime.consume(current, "current", _Mesh()) as media:
                self.assertFalse(runtime.prefetch(oversized, "future", False))
                self.assertEqual(runtime.metrics["prefetch_rejected"], 0)
                self.assertIsNone(runtime._materializer._executor)
                for name, tensor in media.tensors.items():
                    self.assertTrue(torch.equal(tensor, expected[name]))
        self.assertEqual(leader.events, follower.events)
        self.assertEqual(len(_PROCESSED), 1)

    def test_prefetch_consumption_is_once_and_lease_accounting_is_logical(self):
        plan = make_plan()
        with MediaRuntime(configuration(prefetch_batches=1)) as runtime:
            self.assertTrue(runtime.prefetch(plan, "step-0/micro-0", True))
            self.assertFalse(runtime.prefetch(plan, "other", True))
            with runtime.consume(plan, "step-0/micro-0") as media:
                self.assertEqual(media.metrics["processed_entries"], 3)
                self.assertGreater(runtime.metrics["host_reserved_bytes"], 0)
                self.assertEqual(runtime.metrics["lookahead_groups"], 0)
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["reservation_scope_only"], 1)
            self.assertEqual(len(_PROCESSED), 3)
            self.assertEqual(media.tensors["pixel_values"].shape, (6, 2))

    def test_no_headroom_skips_lookahead_without_stalling_current(self):
        plan = make_plan()
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(plan)
        probe.close()
        with MediaRuntime(configuration(max_inflight_bytes=required, prefetch_batches=1)) as runtime:
            with runtime.consume(plan, "current"):
                self.assertFalse(runtime.prefetch(plan, "future", True))
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)

    def test_pending_other_group_can_be_cancelled_to_admit_current(self):
        plan = make_plan()
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(plan)
        probe.close()
        with MediaRuntime(configuration(max_inflight_bytes=required, prefetch_batches=1)) as runtime:
            self.assertTrue(runtime.prefetch(plan, "unused", True))
            with runtime.consume(plan, "current") as media:
                self.assertEqual(media.nbytes, plan.output_bytes)
            self.assertEqual(runtime.metrics["prefetch_cancelled"], 1)

    def test_sp_current_and_prefetch_share_control_reservation_without_double_charge(self):
        plan = make_plan(count=1)
        probe = MediaMaterializer(configuration())
        required = probe.reservation_bytes(plan)
        probe.close()
        config = configuration(
            max_inflight_bytes=required,
            prefetch_batches=1,
            distribution="sp_leader",
            max_transport_bytes=2**20,
        )
        with MediaRuntime(config) as runtime:
            with patch.object(transport_module, "dist", _Collectives()):
                with runtime.consume(plan, "current", _Mesh()) as media:
                    self.assertEqual(media.nbytes, plan.output_bytes)
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], required)
            self.assertTrue(runtime.prefetch(plan, "next", True))
            with patch.object(transport_module, "dist", _Collectives()):
                with runtime.consume(plan, "next", _Mesh()):
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], required)
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)

    def test_sp_native_pixel_delivery_balances_leader_and_follower_host_leases(self):
        plan = make_plan(count=1)
        config = configuration(distribution="sp_leader", max_transport_bytes=2**20)
        cpu = MaterializedMedia(
            {
                "pixel_values": torch.tensor([[1.0001, -0.0], [3.1415927, -2.0001]], dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.int64),
            }
        )
        leader = _Collectives(rank=0, leader=8)
        with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", leader):
            required = runtime._materializer.reservation_bytes(plan)
            with patch.object(runtime._materializer, "materialize", return_value=cpu) as materialize:
                with runtime.consume(plan, "cpu-lease", _Mesh()) as media:
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], required)
                    self.assertEqual(runtime.metrics["peak_host_reserved_bytes"], required)
                    for name, tensor in cpu.tensors.items():
                        delivered = media.tensors[name]
                        self.assertEqual(delivered.dtype, tensor.dtype)
                        self.assertEqual(tuple(delivered.shape), tuple(tensor.shape))
                        self.assertTrue(torch.equal(delivered.view(torch.uint8), tensor.view(torch.uint8)))
                        if name == "pixel_values":
                            self.assertIs(delivered, tensor)
                materialize.assert_called_once()
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["active_consumptions"], 0)
            self.assertEqual(runtime.metrics["poisoned"], 0)
            self.assertIsNone(runtime._materializer._executor)

        for consumer_error in (False, True):
            with self.subTest(consumer_error=consumer_error):
                follower = _Collectives(rank=1, leader=8, wire=leader.wire)
                with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", follower):
                    pixel_bytes = cpu.tensors["pixel_values"].numel() * cpu.tensors["pixel_values"].element_size()
                    expected_host = runtime._materializer.control_bytes(plan) + pixel_bytes
                    with patch.object(runtime._materializer, "materialize") as materialize:
                        try:
                            with runtime.consume(plan, "cpu-lease", _Mesh()) as media:
                                self.assertEqual(runtime.metrics["host_reserved_bytes"], expected_host)
                                self.assertEqual(runtime.metrics["peak_host_reserved_bytes"], expected_host)
                                for name, tensor in cpu.tensors.items():
                                    delivered = media.tensors[name]
                                    self.assertEqual(delivered.device.type, "cpu")
                                    self.assertEqual(delivered.dtype, tensor.dtype)
                                    self.assertEqual(tuple(delivered.shape), tuple(tensor.shape))
                                    self.assertTrue(torch.equal(delivered.view(torch.uint8), tensor.view(torch.uint8)))
                                if consumer_error:
                                    raise RuntimeError("fixture consumer failure")
                        except RuntimeError as error:
                            if not consumer_error or str(error) != "fixture consumer failure":
                                raise
                        else:
                            self.assertFalse(consumer_error)
                        materialize.assert_not_called()
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["active_consumptions"], 0)
                    self.assertEqual(runtime.metrics["poisoned"], 0)
                    self.assertIsNone(runtime._materializer._executor)

    def test_sp_follower_host_payload_budget_refuses_before_payload_and_balances(self):
        plan = make_plan(count=1)
        probe = MediaMaterializer(configuration())
        control_bytes = probe.control_bytes(plan)
        probe.close()
        pixel_bytes = next(spec.nbytes for spec in plan.output_specs if spec.name == "pixel_values")
        config = configuration(
            distribution="sp_leader",
            max_transport_bytes=2**20,
            max_inflight_bytes=control_bytes + pixel_bytes - 1,
        )
        cpu = MaterializedMedia(
            {
                "pixel_values": torch.ones((2, 2), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.int64),
            }
        )
        leader = _Collectives(rank=0, leader=8)
        with patch.object(transport_module, "dist", leader):
            MediaTransport(config).deliver(plan, "host-refusal", _Mesh(), "cpu", lambda: cpu)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        with MediaRuntime(config) as runtime, patch.object(transport_module, "dist", follower):
            with self.assertRaises(MediaTransportError) as raised:
                with runtime.consume(plan, "host-refusal", _Mesh()):
                    self.fail("follower payload exceeded the host budget")
            self.assertFalse(raised.exception.fail_stop)
            self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
            self.assertEqual(
                follower.events, ["fingerprint", "broadcast:torch.int64", "broadcast:torch.uint8", "admission"]
            )
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["active_consumptions"], 0)
            self.assertEqual(runtime.metrics["consumed"], 0)
            self.assertEqual(runtime.metrics["poisoned"], 0)
            self.assertIsNone(runtime._materializer._executor)

    def test_nonproducer_and_disabled_do_not_start_workers(self):
        with MediaRuntime(configuration(prefetch_batches=1)) as runtime:
            self.assertFalse(runtime.prefetch(make_plan(), "follower", False))
            self.assertIsNone(runtime._materializer._executor)
        with MediaRuntime(MediaPipelineConfig()) as runtime:
            with runtime.consume(None, "empty") as media:
                self.assertIsNone(media)
            with self.assertRaises(MediaPipelineError):
                with runtime.consume(make_plan(), "unresolved"):
                    self.fail("disabled runtime accepted unresolved plan")
            self.assertIsNone(runtime._materializer._executor)

    def test_duplicate_or_invalid_sp_key_is_rejected_during_fingerprint(self):
        config = configuration(distribution="sp_leader", max_transport_bytes=2**20)
        with MediaRuntime(config) as runtime:
            with runtime.consume(None, "active"):
                fake = _Collectives()
                with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
                    with runtime.consume(None, "active", _Mesh()):
                        self.fail("duplicate active key accepted")
                self.assertEqual(fake.events, ["fingerprint"])
                self.assertEqual(runtime.metrics["active_consumptions"], 1)
            fake = _Collectives()
            with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
                with runtime.consume(None, "", _Mesh()):
                    self.fail("empty key accepted")
            self.assertEqual(fake.events, ["fingerprint"])
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)

    def test_prefetched_failure_is_observed_on_consume_or_cancel(self):
        plan = make_plan(behavior="failure")
        with MediaRuntime(configuration(prefetch_batches=1)) as runtime:
            self.assertTrue(runtime.prefetch(plan, "failure", True))
            with self.assertRaises(ValueError):
                with runtime.consume(plan, "failure"):
                    self.fail("provider error was hidden")
            self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)

    def _assert_terminal_payload_released(self, *, provider_fault, prefetch):
        references, storages = [], []
        ready, returned = threading.Event(), threading.Event()
        runtime = MediaRuntime(configuration(prefetch_batches=1, decode_threads=1, max_pending_tasks=1))
        plan = make_plan(count=1)

        def process(provider, entry):
            inputs = torch.ones((1, 2, 2), dtype=torch.float32)
            pixels = inputs[0]
            grid = torch.tensor([[1, 1, 2]], dtype=torch.int64)
            media = MaterializedMedia({"pixel_values": pixels, "image_grid_thw": grid})
            references.extend(weakref.ref(tensor) for tensor in (inputs, pixels, grid))
            storages.extend(StorageWeakRef(tensor.untyped_storage()) for tensor in (inputs, grid))
            ready.set()
            if provider_fault:
                raise ValueError("fixture terminal provider failure after payload creation")
            if not runtime._pending.stop.wait(2):
                raise TimeoutError("fixture terminal cancellation was not requested")
            returned.set()
            return media

        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch.object(_Provider, "process", process):
                if prefetch:
                    self.assertTrue(runtime.prefetch(plan, "terminal", True))
                    self.assertTrue(ready.wait(2))
                    self.assertTrue(all(reference() is not None for reference in references))
                    self.assertTrue(all(not storage.expired() for storage in storages))
                if provider_fault:
                    try:
                        with runtime.consume(plan, "terminal"):
                            self.fail("provider failure was hidden")
                    except ValueError as error:
                        self.assertEqual(str(error), "fixture terminal provider failure after payload creation")
                        traceback_names = []
                        traceback_node = error.__traceback__
                        while traceback_node is not None:
                            traceback_names.append(traceback_node.tb_frame.f_code.co_name)
                            traceback_node = traceback_node.tb_next
                        self.assertIn("process", traceback_names)
                    else:
                        self.fail("provider failure was not propagated")
                else:
                    runtime.cancel_pending()
                    self.assertTrue(returned.is_set())
                    self.assertEqual(runtime.metrics["prefetch_cancelled"], 1)
                    self.assertEqual(runtime.metrics["prefetch_errors_observed"], 0)
                runtime.close()
                self.assertTrue(ready.is_set())
                self.assertFalse(gc.isenabled())
                self.assertEqual(len(references), 3)
                self.assertEqual(len(storages), 2)
                self.assertTrue(all(reference() is None for reference in references))
                self.assertTrue(all(storage.expired() for storage in storages))
                for name in (
                    "host_reserved_bytes",
                    "transport_reserved_bytes",
                    "active_consumptions",
                    "lookahead_groups",
                    "poisoned",
                ):
                    self.assertEqual(runtime.metrics[name], 0)
        finally:
            try:
                runtime.close()
            finally:
                if gc_enabled:
                    gc.enable()

    def test_provider_fault_releases_input_storage_without_gc(self):
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                self._assert_terminal_payload_released(provider_fault=True, prefetch=prefetch)

    def test_post_completion_cancel_releases_input_storage_without_gc(self):
        self._assert_terminal_payload_released(provider_fault=False, prefetch=True)

    def test_close_deadline_poison_preserves_unjoined_reservation(self):
        runtime = MediaRuntime(configuration(prefetch_batches=1, shutdown_timeout_seconds=0.01))
        try:
            self.assertTrue(runtime.prefetch(make_plan(behavior="blocked"), "blocked", True))
            self.assertTrue(_STARTED.wait(2))
            with self.assertRaises(MediaShutdownTimeout):
                runtime.close()
            self.assertGreater(runtime.metrics["host_reserved_bytes"], 0)
            self.assertEqual(runtime.metrics["poisoned"], 1)
        finally:
            _RELEASE.set()
            if runtime._pending is not None:
                runtime._pending.thread.join(4)
                self.assertFalse(runtime._pending.thread.is_alive())
                runtime._pending.future.exception()
            runtime._materializer.close()

    def test_owner_thread_is_required(self):
        runtime = MediaRuntime(configuration(prefetch_batches=1))
        errors = []

        def invalid_owner():
            try:
                runtime.prefetch(make_plan(), "wrong-thread", True)
            except MediaPipelineError as error:
                errors.append(error)

        thread = threading.Thread(target=invalid_owner)
        thread.start()
        thread.join(2)
        runtime.close()
        self.assertEqual(len(errors), 1)


class TestTypedMediaTransport(unittest.TestCase):
    def setUp(self):
        self.config = configuration(distribution="sp_leader", max_transport_bytes=2**20)
        self.plan = make_plan(count=1)
        self.cpu = MaterializedMedia(
            {"pixel_values": torch.ones((2, 2)), "image_grid_thw": torch.tensor([[1, 1, 2]])},
            metadata={"source": "fixture"},
        )

    def _chunked_fixture(self):
        bits = torch.tensor([0, -2147483648, 1065353217, 2139095040, 2143289345, -4194303, 1], dtype=torch.int32)
        pixels = bits.repeat(11)[:71].clone().view(torch.float32)
        videos = bits.repeat(11)[:73].clone().view(torch.float32)
        specs = (
            TensorSpec("pixel_values", "float32", (71,)),
            TensorSpec("pixel_values_videos", "float32", (73,)),
            self.plan.output_specs[1],
        )
        entry = replace(
            self.plan.entries[0], output_specs=specs, estimated_peak_bytes=sum(spec.nbytes for spec in specs)
        )
        plan = replace(self.plan, entries=(entry,), output_specs=specs)
        media = MaterializedMedia(
            {
                "pixel_values": pixels,
                "pixel_values_videos": videos,
                "image_grid_thw": self.cpu.tensors["image_grid_thw"],
            },
            metadata={"source": "chunk-fixture"},
        )
        return plan, media

    @contextmanager
    def _chunked_cuda(self, fake, fail_pinned=False, copy_hook=None):
        original_empty = torch.empty
        original_zeros = torch.zeros
        owner = threading.get_ident()
        allocations = {
            "gpu_slots": [],
            "pinned_bytes": [],
            "events": [],
            "streams": [],
            "executors": [],
            "cpu_submissions": [],
        }

        def owner_only(*args, **kwargs):
            self.assertEqual(threading.get_ident(), owner)

        def executor(**kwargs):
            self.assertEqual(kwargs["max_workers"], 8)
            pool = ThreadPoolExecutor(**kwargs)
            allocations["executors"].append(pool)
            submit = pool.submit

            def submit_copy(function, source, **copy_kwargs):
                owner_only()
                self.assertIn("admission", fake.events)
                self.assertEqual(function.__name__, "copy_")
                self.assertEqual(function.__self__.device.type, "cpu")
                self.assertEqual(source.device.type, "cpu")
                ordinal = len(allocations["cpu_submissions"])
                allocations["cpu_submissions"].append((function.__self__.numel(), source.numel()))

                def copy():
                    self.assertNotEqual(threading.get_ident(), owner)
                    if copy_hook is not None:
                        return copy_hook(function, source, ordinal, **copy_kwargs)
                    return function(source, **copy_kwargs)

                return submit(copy)

            pool.submit = submit_copy
            return pool

        def allocate(*args, **kwargs):
            requested = torch.device(kwargs.get("device") or "cpu")
            pinned = kwargs.pop("pin_memory", False)
            if pinned and fail_pinned:
                raise RuntimeError("fixture pinned allocation refused")
            tensor = original_empty(*args, **{**kwargs, "device": "cpu"})
            if pinned:
                allocations["pinned_bytes"].append(tensor.numel() * tensor.element_size())
            if requested.type == "cuda" and tensor.dtype == torch.uint8 and tensor.numel() == 16:
                allocations["gpu_slots"].append(weakref.ref(tensor))
            return tensor

        def event():
            result = Mock()
            result.record.side_effect = owner_only
            result.synchronize.side_effect = owner_only
            allocations["events"].append(result)
            return result

        def stream(**kwargs):
            owner_only()
            result = Mock()
            allocations["streams"].append(result)
            return result

        with (
            patch.object(transport_module, "dist", fake),
            patch.object(transport_module, "_RAWPIXEL_CHUNK_BYTES", 16),
            patch.object(transport_module, "ThreadPoolExecutor", side_effect=executor),
            patch.object(torch, "empty", new=allocate),
            patch.object(
                torch, "zeros", new=lambda *args, **kwargs: original_zeros(*args, **{**kwargs, "device": "cpu"})
            ),
            patch.object(torch.cuda, "current_stream", return_value=Mock()),
            patch.object(torch.cuda, "Stream", side_effect=stream),
            patch.object(torch.cuda, "Event", side_effect=event),
            patch.object(torch.cuda, "device", side_effect=lambda device: nullcontext()),
            patch.object(torch.cuda, "stream", side_effect=lambda current: nullcontext(current)),
        ):
            try:
                yield allocations
            finally:
                for pool in allocations["executors"]:
                    pool.shutdown(wait=True, cancel_futures=True)

    def test_chunked_rawpixel_roundtrip_wraps_sixteen_slots_and_preserves_all_bytes(self):
        plan, media = self._chunked_fixture()
        wire = []
        for rank in (0, 1):
            fake = _Collectives(rank=rank, wire=wire)
            charges = []

            def reserve_host(amount):
                self.assertNotIn("admission", fake.events)
                charges.append(amount)

            with self._chunked_cuda(fake) as allocations:
                result = MediaTransport(self.config).deliver(
                    plan,
                    "chunks",
                    _Mesh(),
                    "cuda:0",
                    lambda: media if rank == 0 else self.fail("follower invoked producer"),
                    reserve_host_bytes=reserve_host,
                )
                self.assertEqual(charges, [256 + (0 if rank == 0 else 576)])
                self.assertEqual(allocations["pinned_bytes"], [16] * 16)
                self.assertEqual(len(allocations["gpu_slots"]), 16)
                self.assertTrue(all(reference() is None for reference in allocations["gpu_slots"]))
                self.assertEqual(len(allocations["streams"]), 1)
                self.assertEqual(len(allocations["events"]), 32)
                self.assertEqual([work.payload_size for work in fake.works], [16] * 17 + [12] + [16] * 18 + [4])
                self.assertEqual(len(allocations["cpu_submissions"]), 37)
                self.assertTrue(all(pool._shutdown for pool in allocations["executors"]))
                self.assertTrue(
                    all(not thread.is_alive() for pool in allocations["executors"] for thread in pool._threads)
                )
                for work in fake.works:
                    work.block_current_stream.assert_called_once_with()
                    work.wait.assert_called_once_with()
                ready_events = allocations["events"][::2]
                done_events = allocations["events"][1::2]
                self.assertEqual([event.synchronize.call_count for event in done_events], [5, 5, 4] + [3] * 13)
                self.assertEqual([event.record.call_count for event in done_events], [5, 5, 4] + [3] * 13)
                self.assertEqual(
                    [event.record.call_count for event in ready_events],
                    [5, 5, 4] + [3] * 13 if rank == 0 else [1] * 16,
                )
                for event in done_events:
                    for call in event.record.call_args_list[1:]:
                        if rank == 1:
                            self.assertIs(call.args[0], allocations["streams"][0])
                for event in ready_events:
                    for call in event.record.call_args_list[1:]:
                        self.assertIs(call.args[0], allocations["streams"][0])
            result.validate(plan.output_specs)
            for name, expected in media.tensors.items():
                self.assertTrue(torch.equal(result.tensors[name].view(torch.uint8), expected.view(torch.uint8)))
                if name in transport_module._HOST_PAYLOAD_FIELDS:
                    self.assertEqual(result.tensors[name].device.type, "cpu")
                    if rank == 0:
                        self.assertIs(result.tensors[name], expected)
            self.assertEqual(result.metadata, media.metadata)

    def test_chunked_cpu_copies_overlap_and_all_finish_before_return(self):
        plan, media = self._chunked_fixture()
        wire = []
        for rank in (0, 1):
            with self.subTest(rank=rank):
                fake = _Collectives(rank=rank, wire=wire)
                all_workers_started = threading.Event()
                lock = threading.Lock()
                worker_ids = set()
                active_pinned = set()
                active_counts = []
                started = []
                completed = []

                def copy(function, source, ordinal, **kwargs):
                    pinned = function.__self__ if rank == 0 else source
                    pointer = pinned.untyped_storage().data_ptr()
                    with lock:
                        started.append(ordinal)
                        worker_ids.add(threading.get_ident())
                        self.assertNotIn(pointer, active_pinned)
                        active_pinned.add(pointer)
                        if len(active_pinned) == 8:
                            all_workers_started.set()
                        active_counts.append(len(active_pinned))
                        self.assertLessEqual(len(active_pinned), 8)
                    try:
                        if ordinal < 8 and not all_workers_started.wait(5):
                            raise TimeoutError("eight bounded CPU chunk workers did not start concurrently")
                        result = function(source, **kwargs)
                        with lock:
                            completed.append(ordinal)
                        return result
                    finally:
                        with lock:
                            active_pinned.remove(pointer)

                with self._chunked_cuda(fake, copy_hook=copy) as allocations:
                    result = MediaTransport(self.config).deliver(
                        plan,
                        "parallel-copies",
                        _Mesh(),
                        "cuda:0",
                        lambda: media if rank == 0 else self.fail("follower invoked producer"),
                        reserve_host_bytes=lambda _: None,
                    )
                    self.assertEqual(len(worker_ids), 8)
                    self.assertEqual(max(active_counts), 8)
                    self.assertEqual(sorted(completed), list(range(37)))
                    self.assertEqual(active_pinned, set())
                    self.assertTrue(
                        all(not thread.is_alive() for pool in allocations["executors"] for thread in pool._threads)
                    )
                    for name, expected in media.tensors.items():
                        self.assertTrue(
                            torch.equal(result.tensors[name].view(torch.uint8), expected.view(torch.uint8))
                        )

    def test_chunked_cpu_failure_retains_pending_tasks_and_runtime_reservations(self):
        plan, media = self._chunked_fixture()
        fake = _Collectives()
        failure = RuntimeError("fixture CPU chunk copy failure")
        blocked = threading.Event()
        release = threading.Event()

        def copy(function, source, ordinal, **kwargs):
            if ordinal == 0:
                if not blocked.wait(5):
                    raise TimeoutError("second CPU copy never started")
                raise failure
            if ordinal == 1:
                blocked.set()
                if not release.wait(5):
                    raise TimeoutError("fixture CPU copy was not released")
            return function(source, **kwargs)

        with (
            MediaRuntime(self.config) as runtime,
            self._chunked_cuda(fake, copy_hook=copy) as allocations,
            patch.object(runtime._materializer, "materialize", return_value=media),
        ):
            try:
                with self.assertRaises(MediaTransportError) as raised:
                    with runtime.consume(plan, "failed-cpu-copy", _Mesh(), device="cuda:0"):
                        self.fail("failed CPU copy returned partial media")
                self.assertTrue(raised.exception.fail_stop)
                self.assertIs(raised.exception.__cause__, failure)
                relay = runtime._transport._failed_relay
                self.assertIsNotNone(relay)
                self.assertFalse(relay.executor._shutdown)
                self.assertTrue(
                    any(slot.cpu_future is not None and not slot.cpu_future.done() for slot in relay.slots)
                )
                self.assertTrue(all(reference() is not None for reference in allocations["gpu_slots"]))
                self.assertGreater(runtime.metrics["host_reserved_bytes"], 0)
                self.assertGreater(runtime.metrics["transport_reserved_bytes"], 0)
                self.assertTrue(runtime._poisoned)
                self.assertEqual(fake.events[-2:], ["admission", "broadcast:torch.int64"])
                self.assertEqual(fake.works, [])
                events = list(fake.events)
                with self.assertRaises(MediaTransportError) as repeated:
                    runtime._transport.deliver(None, "no-reuse", _Mesh(), "cuda:0", lambda: None)
                self.assertTrue(repeated.exception.fail_stop)
                self.assertEqual(fake.events, events)
            finally:
                release.set()

    def test_chunked_host_or_pinned_refusal_is_shared_before_payload(self):
        plan, media = self._chunked_fixture()
        for refusal in ("host", "pinned", "remote", "pool"):
            with self.subTest(refusal=refusal):
                fake = _Collectives(reject=refusal == "remote")
                charges = []

                def reserve_host(amount):
                    self.assertNotIn("admission", fake.events)
                    charges.append(amount)
                    if refusal == "host":
                        raise MediaBudgetError("fixture pinned host budget refused")

                with (
                    self._chunked_cuda(fake, fail_pinned=refusal == "pinned") as allocations,
                    (
                        patch.object(
                            transport_module, "ThreadPoolExecutor", side_effect=RuntimeError("fixture pool refusal")
                        )
                        if refusal == "pool"
                        else nullcontext()
                    ),
                    patch.object(transport_module._RawPixelRelay, "transfer") as transfer,
                    self.assertRaises(MediaTransportError) as raised,
                ):
                    MediaTransport(self.config).deliver(
                        plan, "refusal", _Mesh(), "cuda:0", lambda: media, reserve_host_bytes=reserve_host
                    )
                self.assertEqual(charges, [256])
                self.assertEqual(allocations["cpu_submissions"], [])
                self.assertTrue(all(pool._shutdown for pool in allocations["executors"]))
                self.assertFalse(raised.exception.fail_stop)
                transfer.assert_not_called()
                self.assertEqual(fake.works, [])
                self.assertEqual(fake.events[-1], "admission")

    def test_chunked_work_failure_retains_bounded_staging_and_is_fail_stop(self):
        plan, media = self._chunked_fixture()
        failure = RuntimeError("fixture async relay failure")
        fake = _Collectives(work_error=failure)
        transport = MediaTransport(self.config)
        with self._chunked_cuda(fake) as allocations:
            with self.assertRaises(MediaTransportError) as raised:
                transport.deliver(
                    plan, "failed-work", _Mesh(), "cuda:0", lambda: media, reserve_host_bytes=lambda _: None
                )
            self.assertTrue(raised.exception.fail_stop)
            self.assertIs(raised.exception.__cause__, failure)
            self.assertIsNotNone(transport._failed_relay)
            self.assertTrue(all(reference() is not None for reference in allocations["gpu_slots"]))
            events = list(fake.events)
            with self.assertRaises(MediaTransportError) as repeated:
                transport.deliver(None, "no-reuse", _Mesh(), "cuda:0", lambda: None)
            self.assertTrue(repeated.exception.fail_stop)
            self.assertEqual(fake.events, events)

    def test_chunked_runtime_charges_pinned_host_bytes_and_balances_shared_refusal(self):
        plan, media = self._chunked_fixture()
        with MediaRuntime(self.config) as probe:
            required = probe._materializer.reservation_bytes(plan)
        for headroom in (0, 256):
            with self.subTest(headroom=headroom):
                fake = _Collectives()
                config = configuration(
                    distribution="sp_leader", max_transport_bytes=2**20, max_inflight_bytes=required + headroom
                )
                with (
                    MediaRuntime(config) as runtime,
                    self._chunked_cuda(fake),
                    patch.object(runtime._materializer, "materialize", return_value=media),
                ):
                    if headroom:
                        with runtime.consume(plan, "host-charge", _Mesh(), device="cuda:0") as delivered:
                            self.assertEqual(runtime.metrics["host_reserved_bytes"], required + 256)
                            self.assertIs(delivered.tensors["pixel_values"], media.tensors["pixel_values"])
                    else:
                        with self.assertRaises(MediaTransportError) as raised:
                            with runtime.consume(plan, "host-refusal", _Mesh(), device="cuda:0"):
                                self.fail("unreserved pinned staging was admitted")
                        self.assertFalse(raised.exception.fail_stop)
                        self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
                        self.assertEqual(fake.works, [])
                    self.assertEqual(runtime.metrics["host_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["transport_reserved_bytes"], 0)
                    self.assertEqual(runtime.metrics["active_consumptions"], 0)

    def test_local_delivery_preserves_raw_pixel_bytes_and_only_moves_nonhost_fields(self):
        for pixel_field in ("pixel_values", "pixel_values_videos"):
            with self.subTest(pixel_field=pixel_field):
                pixels = torch.tensor([[1.0001, -0.0], [float("nan"), -3.1415927]], dtype=torch.float32)
                grid = self.cpu.tensors["image_grid_thw"]
                media = MaterializedMedia(
                    {pixel_field: pixels, "image_grid_thw": grid},
                    metrics={"processed_entries": 1},
                    metadata={"source": "fixture"},
                )
                plan = replace(
                    self.plan,
                    output_specs=(TensorSpec(pixel_field, "float32", (2, 2)), self.plan.output_specs[1]),
                )
                transfers = []

                def move(tensor, device):
                    self.assertIs(tensor, grid)
                    transfers.append(torch.device(device))
                    return tensor.clone()

                def unexpected_reservation(amount):
                    self.fail("local placement must not request SP transport or receiver-host reservations")

                fake = _Collectives()
                transport = MediaTransport(configuration())
                with patch.object(transport_module, "dist", fake), patch.object(torch.Tensor, "to", new=move):
                    result = transport.deliver(
                        plan,
                        "local",
                        _Mesh(),
                        "cuda:0",
                        lambda: media,
                        reserve_bytes=unexpected_reservation,
                        reserve_host_bytes=unexpected_reservation,
                    )
                result.validate(plan.output_specs)
                self.assertIs(result.tensors[pixel_field], pixels)
                self.assertEqual(result.tensors[pixel_field].device.type, "cpu")
                self.assertTrue(torch.equal(result.tensors[pixel_field].view(torch.uint8), pixels.view(torch.uint8)))
                self.assertEqual(result.tensors["image_grid_thw"].dtype, grid.dtype)
                self.assertEqual(result.tensors["image_grid_thw"].shape, grid.shape)
                self.assertTrue(
                    torch.equal(result.tensors["image_grid_thw"].view(torch.uint8), grid.view(torch.uint8))
                )
                self.assertEqual(transfers, [torch.device("cuda:0")])
                self.assertEqual(result.nbytes, media.nbytes)
                self.assertIs(result.metrics, media.metrics)
                self.assertIs(result.metadata, media.metadata)
                self.assertIs(media.tensors["image_grid_thw"], grid)
                self.assertEqual(fake.events, [])

    def test_local_empty_skips_producer_and_device_work(self):
        prepared = []
        fake = _Collectives()
        with (
            patch.object(transport_module, "dist", fake),
            patch.object(torch.Tensor, "to", side_effect=AssertionError("empty local delivery moved a tensor")),
        ):
            result = MediaTransport(configuration()).deliver(
                None,
                "empty-local",
                _Mesh(),
                "cuda:0",
                lambda: self.fail("empty local delivery invoked producer"),
                prepare_control=lambda producer, reserve: prepared.append((producer, reserve)),
            )
        self.assertIsNone(result)
        self.assertEqual(prepared, [(True, False)])
        self.assertEqual(fake.events, [])

    def test_named_payload_order_real_leader_mapping_and_follower_no_processing(self):
        leader = _Collectives(rank=0, leader=8)
        mesh = _Mesh()
        transport = MediaTransport(self.config)
        with patch.object(transport_module, "dist", leader):
            result = transport.deliver(self.plan, "pack-a/micro-0", mesh, "cpu", lambda: self.cpu)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        with patch.object(transport_module, "dist", follower):
            received = transport.deliver(
                self.plan, "pack-a/micro-0", mesh, "cpu", lambda: self.fail("follower invoked producer")
            )
        for name, tensor in result.tensors.items():
            self.assertTrue(torch.equal(tensor, received.tensors[name]))
        self.assertEqual(received.metadata, {"source": "fixture"})
        self.assertEqual(result.metrics["sp_producer"], 1)
        self.assertEqual(received.metrics["sp_producer"], 0)
        self.assertEqual(list(result.tensors), ["image_grid_thw", "pixel_values"])

    def test_cuda_staging_preserves_native_pixel_cpu_and_grid_target_placement(self):
        self.cpu.tensors["pixel_values"] = torch.tensor(
            [[1.0001, -0.0], [float("nan"), -3.1415927]], dtype=torch.float32
        )
        original_empty = torch.empty
        original_zeros = torch.zeros
        staging = []
        target_devices = {}

        def allocate(*args, **kwargs):
            requested = torch.device(kwargs.get("device") or "cpu")
            tensor = original_empty(*args, **{**kwargs, "device": "cpu"})
            target_devices[id(tensor)] = requested
            if (
                requested.type == "cuda"
                and tuple(tensor.shape) == tuple(self.cpu.tensors["pixel_values"].shape)
                and tensor.dtype == torch.float32
            ):
                staging.append(weakref.ref(tensor))
            return tensor

        def zeros(*args, **kwargs):
            return original_zeros(*args, **{**kwargs, "device": "cpu"})

        leader = _Collectives(rank=0, leader=8)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        for fake in (leader, follower):
            with self.subTest(rank=fake.rank):
                host_charges = []

                def reserve_host(amount):
                    self.assertNotIn("admission", fake.events)
                    host_charges.append(amount)

                def produce():
                    if fake.rank != 0:
                        self.fail("follower invoked producer")
                    return self.cpu

                with (
                    patch.object(transport_module, "dist", fake),
                    patch.object(transport_module.torch, "empty", new=allocate),
                    patch.object(transport_module.torch, "zeros", new=zeros),
                ):
                    result = MediaTransport(self.config).deliver(
                        self.plan, "native-cpu", _Mesh(), "cuda:0", produce, reserve_host_bytes=reserve_host
                    )
                pixel_bytes = (
                    self.cpu.tensors["pixel_values"].numel() * self.cpu.tensors["pixel_values"].element_size()
                )
                self.assertEqual(host_charges, [0 if fake.rank == 0 else pixel_bytes])
                self.assertEqual(len(staging), fake.rank + 1)
                self.assertTrue(all(reference() is None for reference in staging))
                self.assertEqual(result.nbytes, self.plan.output_bytes)
                self.assertEqual(result.metadata, self.cpu.metadata)
                for name, expected in self.cpu.tensors.items():
                    delivered = result.tensors[name]
                    self.assertEqual(delivered.dtype, expected.dtype)
                    self.assertEqual(tuple(delivered.shape), tuple(expected.shape))
                    self.assertEqual(
                        delivered.numel() * delivered.element_size(), expected.numel() * expected.element_size()
                    )
                    self.assertTrue(torch.equal(delivered.view(torch.uint8), expected.view(torch.uint8)))
                    if name == "pixel_values":
                        self.assertEqual(delivered.device.type, "cpu")
                        if fake.rank == 0:
                            self.assertIs(delivered, expected)
                        else:
                            self.assertIsNot(delivered, expected)
                    else:
                        self.assertEqual(target_devices[id(delivered)], torch.device("cuda:0"))
                self.assertEqual(
                    fake.events,
                    [
                        "fingerprint",
                        "broadcast:torch.int64",
                        "broadcast:torch.uint8",
                        "admission",
                        "broadcast:torch.int64",
                        "broadcast:torch.float32",
                    ],
                )

    def test_follower_cpu_pixel_allocation_failure_is_shared_before_payload(self):
        leader = _Collectives(rank=0, leader=8)
        with patch.object(transport_module, "dist", leader):
            MediaTransport(self.config).deliver(self.plan, "cpu-allocation", _Mesh(), "cpu", lambda: self.cpu)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        original_empty = torch.empty
        original_zeros = torch.zeros
        failed_allocations = []

        def allocate(*args, **kwargs):
            requested = torch.device(kwargs.get("device") or "cpu")
            if requested.type == "cpu" and kwargs.get("dtype") == torch.float32:
                failed_allocations.append(args)
                raise MemoryError("fixture CPU receiver allocation failure")
            return original_empty(*args, **{**kwargs, "device": "cpu"})

        def zeros(*args, **kwargs):
            return original_zeros(*args, **{**kwargs, "device": "cpu"})

        with (
            patch.object(transport_module, "dist", follower),
            patch.object(transport_module.torch, "empty", new=allocate),
            patch.object(transport_module.torch, "zeros", new=zeros),
            self.assertRaises(MediaTransportError) as raised,
        ):
            MediaTransport(self.config).deliver(
                self.plan, "cpu-allocation", _Mesh(), "cuda:0", lambda: self.fail("follower invoked producer")
            )
        self.assertEqual(len(failed_allocations), 1)
        self.assertFalse(raised.exception.fail_stop)
        self.assertIsInstance(raised.exception.__cause__, MemoryError)
        self.assertEqual(
            follower.events, ["fingerprint", "broadcast:torch.int64", "broadcast:torch.uint8", "admission"]
        )

        remote_refusal = _Collectives(rank=0, leader=8, reject=True)
        with (
            patch.object(transport_module, "dist", remote_refusal),
            patch.object(MediaTransport, "_copy_payload") as copy_payload,
            self.assertRaises(MediaTransportError) as raised,
        ):
            MediaTransport(self.config).deliver(self.plan, "remote-cpu-refusal", _Mesh(), "cpu", lambda: self.cpu)
        self.assertFalse(raised.exception.fail_stop)
        copy_payload.assert_not_called()
        self.assertEqual(
            remote_refusal.events, ["fingerprint", "broadcast:torch.int64", "broadcast:torch.uint8", "admission"]
        )

    def test_follower_host_reservation_failure_is_shared_before_payload(self):
        leader = _Collectives(rank=0, leader=8)
        with patch.object(transport_module, "dist", leader):
            MediaTransport(self.config).deliver(self.plan, "host-allocation", _Mesh(), "cpu", lambda: self.cpu)
        follower = _Collectives(rank=1, leader=8, wire=leader.wire)
        host_charges = []

        def reserve_host(amount):
            self.assertNotIn("admission", follower.events)
            host_charges.append(amount)
            raise MediaBudgetError("fixture host reservation refusal")

        with patch.object(transport_module, "dist", follower), self.assertRaises(MediaTransportError) as raised:
            MediaTransport(self.config).deliver(
                self.plan,
                "host-allocation",
                _Mesh(),
                "cpu",
                lambda: self.fail("follower invoked producer"),
                reserve_host_bytes=reserve_host,
            )
        pixel_bytes = next(spec.nbytes for spec in self.plan.output_specs if spec.name == "pixel_values")
        self.assertEqual(host_charges, [pixel_bytes])
        self.assertFalse(raised.exception.fail_stop)
        self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
        self.assertEqual(
            follower.events, ["fingerprint", "broadcast:torch.int64", "broadcast:torch.uint8", "admission"]
        )

    def test_empty_uses_status_and_admission_but_no_producer_or_payload(self):
        fake = _Collectives()
        with patch.object(transport_module, "dist", fake):
            result = MediaTransport(self.config).deliver(
                None, "text", _Mesh(), "cpu", lambda: self.fail("EMPTY invoked producer")
            )
        self.assertIsNone(result)
        self.assertEqual(fake.events, ["fingerprint", "broadcast:torch.int64", "admission"])

    def test_sp_size_one_restores_native_field_placement_without_collectives(self):
        fake = _Collectives()
        mesh = _Mesh()
        grid = self.cpu.tensors["image_grid_thw"]
        transfers = []

        def move(tensor, device):
            self.assertIs(tensor, grid)
            transfers.append(torch.device(device))
            return tensor.clone()

        with (
            patch.object(mesh, "size", return_value=1),
            patch.object(transport_module, "dist", fake),
            patch.object(torch.Tensor, "to", new=move),
        ):
            result = MediaTransport(self.config).deliver(self.plan, "one", mesh, "cuda:0", lambda: self.cpu)
        self.assertIs(result.tensors["pixel_values"], self.cpu.tensors["pixel_values"])
        self.assertTrue(torch.equal(result.tensors["image_grid_thw"], grid))
        self.assertEqual(transfers, [torch.device("cuda:0")])
        self.assertEqual(fake.events, [])

    def test_transport_reservation_includes_full_payload_and_control_storage(self):
        fake = _Collectives()
        charges = []
        with patch.object(transport_module, "dist", fake):
            result = MediaTransport(self.config).deliver(
                self.plan, "pack", _Mesh(), "cpu", lambda: self.cpu, charges.append
            )
        self.assertEqual(charges, [self.config.max_header_bytes + 33 * 3 + 32, self.plan.output_bytes])
        self.assertEqual(sum(charges), result.metrics["transport_reserved_bytes"])

    def test_bounded_metadata_and_payload_budget_reject_before_h2d(self):
        fake = _Collectives()
        self.cpu.metadata = {"oversized": "x" * self.config.max_header_bytes}
        with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
            MediaTransport(self.config).deliver(self.plan, "pack", _Mesh(), "cpu", lambda: self.cpu)
        self.assertNotIn("admission", fake.events)
        self.cpu.metadata = {}
        fake = _Collectives()
        config = configuration(
            distribution="sp_leader", max_transport_bytes=1024 + 33 * 3 + 32 + self.plan.output_bytes - 1
        )
        with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
            MediaTransport(config).deliver(self.plan, "pack", _Mesh(), "cpu", lambda: self.cpu)
        self.assertEqual(fake.events[-1], "admission")

    def test_fingerprint_mismatch_prevents_producer(self):
        fake = _Collectives(mismatch=True)
        with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
            MediaTransport(self.config).deliver(
                self.plan, "pack", _Mesh(), "cpu", lambda: self.fail("mismatch invoked producer")
            )
        self.assertEqual(fake.events, ["fingerprint"])

    def test_any_rank_control_refusal_is_shared_before_producer(self):
        for rejecting_rank in (0, 1):
            for rank in (0, 1):
                with self.subTest(rank=rank, rejecting_rank=rejecting_rank):
                    fake = _Collectives(rank=rank, control_error_rank=rejecting_rank)

                    def prepare_control(is_producer, reserve_control):
                        self.assertEqual(is_producer, rank == 0)
                        self.assertTrue(reserve_control)
                        if rank == rejecting_rank:
                            raise MediaBudgetError("fixture local control reservation refused")

                    with (
                        patch.object(transport_module, "dist", fake),
                        self.assertRaises(MediaTransportError) as raised,
                    ):
                        MediaTransport(self.config).deliver(
                            self.plan,
                            "pack",
                            _Mesh(),
                            "cpu",
                            lambda: self.fail("control refusal invoked producer"),
                            prepare_control=prepare_control,
                        )
                    self.assertFalse(raised.exception.fail_stop)
                    self.assertIn("plan/control validation failed before producer execution", str(raised.exception))
                    if rank == rejecting_rank:
                        self.assertIsInstance(raised.exception.__cause__, MediaBudgetError)
                    else:
                        self.assertIsNone(raised.exception.__cause__)
                    self.assertEqual(fake.events, ["fingerprint"])
                    self.assertEqual(fake.wire, [])

    def test_producer_error_and_allocation_refusal_prevent_payload(self):
        fake = _Collectives()

        def failed():
            raise ValueError("fixture CPU failure")

        with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError) as raised:
            MediaTransport(self.config).deliver(self.plan, "pack", _Mesh(), "cpu", failed)
        self.assertFalse(raised.exception.fail_stop)
        self.assertNotIn("admission", fake.events)
        fake = _Collectives(reject=True)
        with patch.object(transport_module, "dist", fake), self.assertRaises(MediaTransportError):
            MediaTransport(self.config).deliver(self.plan, "pack", _Mesh(), "cpu", lambda: self.cpu)
        self.assertEqual(fake.events[-1], "admission")

    def test_h2d_is_after_ack_and_failure_is_fail_stop(self):
        fake = _Collectives()

        def failed_copy(destination, source):
            self.assertEqual(fake.events[-1], "admission")
            raise RuntimeError("injected post-ACK H2D failure")

        with (
            patch.object(transport_module, "dist", fake),
            patch.object(MediaTransport, "_copy_payload", side_effect=failed_copy),
            self.assertRaises(MediaTransportError) as raised,
        ):
            MediaTransport(self.config).deliver(self.plan, "pack", _Mesh(), "cpu", lambda: self.cpu)
        self.assertTrue(raised.exception.fail_stop)
        self.assertEqual(fake.events[-1], "admission")


@unittest.skipUnless(os.environ.get("XTUNER_TEST_MEDIA_NCCL") == "1", "main-guard opt-in two-rank NCCL test")
class TestMediaChunkedRelayNCCL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if int(os.environ.get("WORLD_SIZE", "0")) != 2:
            raise RuntimeError("run this opt-in test with torchrun --nproc-per-node=2")
        cls.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.local_rank)
        torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()

    def test_two_rank_bit_exact_roundtrip_empty_and_shared_refusal(self):
        chunk_bytes = 64 * 1024
        rank = torch.distributed.get_rank()
        device = torch.device("cuda", self.local_rank)
        mesh = _Mesh()
        mesh.group = torch.distributed.group.WORLD
        bits = torch.tensor([0, -2147483648, 1065353217, 2139095040, 2143289345, -4194303, 1], dtype=torch.int32)
        pixel_count = (17 * chunk_bytes + 64) // 4
        video_count = (18 * chunk_bytes + 16) // 4
        pixels = bits.repeat((pixel_count + 6) // 7)[:pixel_count].clone().view(torch.float32)
        videos = bits.flip(0).repeat((video_count + 6) // 7)[:video_count].clone().view(torch.float32)
        specs = (
            TensorSpec("pixel_values", "float32", (pixel_count,)),
            TensorSpec("pixel_values_videos", "float32", (video_count,)),
            TensorSpec("image_grid_thw", "int64", (1, 3)),
        )
        base = make_plan(count=1)
        entry = replace(base.entries[0], output_specs=specs, estimated_peak_bytes=sum(spec.nbytes for spec in specs))
        plan = replace(base, entries=(entry,), output_specs=specs)
        expected = MaterializedMedia(
            {"pixel_values": pixels, "pixel_values_videos": videos, "image_grid_thw": torch.tensor([[1, 1, 2]])},
            metadata={"source": "two-rank-nccl-bits"},
        )
        self.assertEqual(plan.output_bytes, expected.nbytes)
        self.assertEqual(entry.estimated_peak_bytes, expected.nbytes)
        config = configuration(distribution="sp_leader", max_transport_bytes=8 * 1024**2)
        charges = []

        def produce():
            self.assertEqual(rank, 0)
            return expected

        with patch.object(transport_module, "_RAWPIXEL_CHUNK_BYTES", chunk_bytes):
            result = MediaTransport(config).deliver(
                plan, "real-chunks", mesh, device, produce, reserve_host_bytes=charges.append
            )
            result.validate(specs)
            self.assertEqual(charges, [16 * chunk_bytes + (0 if rank == 0 else pixels.nbytes + videos.nbytes)])
            for name, tensor in result.tensors.items():
                self.assertTrue(torch.equal(tensor.cpu().view(torch.uint8), expected.tensors[name].view(torch.uint8)))
                self.assertEqual(
                    tensor.device.type, "cpu" if name in transport_module._HOST_PAYLOAD_FIELDS else "cuda"
                )
                if rank == 0 and name in transport_module._HOST_PAYLOAD_FIELDS:
                    self.assertIs(tensor, expected.tensors[name])
            self.assertEqual(result.metadata, expected.metadata)
            self.assertIsNone(
                MediaTransport(config).deliver(
                    None, "real-empty", mesh, device, lambda: self.fail("EMPTY invoked producer")
                )
            )

            def refuse_host(amount):
                if rank == 1:
                    raise MediaBudgetError("two-rank pinned admission refusal")

            with self.assertRaises(MediaTransportError) as raised:
                MediaTransport(config).deliver(
                    plan, "real-refusal", mesh, device, produce, reserve_host_bytes=refuse_host
                )
            self.assertFalse(raised.exception.fail_stop)


if __name__ == "__main__":
    unittest.main()
