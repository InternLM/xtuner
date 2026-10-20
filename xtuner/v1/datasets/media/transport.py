import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

from .config import MediaPipelineConfig
from .materializer import MediaBudgetError, MediaPipelineError, validate_cpu_media
from .schema import MaterializedMedia, MediaPlan, TensorSpec, canonical_json


_HOST_PAYLOAD_FIELDS = frozenset(("pixel_values", "pixel_values_videos"))
_RAWPIXEL_CHUNK_BYTES = 64 * 1024**2
_RAWPIXEL_RELAY_SLOTS = 16
_RAWPIXEL_COPY_WORKERS = 8


class MediaTransportError(MediaPipelineError):
    def __init__(self, message: str, *, fail_stop: bool = False):
        super().__init__(message)
        self.fail_stop = fail_stop


@dataclass
class _RelaySlot:
    device: torch.Tensor
    pinned: torch.Tensor
    ready: torch.cuda.Event
    done: torch.cuda.Event
    work: dist.Work | None = None
    cpu_future: Future | None = None
    offset: int = 0
    size: int = 0


class _RawPixelRelay:
    def __init__(self, target: torch.device):
        self.owner_stream = torch.cuda.current_stream(target)
        self.copy_stream = torch.cuda.Stream(device=target)
        self.slots = []
        with torch.cuda.device(target):
            for _ in range(_RAWPIXEL_RELAY_SLOTS):
                slot = _RelaySlot(
                    torch.empty(_RAWPIXEL_CHUNK_BYTES, dtype=torch.uint8, device=target),
                    torch.empty(_RAWPIXEL_CHUNK_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True),
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                )
                slot.ready.record(self.owner_stream)
                slot.done.record(self.owner_stream)
                slot.done.synchronize()
                self.slots.append(slot)
        self.copy_stream.wait_stream(self.owner_stream)
        self.executor = ThreadPoolExecutor(
            max_workers=_RAWPIXEL_COPY_WORKERS, thread_name_prefix="xtuner-media-relay-copy"
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def transfer(self, host: torch.Tensor, rank: int, source: int, group) -> None:
        raw = host.view(torch.uint8).reshape(-1)

        def join_copy(slot: _RelaySlot) -> None:
            if slot.cpu_future is not None:
                slot.cpu_future.result()
                slot.cpu_future = None

        def prepare(slot: _RelaySlot, offset: int) -> None:
            slot.offset = offset
            slot.size = min(_RAWPIXEL_CHUNK_BYTES, raw.numel() - offset)
            slot.cpu_future = self.executor.submit(
                slot.pinned[: slot.size].copy_, raw[offset : offset + slot.size], non_blocking=False
            )

        def retire(slot: _RelaySlot) -> None:
            if slot.work is None:
                return
            slot.done.synchronize()
            with torch.cuda.stream(self.owner_stream):
                if not slot.work.wait():
                    raise RuntimeError("rawpixel relay work was aborted")
            if rank != 0:
                slot.cpu_future = self.executor.submit(
                    raw[slot.offset : slot.offset + slot.size].copy_, slot.pinned[: slot.size], non_blocking=False
                )
            slot.work = None

        if rank == 0:
            for ordinal, slot in enumerate(self.slots):
                offset = ordinal * _RAWPIXEL_CHUNK_BYTES
                if offset < raw.numel():
                    prepare(slot, offset)
        previous = None
        for ordinal, offset in enumerate(range(0, raw.numel(), _RAWPIXEL_CHUNK_BYTES)):
            slot = self.slots[ordinal % _RAWPIXEL_RELAY_SLOTS]
            join_copy(slot)
            if rank != 0:
                slot.offset = offset
                slot.size = min(_RAWPIXEL_CHUNK_BYTES, raw.numel() - offset)
            staging = slot.device[: slot.size]
            if rank == 0:
                with torch.cuda.stream(self.copy_stream):
                    staging.copy_(slot.pinned[: slot.size], non_blocking=True)
                    slot.ready.record(self.copy_stream)
            with torch.cuda.stream(self.owner_stream):
                if rank == 0:
                    self.owner_stream.wait_event(slot.ready)
                slot.work = dist.broadcast(staging, src=source, group=group, async_op=True)
                if rank == 0:
                    slot.work.block_current_stream()
                    slot.done.record(self.owner_stream)
            if rank != 0:
                with torch.cuda.stream(self.copy_stream):
                    slot.work.block_current_stream()
                    slot.pinned[: slot.size].copy_(staging, non_blocking=True)
                    slot.done.record(self.copy_stream)
            if previous is not None:
                retire(previous)
                next_offset = previous.offset + _RAWPIXEL_RELAY_SLOTS * _RAWPIXEL_CHUNK_BYTES
                if rank == 0 and next_offset < raw.numel():
                    prepare(previous, next_offset)
            previous = slot
        if previous is not None:
            retire(previous)
        for slot in self.slots:
            join_copy(slot)


class MediaTransport:
    def __init__(self, config: MediaPipelineConfig):
        self.config = config
        self._failed_relay = None

    def group_info(self, sp_mesh):
        if self.config.distribution != "sp_leader" or sp_mesh is None or sp_mesh.size() == 1:
            return None, 1, 0, None
        group = sp_mesh.get_group()
        return group, dist.get_world_size(group), dist.get_rank(group), dist.get_global_rank(group, 0)

    def is_producer(self, sp_mesh) -> bool:
        return self.group_info(sp_mesh)[2] == 0

    def _schema(self, plan: MediaPlan, metadata: dict) -> bytes:
        if len(plan.output_specs) > self.config.max_tensor_fields:
            raise MediaBudgetError("transport field count exceeds limit")
        if any(len(spec.shape) > self.config.max_tensor_rank for spec in plan.output_specs):
            raise MediaBudgetError("transport tensor rank exceeds limit")
        encoded = canonical_json(
            {
                "protocol_version": 1,
                "specs": [asdict(spec) for spec in sorted(plan.output_specs, key=lambda value: value.name)],
                "metadata": metadata,
            }
        ).encode("utf-8")
        if len(encoded) > self.config.max_header_bytes:
            raise MediaBudgetError("transport schema exceeds header byte limit")
        return encoded

    def deliver(
        self,
        plan: MediaPlan | None,
        key: str,
        sp_mesh,
        device,
        produce: Callable[[], MaterializedMedia],
        reserve_bytes: Callable[[int], None] | None = None,
        prepare_control: Callable[[bool, bool], None] | None = None,
        reserve_host_bytes: Callable[[int], None] | None = None,
    ) -> MaterializedMedia | None:
        if self._failed_relay is not None:
            raise MediaTransportError("rawpixel relay failed; launcher fail-stop required", fail_stop=True)
        group, size, rank, source = self.group_info(sp_mesh)
        if size == 1:
            if prepare_control is not None:
                prepare_control(True, False)
            if plan is None:
                return None
            media = produce()
            return MaterializedMedia(
                tensors={
                    name: tensor if name in _HOST_PAYLOAD_FIELDS else tensor.to(device)
                    for name, tensor in media.tensors.items()
                },
                metrics=media.metrics,
                metadata=media.metadata,
            )
        try:
            return self._deliver_sp(
                plan,
                key,
                group,
                size,
                rank,
                source,
                device,
                produce,
                reserve_bytes,
                prepare_control,
                reserve_host_bytes,
            )
        except MediaTransportError:
            raise
        except Exception as error:
            raise MediaTransportError(
                "SP control/CUDA/process-group failure; launcher fail-stop required", fail_stop=True
            ) from error

    @staticmethod
    def _copy_payload(destination: torch.Tensor, source: torch.Tensor) -> None:
        destination.copy_(source, non_blocking=False)

    def _deliver_sp(
        self,
        plan,
        key,
        group,
        size,
        rank,
        source,
        device,
        produce,
        reserve_bytes,
        prepare_control,
        reserve_host_bytes,
    ) -> MaterializedMedia | None:
        control_bytes = self.config.max_header_bytes + 33 * (size + 1) + 32
        if control_bytes > self.config.max_transport_bytes:
            raise MediaTransportError("SP bootstrap control exceeds transport limit", fail_stop=True)
        if reserve_bytes is not None:
            reserve_bytes(control_bytes)
        target = torch.device(device)
        control = torch.zeros(33, dtype=torch.uint8, device=target)
        status = torch.zeros(3, dtype=torch.int64, device=target)
        admission = torch.zeros(1, dtype=torch.int64, device=target)
        gathered = [torch.empty_like(control) for _ in range(size)]
        schema_buffer = None
        local_error = None
        try:
            if prepare_control is not None:
                prepare_control(rank == 0, True)
            if not isinstance(key, str) or not key or len(key.encode("utf-8")) > self.config.max_header_bytes:
                raise ValueError("SP consumption key must be a bounded nonempty string")
            if plan is not None:
                self._schema(plan, {})
            fingerprint = hashlib.sha256(
                canonical_json(
                    {
                        "key": key,
                        "plan": plan.fingerprint if plan is not None else None,
                        "config": self.config.model_dump(),
                        "protocol_version": 1,
                    }
                ).encode("utf-8")
            ).digest()
            schema_buffer = torch.empty(self.config.max_header_bytes, dtype=torch.uint8, device=target)
            control[1:].copy_(torch.frombuffer(bytearray(fingerprint), dtype=torch.uint8))
        except Exception as error:
            local_error = error
            control[0] = 1
        dist.all_gather(gathered, control, group=group)
        if any(int(item[0].item()) for item in gathered):
            raise MediaTransportError("SP plan/control validation failed before producer execution") from local_error
        if any(not torch.equal(item[1:], gathered[0][1:]) for item in gathered[1:]):
            raise MediaTransportError("SP plan, configuration or consumption schedule fingerprints disagree")

        cpu_media = None
        encoded = b""
        producer_error = None
        if rank == 0:
            try:
                if plan is None:
                    status[0] = 0
                else:
                    cpu_media = produce()
                    validate_cpu_media(cpu_media, plan.output_specs)
                    encoded = self._schema(plan, cpu_media.metadata)
                    status[0] = 1
                    status[1] = len(encoded)
                    status[2] = cpu_media.nbytes
            except Exception as error:
                producer_error = error
                cpu_media = None
                status.zero_()
                status[0] = 2
        dist.broadcast(status, src=source, group=group)
        result_status, header_length, payload_bytes = (int(value) for value in status.cpu().tolist())
        if result_status == 2:
            raise MediaTransportError("SP leader CPU materialization or schema validation failed") from producer_error
        if result_status not in (0, 1) or not 0 <= header_length <= self.config.max_header_bytes:
            raise MediaTransportError("invalid SP result status or bounded header length")
        if result_status == 0 and (plan is not None or header_length or payload_bytes):
            raise MediaTransportError("EMPTY result contradicts the agreed media plan")
        if result_status == 1 and (plan is None or payload_bytes != plan.output_bytes or header_length == 0):
            raise MediaTransportError("SP payload length contradicts the agreed media plan")

        if header_length:
            if rank == 0:
                schema_buffer[:header_length].copy_(torch.frombuffer(bytearray(encoded), dtype=torch.uint8))
            dist.broadcast(schema_buffer[:header_length], src=source, group=group)
        outputs: dict[str, torch.Tensor] = {}
        host_outputs: dict[str, torch.Tensor] = {}
        metadata = {}
        specs: tuple[TensorSpec, ...] = ()
        chunked_fields = set()
        relay = None
        allocation_error = None
        try:
            if result_status == 1:
                document = json.loads(schema_buffer[:header_length].cpu().numpy().tobytes().decode("utf-8"))
                if set(document) != {"protocol_version", "specs", "metadata"} or document["protocol_version"] != 1:
                    raise ValueError("unsupported or malformed SP schema envelope")
                if not isinstance(document["specs"], list) or len(document["specs"]) > self.config.max_tensor_fields:
                    raise MediaBudgetError("SP schema field count exceeds bounds")
                specs = tuple(TensorSpec.from_dict(spec) for spec in document["specs"])
                expected = tuple(sorted(plan.output_specs, key=lambda value: value.name))
                if specs != expected or any(len(spec.shape) > self.config.max_tensor_rank for spec in specs):
                    raise ValueError("SP named schema differs from agreed plan")
                metadata = document["metadata"]
                if not isinstance(metadata, dict):
                    raise ValueError("SP metadata must be a bounded JSON object")
            required = payload_bytes + control_bytes
            if required > self.config.max_transport_bytes:
                raise MediaBudgetError("full per-rank SP payload and control exceed transport budget")
            if reserve_bytes is not None:
                reserve_bytes(payload_bytes)
            chunked_fields = {
                spec.name
                for spec in specs
                if target.type == "cuda"
                and spec.name in _HOST_PAYLOAD_FIELDS
                and spec.nbytes > _RAWPIXEL_RELAY_SLOTS * _RAWPIXEL_CHUNK_BYTES
            }
            if reserve_host_bytes is not None:
                host_bytes = sum(spec.nbytes for spec in specs if spec.name in _HOST_PAYLOAD_FIELDS)
                pinned_bytes = _RAWPIXEL_RELAY_SLOTS * _RAWPIXEL_CHUNK_BYTES if chunked_fields else 0
                reserve_host_bytes((0 if rank == 0 else host_bytes) + pinned_bytes)
            if chunked_fields:
                relay = _RawPixelRelay(target)
            for spec in specs:
                if spec.name not in chunked_fields:
                    outputs[spec.name] = torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device=target)
                if spec.name in _HOST_PAYLOAD_FIELDS:
                    host_outputs[spec.name] = (
                        cpu_media.tensors[spec.name]
                        if rank == 0
                        else torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device="cpu")
                    )
                else:
                    host_outputs[spec.name] = outputs[spec.name]
            admission.fill_(1)
        except Exception as error:
            allocation_error = error
            admission.zero_()
        dist.all_reduce(admission, op=dist.ReduceOp.MIN, group=group)
        if not int(admission.item()):
            if relay is not None:
                relay.close()
            outputs.clear()
            host_outputs.clear()
            raise MediaTransportError(
                "SP schema or destination allocation rejected before payload"
            ) from allocation_error

        try:
            for spec in specs:
                if spec.name in chunked_fields:
                    relay.transfer(host_outputs[spec.name], rank, source, group)
                    continue
                output = outputs[spec.name]
                if rank == 0:
                    self._copy_payload(output, cpu_media.tensors[spec.name])
                if output.numel():
                    dist.broadcast(output, src=source, group=group)
                if rank != 0 and spec.name in _HOST_PAYLOAD_FIELDS:
                    host_outputs[spec.name].copy_(output, non_blocking=False)
            if relay is not None:
                relay.close()
        except Exception as error:
            self._failed_relay = relay
            raise MediaTransportError(
                "post-admission H2D/CUDA/payload failure; launcher fail-stop required", fail_stop=True
            ) from error
        if result_status == 0:
            return None
        metrics = dict(cpu_media.metrics) if rank == 0 else {}
        metrics.update(
            transport_bytes=payload_bytes,
            transport_reserved_bytes=payload_bytes + control_bytes,
            sp_producer=int(rank == 0),
        )
        outputs.clear()
        return MaterializedMedia(tensors=host_outputs, metrics=metrics, metadata=metadata)
