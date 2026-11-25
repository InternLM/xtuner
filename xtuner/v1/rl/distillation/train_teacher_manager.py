from __future__ import annotations

import gc
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
import torch.distributed as dist

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss import LogProbConfig, LogProbContext, TopKLogProbConfig, TopKLogProbContext
from xtuner.v1.loss.utils import sp_gather, sp_split
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.model_utils import (
    FrozenModel,
    build_frozen_model,
    build_frozen_model_shell,
    materialize_frozen_model,
    prepare_frozen_model,
)
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from .config import TrainTeacherManagerConfig


_DIST_WORLD: Any | None
try:
    from torch.distributed.distributed_c10d import _world as _DIST_WORLD
except ImportError:  # pragma: no cover - depends on the installed PyTorch internals
    _DIST_WORLD = None


if TYPE_CHECKING:
    from xtuner.v1.rl.loss.base_loss import BaseRLLossContext


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


def _registered_process_groups() -> tuple[Any, ...]:
    """Return PyTorch's registered process groups behind one guarded private
    seam."""
    if not dist.is_available() or not dist.is_initialized():
        return ()
    pg_map = getattr(_DIST_WORLD, "pg_map", None)
    if pg_map is None:
        raise RuntimeError(
            "Dynamic disk Teacher loading requires PyTorch's internal process-group registry, "
            "but distributed_c10d._world.pg_map is unavailable in this PyTorch version."
        )
    return tuple(pg_map)


@dataclass
class TeacherTiming:
    """Wall-clock seconds spent in one frozen Teacher lifecycle."""

    compute: float = 0.0
    onload: float = 0.0
    offload: float = 0.0


@dataclass
class TrainTeacherTimings(TeacherTiming):
    """Wall-clock seconds spent in each frozen Teacher lifecycle."""

    by_teacher: dict[str, TeacherTiming] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return per-Teacher lifecycle timings.

        Returns:
            dict[str, dict[str, float]]: Teacher names mapped to compute, onload, and offload seconds.
        """
        return {
            teacher_name: {
                "compute": timing.compute,
                "onload": timing.onload,
                "offload": timing.offload,
            }
            for teacher_name, timing in self.by_teacher.items()
        }


@dataclass
class TrainTeacherPack:
    """A Teacher-specific logical pack and its source sample metadata."""

    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    sample_indices: tuple[int, ...]
    sample_lengths: tuple[int, ...]


class TrainTeacherManager:
    """Execute training-side Teachers in one TrainingWorker process.

    The manager owns Teacher model construction, Teacher-grouped logical-pack execution, CPU/device residency, and
    output scattering. The caller remains responsible for swapping the Actor and optimizer around the Teacher phase.
    """

    def __init__(self, config: TrainTeacherManagerConfig, *, chunk_size: int | None) -> None:
        self.target_config = config.target_config
        mode = "chunk" if chunk_size is not None else "eager"
        self.logprob_config = LogProbConfig(chunk_size=chunk_size, mode=mode)
        self.topk_logprob_config: TopKLogProbConfig | None = None
        if self.target_config.uses_topk_targets:
            self.topk_logprob_config = TopKLogProbConfig(
                top_k=cast(int, self.target_config.top_k),
                chunk_size=chunk_size,
                mode=mode,
            )

        self._teacher_configs = config.teachers
        self._prefetch_enabled = config.prefetch
        self._teachers: list[FrozenModel | None] = []
        for teacher_config in config.teachers:
            if teacher_config.residency == "disk":
                self._teachers.append(None)
                continue
            teacher = build_frozen_model(
                teacher_config.model_cfg,
                teacher_config.model_path,
                teacher_config.fsdp_cfg,
            )
            self._teachers.append(teacher)

        self._teacher_is_composed = [isinstance(teacher.model_cfg, BaseComposeConfig) for teacher in config.teachers]
        self._teacher_index_by_name = {
            teacher_config.name: teacher_index for teacher_index, teacher_config in enumerate(config.teachers)
        }
        self._teacher_names = [teacher_config.name for teacher_config in config.teachers]
        self._teacher_process_groups: list[list[Any]] = [[] for _ in config.teachers]
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_future: Future[tuple[FrozenModel, Any]] | None = None
        self._prefetch_teacher_index: int | None = None

    @property
    def teacher_count(self) -> int:
        """Return the number of configured training Teachers."""
        return len(self._teacher_configs)

    def _build_disk_teacher(self, teacher_index: int, device: torch.device | str) -> FrozenModel:
        teacher_config = self._teacher_configs[teacher_index]
        fsdp_cfg = teacher_config.fsdp_cfg
        if fsdp_cfg is not None:
            fsdp_cfg = fsdp_cfg.model_copy(update={"cpu_offload": False})
        logger.info(f"[TrainTeacher] load start: teacher={teacher_config.name}, device={device}")
        load_begin = time.monotonic()
        process_groups_before = self._snapshot_process_groups()
        try:
            teacher = build_frozen_model(
                teacher_config.model_cfg,
                teacher_config.model_path,
                fsdp_cfg,
                target_device=device,
                load_device=device,
            )
        except BaseException:
            self._destroy_process_groups(self._new_process_groups(process_groups_before))
            raise
        self._record_process_groups(teacher_index, process_groups_before)
        logger.info(
            f"[TrainTeacher] load finished: teacher={teacher_config.name}, device={device}, "
            f"elapsed={time.monotonic() - load_begin:.3f}s"
        )
        return teacher

    def _start_teacher_prefetch_exact(self, teacher_index: int) -> None:
        """Prefetch one known-to-be-used disk Teacher."""
        if not self._prefetch_enabled or self._prefetch_executor is None or self._prefetch_future is not None:
            return
        teacher_config = self._teacher_configs[teacher_index]
        if teacher_config.residency != "disk":
            return
        self._prefetch_teacher_index = teacher_index
        logger.info(f"[TrainTeacher] prefetch scheduled: teacher={self._teacher_names[teacher_index]}")
        fsdp_cfg = teacher_config.fsdp_cfg
        if fsdp_cfg is not None:
            fsdp_cfg = fsdp_cfg.model_copy(update={"cpu_offload": False})
        process_groups_before = self._snapshot_process_groups()
        try:
            model = build_frozen_model_shell(teacher_config.model_cfg, fsdp_cfg)
        except BaseException:
            self._destroy_process_groups(self._new_process_groups(process_groups_before))
            raise
        self._record_process_groups(teacher_index, process_groups_before)
        self._prefetch_future = self._prefetch_executor.submit(prepare_frozen_model, model, teacher_config.model_path)

    def _take_teacher_prefetch(self, teacher_index: int) -> FrozenModel | None:
        if self._prefetch_future is None:
            return None
        if self._prefetch_teacher_index != teacher_index:
            raise RuntimeError(
                f"Teacher prefetch order mismatch: expected {self._prefetch_teacher_index}, got {teacher_index}"
            )
        future = self._prefetch_future
        self._prefetch_future = None
        self._prefetch_teacher_index = None
        teacher_name = self._teacher_names[teacher_index]
        wait_begin = time.monotonic()
        try:
            model, prepared = future.result()
            teacher = materialize_frozen_model(model, prepared)
        except NotImplementedError:
            self._release_teacher_process_groups(teacher_index)
            logger.info(
                f"[TrainTeacher] prefetch unavailable, falling back to GPU load: teacher={teacher_name}, "
                f"wait={time.monotonic() - wait_begin:.3f}s"
            )
            return None
        except BaseException:
            self._release_teacher_process_groups(teacher_index)
            raise
        logger.info(
            f"[TrainTeacher] prefetch ready: teacher={teacher_name}, wait={time.monotonic() - wait_begin:.3f}s"
        )
        return teacher

    def _finish_teacher_prefetch(self) -> None:
        future = self._prefetch_future
        teacher_index = self._prefetch_teacher_index
        self._prefetch_future = None
        self._prefetch_teacher_index = None
        if future is not None:
            try:
                future.result()
            except Exception:
                pass
            if teacher_index is not None:
                self._release_teacher_process_groups(teacher_index)
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None

    def compute_teacher_outputs_grouped(
        self,
        teacher_packs: dict[int, list[TrainTeacherPack]],
        loss_ctx_by_pack: dict[tuple[int, ...], BaseRLLossContext],
        sample_locations: dict[int, tuple[tuple[int, ...], int, int]],
        *,
        sp_mesh=None,
        pack_max_length: int | None = None,
    ) -> TrainTeacherTimings:
        """Run Teacher-specific logical packs and scatter results to source
        packs.

        The data flow has four stages:

        1. ``teacher_packs`` contains logical packs grouped by Teacher on the
           current rank; these are the only packs sent through Teacher forward.
        2. SP-local Teacher outputs are gathered back to global sequence order.
        3. Each sample's ``(original_pack_id, start, end)`` location maps the
           logical-pack output back to its original training pack.
        4. The global result is split again and stored in the original pack's
           loss context for the Actor update.
        """
        sp_enabled = sp_mesh is not None and sp_mesh.size() > 1
        uses_topk = self.target_config.uses_topk_targets
        output_buffers: dict[tuple[int, ...], dict[str, torch.Tensor | None]] = {}
        for original_pack_id, ctx in loss_ctx_by_pack.items():
            labels = cast(Any, ctx.loss_kwargs).shifted_labels
            global_shape = (labels.shape[0], labels.shape[1])
            if sp_enabled:
                if pack_max_length is None:
                    raise ValueError("pack_max_length is required for grouped Teacher outputs with SP")
                global_shape = (labels.shape[0], pack_max_length)
            if uses_topk:
                teacher_logprobs = torch.zeros(
                    (*global_shape, cast(int, self.target_config.top_k)),
                    dtype=torch.float32,
                    device=DEVICE,
                )
            else:
                teacher_logprobs = torch.zeros(
                    global_shape,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            target_ids_buffer = None
            if uses_topk:
                target_ids_buffer = torch.zeros(
                    (*global_shape, cast(int, self.target_config.top_k)),
                    dtype=torch.long,
                    device=DEVICE,
                )
            output_buffers[original_pack_id] = {
                "teacher_logprobs": teacher_logprobs,
                "target_token_ids": target_ids_buffer,
            }

        # Determine the active Teacher schedule globally before starting any
        # prefetch.  Empty Teachers must not be prefetched: grouped execution
        # skips them, so their futures would otherwise remain unconsumed until
        # the final cleanup and make the training step appear to hang.
        active_teacher_indices: list[int] = []
        active_pack_counts: dict[int, int] = {}
        for teacher_index in range(self.teacher_count):
            global_pack_count = self._global_max_count(len(teacher_packs.get(teacher_index, [])))
            if global_pack_count > 0:
                active_teacher_indices.append(teacher_index)
                active_pack_counts[teacher_index] = global_pack_count

        timings = TrainTeacherTimings()
        try:
            if (
                self._prefetch_enabled
                and any(teacher_config.residency == "disk" for teacher_config in self._teacher_configs)
                and active_teacher_indices
            ):
                self._prefetch_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="train-teacher-prefetch",
                )
                self._start_teacher_prefetch_exact(active_teacher_indices[0])

            for active_pos, teacher_index in enumerate(active_teacher_indices):
                teacher_name = self._teacher_names[teacher_index]
                next_teacher_index = (
                    active_teacher_indices[active_pos + 1] if active_pos + 1 < len(active_teacher_indices) else None
                )
                teacher_local_packs = teacher_packs.get(teacher_index, [])
                num_local_packs = active_pack_counts[teacher_index]
                teacher: FrozenModel | None = None
                try:
                    with self._teacher_on_device(
                        teacher_index,
                        timings,
                        teacher_name,
                        prefetch_next_teacher_index=next_teacher_index,
                    ) as teacher:
                        for local_pack_index in range(num_local_packs):
                            if local_pack_index < len(teacher_local_packs):
                                teacher_local_pack = teacher_local_packs[local_pack_index]
                                # Only one logical chunk is resident on the
                                # device at a time; the remaining regrouped
                                # packs stay on CPU.
                                input_ids = teacher_local_pack.seq_ctx.input_ids
                                if input_ids is None:
                                    raise RuntimeError("Teacher logical pack must contain input_ids")
                                global_length = input_ids.shape[1]
                                seq_ctx = teacher_local_pack.seq_ctx.copy().to(DEVICE)
                                if sp_enabled:
                                    seq_ctx = seq_ctx.split(sequence_parallel_mesh=sp_mesh)
                                shifted_labels = teacher_local_pack.shifted_labels
                            else:
                                teacher_local_pack = None
                                dummy_length = pack_max_length or 1
                                dummy_input_ids = cast(
                                    torch.LongTensor,
                                    torch.zeros((1, dummy_length), dtype=torch.long, device=DEVICE),
                                )
                                seq_ctx = SequenceContext.from_input_ids(
                                    (dummy_input_ids,),
                                    device=DEVICE,
                                )
                                shifted_labels = torch.full((1, dummy_length), -100, dtype=torch.long, device=DEVICE)

                            loss_ctx: LogProbContext | TopKLogProbContext
                            if uses_topk:
                                loss_ctx = cast(TopKLogProbConfig, self.topk_logprob_config).build(data={})
                            else:
                                logprob_ctx = self.logprob_config.build(
                                    data={"shifted_labels": shifted_labels},
                                    sp_mesh=sp_mesh,
                                    device=DEVICE,
                                )
                                assert logprob_ctx is not None
                                loss_ctx = logprob_ctx
                            with torch.no_grad():
                                output = teacher(
                                    seq_ctx=self.construct_teacher_seq_ctx(
                                        seq_ctx,
                                        is_composed=self._teacher_is_composed[teacher_index],
                                    ),
                                    loss_ctx={"lm": loss_ctx},
                                )
                            if teacher_local_pack is None:
                                continue

                            # Teacher output is local under SP.  Gather only
                            # the log-probs/top-k ids needed for distillation;
                            # the model activation path remains sharded.
                            output_loss = self._local_to_global_teacher_output(
                                cast(torch.Tensor, output.loss),
                                sp_mesh=sp_mesh,
                                global_length=global_length,
                            )
                            output_logits = None
                            if uses_topk:
                                output_logits = self._local_to_global_teacher_output(
                                    cast(torch.Tensor, output.logits),
                                    sp_mesh=sp_mesh,
                                    global_length=global_length,
                                )
                            output_offset = 0
                            for sample_index, sample_length in zip(
                                teacher_local_pack.sample_indices, teacher_local_pack.sample_lengths
                            ):
                                original_pack_id, start, end = sample_locations[sample_index]
                                buffer = output_buffers[original_pack_id]
                                buffer_teacher_logprobs = cast(torch.Tensor, buffer["teacher_logprobs"])
                                if uses_topk:
                                    # Top-k outputs have shape ``[B, T, K]``.  The
                                    # sequence range must be selected before the
                                    # final K dimension; ``output_loss[..., start:end]``
                                    # would slice K instead and leave all T tokens
                                    # in ``output_slice``.
                                    output_slice = output_loss[..., output_offset : output_offset + sample_length, :]
                                    buffer_teacher_logprobs[..., start:end, :] = output_slice
                                else:
                                    output_slice = output_loss[..., output_offset : output_offset + sample_length]
                                    buffer_teacher_logprobs[..., start:end] = output_slice
                                buffer_target_token_ids = cast(torch.Tensor | None, buffer["target_token_ids"])
                                if buffer_target_token_ids is not None:
                                    buffer_target_token_ids[..., start:end, :] = cast(torch.Tensor, output_logits)[
                                        ..., output_offset : output_offset + sample_length, :
                                    ]
                                output_offset += sample_length
                finally:
                    teacher = None
                    if self._teacher_configs[teacher_index].residency == "disk":
                        # This runs after the ``with`` statement has released
                        # its caller-side Teacher reference.  Logging inside
                        # _teacher_on_device() would sample memory too early.
                        self._collect_and_log_release(teacher_name)
        finally:
            self._finish_teacher_prefetch()
            for original_pack_id, ctx in loss_ctx_by_pack.items():
                loss_kwargs = cast(Any, ctx.loss_kwargs)
                buffer = output_buffers[original_pack_id]
                teacher_logprobs = cast(torch.Tensor, buffer["teacher_logprobs"])
                loss_kwargs.teacher_logprobs = (
                    sp_split(
                        teacher_logprobs,
                        sp_mesh,
                        split_dim=1,
                        padding_value=0.0,
                    )
                    if sp_enabled
                    else teacher_logprobs
                )
                target_ids_buffer = cast(torch.Tensor | None, buffer["target_token_ids"])
                if target_ids_buffer is not None:
                    loss_kwargs.target_token_ids = (
                        sp_split(
                            target_ids_buffer,
                            sp_mesh,
                            split_dim=1,
                            padding_value=0,
                        )
                        if sp_enabled
                        else target_ids_buffer
                    )
            self.offload_all_to_cpu()
        return timings

    @staticmethod
    def _local_to_global_teacher_output(
        output: torch.Tensor,
        *,
        sp_mesh,
        global_length: int,
    ) -> torch.Tensor:
        """Gather a local-SP Teacher output and remove sequence padding.

        Teacher forwards consume a sequence-parallel ``SequenceContext`` and
        therefore return only this rank's sequence shard.  The regrouping
        scatter is expressed in global sample coordinates, so gather the
        sequence dimension once here before copying token ranges back to the
        original packs.  ``narrow`` works for both sampled log-probs ``[B, T]``
        and top-k outputs ``[B, T, K]``.
        """
        if sp_mesh is not None and sp_mesh.size() > 1:
            output = sp_gather(output, sp_mesh, dim=1)
        return output.narrow(1, 0, global_length)

    @staticmethod
    def _global_max_count(local_count: int) -> int:
        """Make Teacher/chunk loop counts identical on every distributed
        rank."""
        if not dist.is_available() or not dist.is_initialized():
            return local_count
        count = torch.tensor(local_count, dtype=torch.long, device=DEVICE)
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
        return int(count.item())

    def offload_all_to_cpu(self) -> None:
        for teacher_index, (teacher, teacher_config) in enumerate(zip(self._teachers, self._teacher_configs)):
            if teacher is None:
                continue
            if teacher_config.residency == "disk":
                self._drop_teacher(teacher_index)
            else:
                self._offload_to_cpu(teacher)

    def offload_to_disk(self, teacher_name: str) -> None:
        """Drop a Teacher's weights so its next use reloads them from disk."""
        if teacher_name not in self._teacher_index_by_name:
            raise KeyError(f"Unknown training Teacher: {teacher_name!r}")
        self._drop_teacher(self._teacher_index_by_name[teacher_name])

    @staticmethod
    def _empty_device_cache() -> None:
        if hasattr(DEVICE_MODULE, "empty_cache"):
            DEVICE_MODULE.empty_cache()

    @staticmethod
    def _offload_to_cpu(teacher: FrozenModel) -> None:
        teacher.to_device("cpu")
        TrainTeacherManager._empty_device_cache()

    @staticmethod
    def _synchronize_device() -> None:
        if str(DEVICE) != "cpu" and hasattr(DEVICE_MODULE, "synchronize"):
            DEVICE_MODULE.synchronize()

    @staticmethod
    def _snapshot_process_groups() -> tuple[Any, ...]:
        """Return process groups registered in this worker, in creation
        order."""
        # PyTorch has no public process-group enumeration API. DeviceMesh uses
        # this registry internally, so taking an identity snapshot around model
        # construction lets disk Teachers release only the groups they own.
        return _registered_process_groups()

    def _new_process_groups(self, before: tuple[Any, ...]) -> list[Any]:
        before_ids = {id(group) for group in before}
        world_group = dist.group.WORLD if dist.is_available() and dist.is_initialized() else None
        return [
            group
            for group in self._snapshot_process_groups()
            if id(group) not in before_ids and group is not world_group
        ]

    def _record_process_groups(self, teacher_index: int, before: tuple[Any, ...]) -> None:
        groups = self._new_process_groups(before)
        if self._teacher_process_groups[teacher_index]:
            self._destroy_process_groups(groups)
            raise RuntimeError(f"Teacher {teacher_index} still owns process groups from a previous build")
        self._teacher_process_groups[teacher_index] = groups

    @staticmethod
    def _destroy_process_groups(groups: list[Any]) -> None:
        # Every rank builds and drops Teachers in the same order. Preserve the
        # registry's creation order so NCCL communicators are destroyed in the
        # same order on every rank.
        for group in groups:
            dist.destroy_process_group(group)

    def _release_teacher_process_groups(self, teacher_index: int) -> None:
        process_groups = self._teacher_process_groups[teacher_index]
        self._teacher_process_groups[teacher_index] = []
        self._destroy_process_groups(process_groups)

    def _drop_teacher(self, teacher_index: int) -> None:
        self._teachers[teacher_index] = None
        self._synchronize_device()
        self._release_teacher_process_groups(teacher_index)
        self._empty_device_cache()

    def _collect_and_log_release(self, teacher_name: str) -> None:
        gc.collect()
        self._empty_device_cache()
        if str(DEVICE) != "cpu":
            logger.info(
                f"[TrainTeacher] released: teacher={teacher_name}, "
                f"allocated={DEVICE_MODULE.memory_allocated() / 1024**3:.2f}GB, "
                f"reserved={DEVICE_MODULE.memory_reserved() / 1024**3:.2f}GB"
            )
            if DEVICE == "cuda":
                device = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(device)
                logger.info(
                    f"[GPU memory] event=teacher released, teacher={teacher_name}, device=cuda:{device}, "
                    f"free={free / 1024**3:.2f}GiB, total={total / 1024**3:.2f}GiB, "
                    f"used={(total - free) / 1024**3:.2f}GiB"
                )

    @contextmanager
    def _teacher_on_device(
        self,
        teacher_index: int,
        timings: TrainTeacherTimings,
        teacher_name: str | None = None,
        prefetch_next_teacher_index: int | None = None,
    ) -> Iterator[FrozenModel]:
        teacher_config = self._teacher_configs[teacher_index]
        teacher_timing = (
            timings.by_teacher.setdefault(teacher_name, TeacherTiming()) if teacher_name is not None else None
        )

        onload_begin = time.perf_counter()
        teacher = self._teachers[teacher_index]
        if teacher is None and teacher_config.residency == "disk":
            teacher = self._take_teacher_prefetch(teacher_index)
            if teacher is None:
                teacher = self._build_disk_teacher(teacher_index, DEVICE)
        elif teacher is None:
            teacher = build_frozen_model(
                teacher_config.model_cfg,
                teacher_config.model_path,
                teacher_config.fsdp_cfg,
            )
        if teacher is None:
            raise RuntimeError(f"Failed to materialize Teacher {teacher_config.name!r}")
        self._teachers[teacher_index] = teacher
        h2d_begin = time.monotonic()
        teacher.to_device(DEVICE)
        h2d_elapsed = time.monotonic() - h2d_begin
        # The grouped caller passes the next *active* Teacher explicitly.
        # Do not fall back to ``teacher_index + 1``: a configured Teacher may
        # have no samples in this batch and must not be prefetched.
        if teacher_config.residency == "disk" and prefetch_next_teacher_index is not None:
            self._start_teacher_prefetch_exact(prefetch_next_teacher_index)
        onload_elapsed = time.perf_counter() - onload_begin
        timings.onload += onload_elapsed
        if teacher_timing is not None:
            teacher_timing.onload += onload_elapsed

        compute_begin = time.perf_counter()
        try:
            yield teacher
            self._synchronize_device()
        finally:
            compute_elapsed = time.perf_counter() - compute_begin
            timings.compute += compute_elapsed
            if teacher_timing is not None:
                teacher_timing.compute += compute_elapsed
            offload_begin = time.perf_counter()
            if teacher_config.residency == "disk":
                self._drop_teacher(teacher_index)
                # Clear the context-manager's local Teacher reference.  The
                # caller performs GC after the ``with`` statement so its
                # context-manager target has also been released.
                teacher = None
            else:
                self._offload_to_cpu(teacher)
            offload_elapsed = time.perf_counter() - offload_begin
            timings.offload += offload_elapsed
            if teacher_timing is not None:
                teacher_timing.offload += offload_elapsed
            logger.info(
                f"[TrainTeacher] lifecycle: teacher={teacher_config.name}, residency={teacher_config.residency}, "
                f"onload={onload_elapsed:.3f}s, h2d={h2d_elapsed:.3f}s, compute={compute_elapsed:.3f}s, "
                f"offload={offload_elapsed:.3f}s"
            )

    @staticmethod
    def construct_teacher_seq_ctx(seq_ctx: SequenceContext, *, is_composed: bool) -> SequenceContext:
        """Construct a Teacher context from the Student rollout context.

        Rollout-only expert-routing metadata must not be reused by the Teacher.

        Composed/VLM Teachers keep visual fields and M-RoPE positions. Plain-text Teachers drop visual
        tensors and rebuild 2D packed ``position_ids`` so they can still run an FSDP-safe forward on the
        shared Student ``input_ids`` (image placeholder token ids stay in-vocab; only tokens routed via
        ``teacher_indices`` are consumed as distillation targets).
        """
        position_ids = seq_ctx.position_ids
        if not is_composed and position_ids is not None and position_ids.ndim == 3:
            # SequenceContext regenerates standard 2D packed positions when position_ids is None.
            position_ids = None

        # 复用公共复制契约，保留视觉输入、CPU offsets 和 SP 元数据，只清除 teacher 不该继承的状态。
        drop_visual = (
            {}
            if is_composed
            else {
                "pixel_values": None,
                "image_grid_thw": None,
                "pixel_values_videos": None,
                "video_grid_thw": None,
                "mm_token_type_ids": None,
                "num_img_tokens": None,
            }
        )
        return seq_ctx.copy(
            position_ids=position_ids,
            deepstack_visual_embeds=None,
            visual_pos_masks=None,
            inputs_embeds=None,
            rollout_routed_experts=None,
            offload_rollout_routed_experts=False,
            raw_inputs_embeds=None,
            **drop_visual,
        )
