"""Symmetric role-worker ABC hosted inside one ``TrainingWorker`` process."""

from __future__ import annotations

import contextlib
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal

import numpy as np
import ray
import torch
import torch.distributed as dist

from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.loss import BaseRLLossConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_device, get_torch_device_module

from .pack import DPRankPackIndices
from .schedule import optimizer_step_succeeds, resume_lr_scheduler, save_lr_scheduler


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

if TYPE_CHECKING:
    from logging import Logger

    from torch.distributed.device_mesh import DeviceMesh

    from xtuner.v1.data_proto.sequence_context import SequenceContext
    from xtuner.v1.loss import BaseLossContext
    from xtuner.v1.model.base import TransformerConfig

    from .worker import TrainBatchAttr, TrainingWorker, TrainSample, WorkerLogItem

# One prepared (seq_ctx, loss_ctx dict) pair per pack; roles build the dict
# (e.g. ``{"lm": ctx}`` or ``{"lm": ctx, "mtp": [depth ctxs]}``) before handing
# the step to ``_train_optimizer_steps``.
_PackItem = tuple["SequenceContext", dict[str, "BaseLossContext"]]

# Marks "no more packs" on the prefetch queue; distinct from any exception
# that _produce tunnels to the consumer.
_SENTINEL = object()

_UINT16_CAPACITY = 1 << 16

# A role name; None means no role resides on the GPU.
RoleName = Literal["actor", "critic"]


def _rollout_routed_experts_storage_dtype(n_routed_experts: int) -> torch.dtype:
    """Choose a compact lossless dtype for stored routed-expert IDs."""
    if n_routed_experts <= _UINT16_CAPACITY:
        return torch.uint16
    return torch.long


def _as_rollout_routed_experts_tensor(
    rollout_routed_experts: np.ndarray,
    *,
    n_routed_experts: int,
) -> torch.Tensor:
    """Convert finalized routed-expert IDs to their CPU storage dtype."""
    storage_dtype = _rollout_routed_experts_storage_dtype(n_routed_experts)
    if storage_dtype == torch.uint16 and rollout_routed_experts.dtype != np.uint16:
        if rollout_routed_experts.size > 0:
            min_expert_id = rollout_routed_experts.min()
            max_expert_id = rollout_routed_experts.max()
            if min_expert_id < 0 or max_expert_id >= _UINT16_CAPACITY:
                raise ValueError(
                    f"Routed-expert IDs cannot be represented as uint16: min={min_expert_id}, max={max_expert_id}"
                )
    return torch.as_tensor(rollout_routed_experts, dtype=storage_dtype)


def resolve_mm_inputs(seq_ctx: SequenceContext) -> None:
    """Materialize multimodal pixel-value object refs into one CPU tensor.

    Pure CPU / ray.get work: safe to run inside a prefetch producer thread.
    """
    pixel_values = seq_ctx.pixel_values
    if pixel_values is None:
        return
    if isinstance(pixel_values, np.ndarray):
        raise NotImplementedError("The case where pixel_values is a numpy array is not implemented yet.")
    assert isinstance(pixel_values, list), f"pixel_values should be list of tensor, got {type(pixel_values)}"
    pixel_values = ray.get(list(pixel_values))
    pixel_values = [torch.as_tensor(pixel_value) for pixel_value in pixel_values]
    seq_ctx.pixel_values = torch.cat(pixel_values, dim=0)


def resolve_routed_experts(
    seq_ctx: SequenceContext,
    *,
    model_cfg: TransformerConfig | BaseComposeConfig,
    pack_max_length: int,
) -> None:
    """Materialize rollout routed-expert refs into one packed CPU tensor.

    Pure CPU / ray.get work on the calling thread. Never frees refs: actor
    fit materializes each pack several times (old/ref/teacher/train passes)
    and each pass re-resolves from the pristine cache, so freeing happens
    exactly once, at the end of the batch, via
    ``free_rollout_routed_expert_refs``.
    """
    rollout_routed_experts = seq_ctx.rollout_routed_experts
    if rollout_routed_experts is None:
        # Dense models: rollout carries no routed-expert refs.
        return
    language_cfg = model_cfg.text_config if isinstance(model_cfg, BaseComposeConfig) else model_cfg
    if not getattr(language_cfg, "n_routed_experts", 0):
        # Dense model with stale refs (misconfigured rollout): drop them; the
        # forward routes on its own.
        seq_ctx.rollout_routed_experts = None
        return
    storage_dtype = _rollout_routed_experts_storage_dtype(language_cfg.n_routed_experts)

    if not isinstance(rollout_routed_experts, list):
        # All-padding pack: rollout_routed_experts is already a dummy tensor.
        assert isinstance(rollout_routed_experts, torch.Tensor), (
            f"padding experts should be a dummy tensor, bug got {type(rollout_routed_experts)}"
        )
        seq_ctx.rollout_routed_experts = torch.randint(
            low=0,
            high=language_cfg.n_routed_experts,
            size=(pack_max_length, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok),
            dtype=storage_dtype,
        )
        return

    # list[n,l,e]
    out_rollout_routed_expert = []
    for rollout_routed_expert in rollout_routed_experts:
        if isinstance(rollout_routed_expert, torch.Tensor):
            # Padding slot inside a mixed pack: a dummy tensor marks padding
            # positions; re-materialize it at the packed (l, e) shape.
            rollout_routed_experts_tensor = torch.randint(
                low=0,
                high=language_cfg.n_routed_experts,
                size=(
                    rollout_routed_expert.size(0),
                    language_cfg.num_hidden_layers,
                    language_cfg.num_experts_per_tok,
                ),
                dtype=storage_dtype,
            )
            out_rollout_routed_expert.append(rollout_routed_experts_tensor)
            continue
        if isinstance(rollout_routed_expert, ray.ObjectRef):
            rollout_routed_expert = ray.get(rollout_routed_expert)
        elif isinstance(rollout_routed_expert, list):
            _rollout_routed_expert = []
            for rollout_routed_expert_ref in rollout_routed_expert:
                _rollout_routed_expert.append(ray.get(rollout_routed_expert_ref))  # np
            rollout_routed_expert = np.concatenate(_rollout_routed_expert, axis=0)[1:, ...]
        else:
            raise ValueError(f"Invalid rollout_routed_expert_refs type: {type(rollout_routed_expert)}")
        rollout_routed_expert = _as_rollout_routed_experts_tensor(
            rollout_routed_expert,
            n_routed_experts=language_cfg.n_routed_experts,
        )
        rollout_routed_expert = rollout_routed_expert.reshape(
            -1, language_cfg.num_hidden_layers, language_cfg.num_experts_per_tok
        )
        out_rollout_routed_expert.append(rollout_routed_expert)

    seq_ctx.rollout_routed_experts = torch.cat(out_rollout_routed_expert, dim=0)  # max_len,l,e
    assert seq_ctx.rollout_routed_experts.size(0) == seq_ctx.input_ids.size(1), (
        f"rollout_routed_experts.size(0) {seq_ctx.rollout_routed_experts.size(0)} "
        f"!= input_ids.size(1) {seq_ctx.input_ids.size(1)}"
    )


def free_rollout_routed_expert_refs(samples: list[TrainSample], *, sp_mesh: DeviceMesh | None) -> None:
    """Release the cached routed-expert object refs once, after every consumer
    of the batch is done.

    Agent-loop trace-store refs may be shared by multiple trainable segments and replicated train workers, so within an
    SP mesh only rank 0 frees, after a mesh-wide barrier; without SP every rank frees its own refs.
    """
    refs: list[ray.ObjectRef] = []
    for sample in samples:
        rollout_routed_experts = sample.seq_ctx.rollout_routed_experts
        if rollout_routed_experts is None or isinstance(rollout_routed_experts, torch.Tensor):
            continue
        if isinstance(rollout_routed_experts, ray.ObjectRef):
            refs.append(rollout_routed_experts)
        elif isinstance(rollout_routed_experts, list):
            refs.extend(ref for ref in rollout_routed_experts if isinstance(ref, ray.ObjectRef))

    if sp_mesh is not None and sp_mesh.size() > 1:
        # Every rank must enter the barrier; only then may rank 0 free refs
        # that other ranks of the mesh may still be resolving.
        dist.barrier()
        if sp_mesh.get_local_rank() == 0 and refs:
            ray.internal.free(refs, local_only=False)
    elif refs:
        ray.internal.free(refs, local_only=False)


class RoleWorker(ABC):
    """A training role (actor / critic / ...) living inside one
    ``TrainingWorker``.

    All shared infrastructure (process group, device meshes, batch cache, pack
    primitives) is reached through ``self.host``; a role never calls the host's
    RPC or lifecycle methods.
    """

    role_name: ClassVar[RoleName]

    engine: TrainEngine
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    logger: Logger
    rank: int
    # ``finalize_metrics`` is only defined on RL loss configs.
    loss_cfg: BaseRLLossConfig | None = None

    def __init__(self, host: TrainingWorker) -> None:
        if host is None:
            raise ValueError(f"{type(self).__name__} must be attached to a TrainingWorker")
        self.host = host
        self._global_train_step = 0

    def onload(self) -> None:
        """Move model + optimizer to the GPU as a whole."""
        self.onload_model()
        self.onload_optimizer()

    def offload(self) -> None:
        """Move model + optimizer to the CPU as a whole."""
        self.offload_optimizer()
        self.offload_model()

    def onload_model(self) -> bool:
        """Move the model to the GPU; return whether it actually moved."""
        model_moved = self.engine.put_model_to_device(DEVICE)
        if not model_moved:
            self.logger.info("Skip model onload because model placement is unchanged.")
        return model_moved

    def onload_optimizer(self) -> bool:
        """Move the optimizer to the GPU; return whether it actually moved."""
        optimizer_moved = self.engine.put_optimizer_to_device(DEVICE)
        if not optimizer_moved:
            self.logger.info("Skip optimizer onload because no optimizer state needs to be moved.")
        return optimizer_moved

    def offload_model(self) -> bool:
        """Move the model to the CPU; return whether it actually moved."""
        model_moved = self.engine.put_model_to_device("cpu")
        if not model_moved:
            self.logger.info("Skip model offload because model placement is unchanged.")
            return False
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        return True

    def offload_optimizer(self) -> bool:
        """Move the optimizer to the CPU; return whether it actually moved."""
        optimizer_moved = self.engine.put_optimizer_to_device("cpu")
        if not optimizer_moved:
            self.logger.info("Skip optimizer offload because no optimizer state needs to be moved.")
            return False
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )
        return True

    @abstractmethod
    def fit(self) -> WorkerLogItem:  # noqa: F821
        """Run one training round on the batch cached by the host.

        The in-process signature is parameter-less by design: training inputs
        (scalar/token advantages, returns) arrive through named RPCs
        (``actor_fit(advantages)`` / ``critic_fit(token_returns)``); the host
        writes them into the sample cache via ``_add_sample_fields`` and the
        role reads them from there. This keeps every role's in-process
        signature uniform and decouples training inputs from the pack plan.
        """

    def save(self, path: Path, no_save_optimizer: bool = False) -> None:
        """Save a DCP checkpoint under ``path`` (the host already appended the
        role dir)."""
        self.engine.save_dcp(weights_dir=path, save_optimizer=not no_save_optimizer)
        save_lr_scheduler(self.lr_scheduler, path)

    def resume(self, path: Path, cfg: LoadCheckpointConfig) -> None:
        """Restore from the DCP checkpoint under ``path``."""
        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"Checkpoint at {path} has no DCP metadata.")
        self.engine.load_dcp(
            weights_dir=path,
            load_states=cfg.load_optimizer_states,
            load_args=cfg.load_optimizer_args,
        )
        resume_lr_scheduler(self.lr_scheduler, path, load_scheduler=cfg.load_scheduler)

    def save_hf(self, hf_dir: Path, save_dtype: torch.dtype) -> None:
        """Save the role model in Hugging Face format under ``hf_dir``."""
        self.engine.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def _cached(self) -> tuple[list[TrainSample], DPRankPackIndices, TrainBatchAttr]:
        return self.host.get_cached_batch()  # type: ignore[return-value]

    def _plan(self) -> DPRankPackIndices:
        """Return the cached pack plan (index-only)."""
        return self.host.get_cached_batch()[1]  # type: ignore[return-value]

    def _pack_loss_rows(
        self,
        samples: list[TrainSample],
        pack: list[int],
        rows: dict[str, list[torch.Tensor] | None],
    ) -> dict[str, torch.Tensor | None]:
        """Concatenate one pack's per-sample loss rows, tail-padded to
        ``pack_max_length``; an empty pack materializes entirely as padding.

        Args:
            samples (list[TrainSample]): The host's cached samples.
            pack (list[int]): Sample indices of the pack.
            rows (dict[str, list[torch.Tensor] | None]): Per-sample tensors by
                field name. ``None`` marks the field absent for the whole pack
                and passes through as ``None``; an empty list marks it present
                so a padding-only pack still materializes it.

        Returns:
            dict[str, torch.Tensor | None]: Packed tensors keyed by field name.
        """
        if pack:
            real_tokens = sum(samples[index].seq_ctx.input_ids.shape[1] for index in pack)  # type: ignore[union-attr,misc]
            padding_len = self.host.config.pack_max_length - int(real_tokens)
            if padding_len < 0:
                raise ValueError(
                    f"Pack tokens {real_tokens} exceed pack_max_length {self.host.config.pack_max_length}"
                )
            return self.host.cat_pack_fields(rows, padding_len)
        padding_len = self.host.config.pack_max_length
        return {
            key: (self.host.pack_padding_loss_input(key, padding_len) if row is not None else None)
            for key, row in rows.items()
        }

    def _resolve_refs(self, seq_ctx: SequenceContext) -> None:
        """Resolve per-pack object refs on the CPU side.

        Default: multimodal refs only. Roles override to add or drop fields:
        the actor also resolves routed experts, the critic drops them
        (it routes on its own).
        """
        resolve_mm_inputs(seq_ctx)

    def _iter_packs(
        self, packs: list[list[int]], *, split: bool = True
    ) -> Iterator[tuple[list[int], SequenceContext]]:
        """Materialize packs one at a time and yield ``(pack_indices,
        prepared_seq_ctx)``, overlapping each pack's CPU preparation with the
        previous pack's GPU work.

        This is the single materialization point: only one pack's copy is
        alive at a time (a list comprehension would materialize the whole
        batch at once and void the memory win). While the consumer trains on
        pack N, a producer thread prepares pack N+1 — but strictly CPU work
        only: ``pack_seq_ctx``'s cat plus ``_resolve_refs``' ray.get. H2D and
        SP split stay on the consuming thread so CUDA call-thread ownership
        stays simple.

        Hard constraints (breaking either reintroduces the bugs this design
        fixed):

        - ``_resolve_refs`` and ``pack_seq_ctx`` must stay pure CPU: no CUDA
          calls and no ``dist.*`` collectives in the producer thread. Ref
          freeing lives in the roles' fit finally, never here.
        - The consumer side must always run to ``finally``: the stop flag +
          queue drain there is what unblocks a producer blocked on ``put``
          when the generator is abandoned (consumer exception, early return).
        """
        stop = threading.Event()
        buffer: queue.Queue = queue.Queue(maxsize=1)

        def _produce() -> None:
            try:
                for p in packs:
                    if stop.is_set():
                        return
                    seq_ctx = self.host.pack_seq_ctx(p)  # CPU cat
                    self._resolve_refs(seq_ctx)  # role-level refs resolution (ray.get)
                    buffer.put((p, seq_ctx))
                buffer.put(_SENTINEL)
            except BaseException as exc:  # noqa: BLE001 - tunneled to the consumer below
                buffer.put(exc)

        threading.Thread(target=_produce, daemon=True).start()
        try:
            while (item := buffer.get()) is not _SENTINEL:
                if isinstance(item, BaseException):
                    raise item
                p, seq_ctx = item
                seq_ctx.offload_rollout_routed_experts = self.host.config.offload_rollout_routed_experts
                seq_ctx = seq_ctx.to(DEVICE)
                sp_mesh = self.host.sp_mesh if split else None
                if sp_mesh is not None and sp_mesh.size() > 1:
                    seq_ctx = seq_ctx.split(sequence_parallel_mesh=sp_mesh)
                yield p, seq_ctx
        finally:
            # Consumer exited early (exception or generator close): let the
            # producer see the stop flag and drain the queue so its blocked
            # ``put`` returns and the thread (and its materialized seq_ctx)
            # can be released.
            stop.set()
            try:
                while True:
                    buffer.get_nowait()
            except queue.Empty:
                pass

    def _iter_steps(self, *, split: bool = True) -> Iterator[list[tuple[list[int], SequenceContext]]]:
        """Group ``_iter_packs`` deliveries by optimizer-step boundary.

        ``_build_step_input`` needs the whole step's pack list, so this
        buffers one step at a time. Materialization stays lazy across step
        boundaries.
        """
        plan = self._plan()
        stream = self._iter_packs([p for step in plan for p in step], split=split)
        for step in plan:
            yield [next(stream) for _ in step]

    @abstractmethod
    def _build_step_input(self, items: list[_PackItem]) -> list[ModelItem]:
        """Merge one optimizer step's prepared packs into engine input.

        Runs the role's ``LossContext.build_batches`` along the step boundary
        so the global-normalization denominator is all-reduced once per step.
        Role-specific batching (MTP depths, critic contexts) lives here.
        """

    def _train_optimizer_steps(self, optimizer_step_packs: Iterator[list[_PackItem]]) -> WorkerLogItem:
        """Consume prepared optimizer steps and train.

        Per optimizer update:
          → ``_build_step_input`` (build_batches along the step boundary, one
            all-reduce per normalization denominator)
          → ``engine.train_step`` → ``clip_grad_norm`` →
            ``optimizer_step_succeeds`` → ``step_optimizer``
          → the lr scheduler only steps when the optimizer update applies
          → accumulate ``WorkerLogItem.train_metrics``

        Empty-update filtering and multi-pass expansion are done by the role
        before calling this; they do not enter the skeleton.
        """
        from .worker import WorkerLogItem as _WorkerLogItem

        worker_log_item: WorkerLogItem = _WorkerLogItem(train_entropy=0.0, train_metrics=[])
        for step_items in optimizer_step_packs:
            self._global_train_step += 1
            global_train_step = self._global_train_step
            engine_input = self._build_step_input(step_items)

            # Make max_memory/reserved_memory below this step's peak instead of the
            # process-lifetime peak (~2us, no sync), so the memory columns line up
            # with per-step metrics such as `ep_load_peak_ratio`.
            DEVICE_MODULE.reset_peak_memory_stats()
            with self._maybe_profiling(global_train_step):
                train_step_info = self.engine.train_step(data_batches=engine_input)
            grad_norm = self.engine.clip_grad_norm()
            stepped = optimizer_step_succeeds(self.engine, grad_norm)
            self.engine.step_optimizer(grad_norm)
            if stepped and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            engine_logs_info = dict(train_step_info.pop("logs_info"))  # type: ignore[misc]
            engine_extra_info = train_step_info.pop("extra_info")  # type: ignore[misc]
            train_step_info.pop("total_loss", None)  # type: ignore[misc]
            if hasattr(engine_extra_info, "get"):
                extra_info_dict = engine_extra_info.get()
            else:
                extra_info_dict = engine_extra_info
            extra_info_dict = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in extra_info_dict.items()
                if isinstance(v, (torch.Tensor, int, float))
            }
            if self.loss_cfg is not None:
                extra_info_dict = self.loss_cfg.finalize_metrics(extra_info_dict, DEVICE)
            extra_info_dict.update(self._extra_step_metrics())
            max_memory = DEVICE_MODULE.max_memory_allocated() / (1024**3)
            reserved_memory = DEVICE_MODULE.max_memory_reserved() / (1024**3)

            train_log_item = {
                **engine_logs_info,
                **train_step_info,
                **extra_info_dict,
                "grad_norm": grad_norm.item(),
                "max_memory": max_memory,
                "reserved_memory": reserved_memory,
            }
            worker_log_item["train_metrics"].append(train_log_item)  # type: ignore[typeddict-item,arg-type]

            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in train_log_item.items()
                if not key.startswith("reduced_train_policy_") and key != "max_ratio"
            )
            self.logger.info(f"Rank{self.rank} GlobalStep {global_train_step}: {log_str}")
        return worker_log_item

    def _maybe_profiling(self, global_train_step: int) -> Any:
        """Profile matching steps when the role enables profiling; no-op
        default."""
        return contextlib.nullcontext()

    def _extra_step_metrics(self) -> dict[str, float]:
        """Role-specific per-step metrics merged into ``train_metrics``."""
        return {}

    def _masked_token_count(self, packs: list[list[int]]) -> int:
        """Count trainable tokens (masked-label positions) across all ranks.

        Reads the host's cached CPU labels, so it works on pack indices before any materialization. The critic uses it
        to drop empty updates symmetrically across ranks.
        """
        samples = self.host.get_cached_batch()[0]
        local_tokens = sum(int(samples[index].shifted_labels.ne(-100).sum()) for pack in packs for index in pack)
        # NCCL-only process groups reject CPU tensors; keep it on the device.
        total = torch.tensor(float(local_tokens), device=DEVICE)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        return int(total.item())
