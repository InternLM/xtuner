from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future, wait
from pathlib import Path
from typing import Any, Dict, List, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from typing_extensions import TypedDict


try:
    from torch.distributed.checkpoint.state_dict_saver import AsyncSaveResponse
except ImportError:
    AsyncSaveResponse = None

try:
    from torch.distributed.checkpoint.staging import AsyncStager, BlockingAsyncStager
except ImportError:
    AsyncStager = None  # type: ignore[assignment,misc]
    BlockingAsyncStager = None  # type: ignore[assignment,misc]


from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
)

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.base import (
    BaseModel,
    BatchForwardInfo,
    DataBatchInfo,
    ModelItem,
    ModelOutputs,
    XTunerBaseModelConfig,
)
from xtuner.v1.profiler.prober import ProberList
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module, profile_time_and_memory
from xtuner.v1.utils.grad_norm import cal_grad_norm


if BlockingAsyncStager is not None:

    class _CachingStagingWriter(FileSystemWriter):
        """FileSystemWriter that delegates staging to a shared BlockingAsyncStager.

        PyTorch 2.8's ``dcp.async_save()`` activates pinned-memory staging only
        when ``storage_writer`` implements the ``AsyncStager`` protocol.  This
        thin wrapper inherits ``FileSystemWriter`` (for disk I/O) and registers
        as a virtual subclass of ``AsyncStager`` so the ``isinstance`` check
        inside ``async_save`` passes, while the actual staging logic is
        delegated to an externally-owned ``BlockingAsyncStager`` whose pinned
        buffers are reused across checkpoint calls.

        Args:
            path (str): Directory for checkpoint files.
            stager (BlockingAsyncStager): Shared stager that owns the pinned buffer cache.
        """

        _synchronize_after_execute: bool = False

        def __init__(self, path: str, stager: BlockingAsyncStager) -> None:
            super().__init__(path)
            self._stager = stager

        def stage(self, state_dict: dict[str, Any]) -> dict[str, Any]:
            return self._stager.stage(state_dict)

        def synchronize_staging(self) -> None:
            self._stager.synchronize_staging()

    # Register as virtual subclass so isinstance(writer, AsyncStager) is True.
    AsyncStager.register(_CachingStagingWriter)  # type: ignore[union-attr]


class AsyncCheckpointFutures(TypedDict):
    """Futures returned by async_save_dcp for two-phase checkpoint tracking.

    Args:
        staging (list[Future]): Futures that complete when GPU->pinned CPU staging finishes (GPU memory released).
        upload (list[Future]): Futures that complete when background disk I/O finishes.
    """

    staging: list[Future]
    upload: list[Future]


class TrainStepInfo(DataBatchInfo, BatchForwardInfo):
    total_loss: float


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()

threading_lock = threading.Lock()


class CPUThreadTaskCoordinator:
    def __init__(self, futures, callback):
        self.futures = futures
        self.callback = callback
        self._completed_tasks = 0

        assert isinstance(self.futures, list), "futures should be a list"
        for future in futures:
            future.add_done_callback(self.task_done)

    def task_done(self, future):
        # To call the callback only when all futures are done
        with threading_lock:
            self._completed_tasks += 1
            if self._completed_tasks == len(self.futures):
                self._completed_tasks = 0
                if self.callback is not None:
                    self.callback(future)

    def wait(self):
        wait(self.futures)


class HFCheckpointLoader:
    def __init__(self, model_path, cache_dir=None, from_hub="huggingface"):
        self.model_path = model_path

        if "model.safetensors.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.safetensors.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = True
        elif "model.bin.index.json" in os.listdir(self.model_path):
            index_json = os.path.join(self.model_path, "model.bin.index.json")
            self.weight_map = json.load(open(index_json))["weight_map"]
            self.use_safetensors = False
        elif "model.safetensors" in os.listdir(self.model_path):
            with safe_open(os.path.join(self.model_path, "model.safetensors"), framework="pt") as f:
                self.weight_map = dict.fromkeys(f.keys(), "model.safetensors")
            self.use_safetensors = True
        else:
            raise FileNotFoundError

        self.current_file = None
        self.buffer = None

    def load(self, key):
        if key not in self.weight_map:
            logger.warning(f"{key} not in checkpoint.")
            return

        _file = self.weight_map[key]

        if self.use_safetensors:
            if self.current_file is None:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = safe_open(os.path.join(self.model_path, _file), framework="pt")
                self.current_file = _file
            weight = self.buffer.get_tensor(key)

        else:
            if self.current_file is None:
                self.buffer = torch.load(os.path.join(self.model_path, _file))
                self.current_file = _file

            if _file != self.current_file:
                self.buffer = torch.load(os.path.join(self.model_path, _file))

            weight = self.buffer[key]

        return weight


class TrainEngine:
    model: BaseModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        model_cfg: XTunerBaseModelConfig,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig,
        intra_layer_micro_batch: int = 1,
        async_checkpoint: bool = False,
    ) -> None:
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.fsdp_cfg = fsdp_cfg
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(optim_cfg)
        self.intra_layer_micro_batch = intra_layer_micro_batch
        self._count = 0
        self.has_freeze_params = self.__has_freeze_params()
        self._async_checkpoint_pg: dist.ProcessGroup | None = None
        self._stager: BlockingAsyncStager | None = None
        if async_checkpoint:
            self._async_checkpoint_pg = dist.new_group(backend="gloo")
            if BlockingAsyncStager is not None:
                self._stager = BlockingAsyncStager(cache_staged_state_dict=True)

    def __has_freeze_params(self) -> bool:
        has_freeze_params = False
        for param in self.model.parameters(recurse=True):
            if not param.requires_grad:
                has_freeze_params = True
                break
        return has_freeze_params

    def build_model(self) -> BaseModel:
        with torch.device("meta"):
            model = self.model_cfg.build()

        model = model.fully_shard(self.fsdp_cfg)

        if dist.get_rank() == 0:
            logger.info(model)
        return model

    def build_optimizer(self, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
        return optim_cfg.build(self.model)

    @property
    def data_replicate_size(self) -> int:
        # todo: consider pp
        return self.fsdp_cfg.tp_size

    @torch.no_grad()
    def forward_only(self, seq_ctx: SequenceContext):
        output = self.model(seq_ctx=seq_ctx, loss_ctx=None)
        return output

    def grad_accumulation_steps(self, data_batches_len: int):
        intra_layer_micro_batch = self.intra_layer_micro_batch
        return data_batches_len // intra_layer_micro_batch

    def train_step(self, data_batches: list[ModelItem]) -> TrainStepInfo:
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
        """
        self._maybe_precompute_float8_dynamic_scale_for_fsdp()

        intra_layer_micro_batch = self.intra_layer_micro_batch
        assert len(data_batches) % intra_layer_micro_batch == 0, (
            f"data_batches length {len(data_batches)} is not divisible by intra_layer_micro_batch {intra_layer_micro_batch}"
        )
        iters_per_step = self.grad_accumulation_steps(len(data_batches))

        if self._count == 0:
            logger.info(f"grad_accumulation_steps: {iters_per_step}")
            self._count += 1

        micro_batch_iter = 0
        micro_batch_results = []

        data_batch_info = self.model.pre_micro_batch_forward(data_batches)
        total_loss = torch.tensor(0.0, device=DEVICE)

        for i in range(0, len(data_batches), intra_layer_micro_batch):
            ProberList.set_micro_batch_iter(micro_batch_iter)
            micro_batch_iter += 1
            data_batch = data_batches[i : i + intra_layer_micro_batch]
            seq_ctx_list = [i["seq_ctx"] for i in data_batch]
            loss_ctx_list = [i["loss_ctx"] for i in data_batch]

            if self.intra_layer_micro_batch == 1:
                output = self.model(seq_ctx=seq_ctx_list[0], loss_ctx=loss_ctx_list[0])
            else:
                # For intra_layer_micro_batch > 1, we need to handle the data batches differently.
                # Here we assume that the model can handle a list of seq_ctx and loss_ctx.
                output = self.model(
                    seq_ctx=seq_ctx_list,
                    loss_ctx=loss_ctx_list,
                )
            output.free_nongrad_feature()

            micro_batch_results.append(output)

            loss = self._get_total_loss(output)
            loss.backward()
            total_loss += loss.detach()
            # call dump_forward_records after backward to record the recomputed activations
            ProberList.after_micro_iter_forward()

        batch_forward_info = self.model.post_micro_batch_forward(micro_batch_results)
        return TrainStepInfo(total_loss=total_loss.item(), **data_batch_info, **batch_forward_info)

    def from_hf(self, hf_path: str | Path, strict: bool = False):
        self.model.from_hf(hf_path=hf_path, strict=strict)

    def init_model_weights(self):
        self.model.init_weights()

    @_no_grad
    def clip_grad_norm(self, do_clip: bool = True, dtype=torch.float32):
        ProberList.before_clip_grad_norm(self.model)
        self.model.scale_and_reduce_grad()
        params = self.model.trainable_parameters()
        grads = [p.grad for _, p in params if p.grad is not None]
        grad_norm, grouped_grads = cal_grad_norm(grads, dtype=dtype)
        if do_clip:
            clip_coef = self.optim_cfg.max_grad_norm / (grad_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for grads in grouped_grads.values():
                device = grads[0].device
                if _device_has_foreach_support(device):
                    torch._foreach_mul_(grads, clip_coef_clamped.to(device))
                else:
                    clip_coef_clamped_device = clip_coef_clamped.to(device)
                    for g in grads:
                        g.mul_(clip_coef_clamped_device)
        ProberList.after_clip_grad_norm(self.model, grad_norm)
        return grad_norm

    def step_optimizer(self, grad_norm):
        """Step the optimizer to update the model parameters."""
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning(f"Gradient norm {grad_norm} is invalid, skipping optimizer step.")
            self.optimizer.zero_grad()
        elif (
            self.optim_cfg.skip_grad_norm_threshold is not None and grad_norm > self.optim_cfg.skip_grad_norm_threshold
        ):
            logger.warning(
                f"Gradient norm {grad_norm} exceeds the threshold {self.optim_cfg.skip_grad_norm_threshold}, skipping optimizer step."
            )
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm

    # TODO: Should be removed
    @staticmethod
    def clean_param_name(name: str) -> str:
        if "_checkpoint_wrapped_module." in name:
            name = name.replace("_checkpoint_wrapped_module.", "")
        if "_orig_mod." in name:
            name = name.replace("_orig_mod.", "")
        return name

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        """Save the hf model to the given directory.

        Args:
            hf_dir (str): The directory to save the model.
            save_dtype (torch.dtype): The dtype to save the model parameters, bfloat16 or float8.
        """
        self.model.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)

    def save_dcp(
        self,
        model_dir: Path,
        optimizer_dir: Path | None = None,
    ):
        rank = dist.get_rank()

        if rank == 0:
            model_dir.mkdir(parents=True, exist_ok=True)
            if optimizer_dir is not None:
                optimizer_dir.mkdir(parents=True, exist_ok=True)

        _options = StateDictOptions(cpu_offload=True, ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params)

        with profile_time_and_memory(f"[DCP Collect Model State Dict for {model_dir}]"):
            model_state = get_model_state_dict(self.model, options=_options)

        with profile_time_and_memory(f"[DCP save(model) for {model_dir}]"):
            dcp.save(
                model_state,
                checkpoint_id=model_dir,
            )

        if optimizer_dir is not None:
            with profile_time_and_memory(f"[DCP Collect Optimizer State Dict for {optimizer_dir}]"):
                shard_optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=_options)

            with profile_time_and_memory(f"[DCP save(optimizer) for {optimizer_dir}]"):
                dcp.save(
                    shard_optimizer_state_dict,
                    checkpoint_id=optimizer_dir,
                )

    # def benchmark_cpu_offload(self) -> None:
    #     """Benchmark get_model/optimizer_state_dict with cpu_offload=True vs False.

    #     Measures the FSDP state dict collection overhead with and without D2H copy
    #     to determine whether building a custom async pipeline (bypassing PyTorch DCP)
    #     is worthwhile.  Results are logged at INFO level on all ranks.

    #     This method does NOT save to disk — it only collects state dicts and discards
    #     them immediately.  Safe to call at any point during training.
    #     """
    #     import gc
    #     import time

    #     logger = get_logger()
    #     rank = dist.get_rank()
    #     ignore_frozen = self.model_cfg.dcp_ignore_frozen_params

    #     # Warmup: ensure CUDA caches are populated
    #     torch.cuda.synchronize()
    #     dist.barrier()

    #     results: dict[str, dict[str, float]] = {}

    #     for offload_label, cpu_offload in [("cpu_offload=True", True), ("cpu_offload=False", False)]:
    #         opts = StateDictOptions(cpu_offload=cpu_offload, ignore_frozen_params=ignore_frozen)

    #         # --- model state dict ---
    #         torch.cuda.synchronize()
    #         dist.barrier()
    #         t0 = time.perf_counter()
    #         model_sd = get_model_state_dict(self.model, options=opts)
    #         torch.cuda.synchronize()
    #         t_model = time.perf_counter() - t0

    #         # Check tensor device
    #         sample_device = "N/A"
    #         for v in model_sd.values():
    #             if isinstance(v, torch.Tensor):
    #                 sample_device = str(v.device)
    #                 break

    #         del model_sd
    #         gc.collect()
    #         if not cpu_offload:
    #             torch.cuda.empty_cache()

    #         # --- optimizer state dict ---
    #         torch.cuda.synchronize()
    #         dist.barrier()
    #         t0 = time.perf_counter()
    #         optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=opts)
    #         torch.cuda.synchronize()
    #         t_optim = time.perf_counter() - t0

    #         del optim_sd
    #         gc.collect()
    #         if not cpu_offload:
    #             torch.cuda.empty_cache()

    #         results[offload_label] = {
    #             "model": t_model,
    #             "optimizer": t_optim,
    #             "total": t_model + t_optim,
    #             "sample_device": sample_device,
    #         }

    #         logger.info(
    #             f"[Benchmark cpu_offload] RANK {rank} | {offload_label}: "
    #             f"model={t_model:.3f}s, optimizer={t_optim:.3f}s, "
    #             f"total={t_model + t_optim:.3f}s, tensor_device={sample_device}"
    #         )

    #     # Summary comparison
    #     t_true = results["cpu_offload=True"]["total"]
    #     t_false = results["cpu_offload=False"]["total"]
    #     diff = t_true - t_false
    #     logger.info(
    #         f"[Benchmark cpu_offload] RANK {rank} | SUMMARY: "
    #         f"cpu_offload=True total={t_true:.3f}s, "
    #         f"cpu_offload=False total={t_false:.3f}s, "
    #         f"D2H overhead estimate={diff:.3f}s ({diff / t_true * 100:.1f}% of True)"
    #     )

    #     dist.barrier()

    def async_save_dcp_merged(
        self,
        weights_dir: Path,
    ) -> AsyncCheckpointFutures:
        """Asynchronously save model and optimizer in a single DCP call (Megatron-style).

        Uses ``cpu_offload=False`` so that ``get_model/optimizer_state_dict``
        only collects GPU tensor references (~0.04s each) instead of doing
        synchronous GPU-to-CPU copies (~7.7s total).  The single
        ``dcp.async_save()`` call stages all tensors from GPU to pinned CPU
        via ``BlockingAsyncStager`` in one pass, then writes to disk
        asynchronously in the background.

        Compared to the legacy ``async_save_dcp``, this eliminates:
        - D2H copy inside state_dict collection (7.7s → 0.08s)
        - The redundant CPU → pinned-CPU staging copy (~1.0s saved)
        - Two-call serialisation overhead (~0.5s saved)

        Args:
            weights_dir (Path): Directory to save the merged checkpoint.

        Returns:
            AsyncCheckpointFutures: Staging and upload futures.
        """
        if not hasattr(dcp, "async_save"):
            raise RuntimeError(
                "dcp.async_save is not available in this PyTorch version. "
                "Please upgrade PyTorch or set async_checkpoint=False."
            )

        rank = dist.get_rank()
        if rank == 0:
            weights_dir.mkdir(parents=True, exist_ok=True)

        # cpu_offload=False: collect GPU tensor references only, no D2H copy.
        # D2H will happen once during BlockingAsyncStager.stage() (GPU → pinned CPU).
        _options = StateDictOptions(
            cpu_offload=False,
            ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params,
        )

        with profile_time_and_memory(f"[DCP Collect Model State Dict (no offload) for {weights_dir}]"):
            model_state = get_model_state_dict(self.model, options=_options)

        with profile_time_and_memory(f"[DCP Collect Optimizer State Dict (no offload) for {weights_dir}]"):
            optim_state = get_optimizer_state_dict(self.model, self.optimizer, options=_options)

        # Merge into single state dict for one DCP call (eliminates gloo PG deadlock
        # risk and two-call serialisation overhead).
        merged: dict[str, Any] = {"model": model_state, "optimizer": optim_state}

        with profile_time_and_memory(f"[DCP async_save(merged) for {weights_dir}]"):
            async_save_kwargs: dict[str, Any] = {
                "checkpoint_id": weights_dir,
                "process_group": self._async_checkpoint_pg,
            }
            if self._stager is not None:
                async_save_kwargs["storage_writer"] = _CachingStagingWriter(
                    str(weights_dir), self._stager
                )
            result = dcp.async_save(merged, **async_save_kwargs)

        staging_futures: list[Future] = []
        upload_futures: list[Future] = []
        if AsyncSaveResponse is not None and isinstance(result, AsyncSaveResponse):
            staging_futures.append(result.staging_completion)
            upload_futures.append(result.upload_completion)
        else:
            staging_futures.append(result)
            upload_futures.append(result)

        return AsyncCheckpointFutures(staging=staging_futures, upload=upload_futures)

    # def async_save_dcp(
    #     self,
    #     model_dir: Path,
    #     optimizer_dir: Path | None = None,
    # ) -> AsyncCheckpointFutures:
    #     """Asynchronously save model and optimizer via DCP (legacy, deprecated).

    #     .. deprecated::
    #         Use :meth:`async_save_dcp_merged` instead, which collects state
    #         dicts with ``cpu_offload=False`` and saves in a single DCP call
    #         for significantly lower blocking time.

    #     Args:
    #         model_dir (Path): Directory to save the model checkpoint.
    #         optimizer_dir (Path | None): Directory to save the optimizer checkpoint.

    #     Returns:
    #         AsyncCheckpointFutures: Two groups of futures for staging and upload.
    #     """
    #     import warnings

    #     warnings.warn(
    #         "async_save_dcp() is deprecated. Use async_save_dcp_merged() instead "
    #         "for ~5x lower blocking time.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     if not hasattr(dcp, "async_save"):
    #         raise RuntimeError(
    #             "dcp.async_save is not available in this PyTorch version. "
    #             "Please upgrade PyTorch or set async_checkpoint=False."
    #         )

    #     rank = dist.get_rank()

    #     if rank == 0:
    #         model_dir.mkdir(parents=True, exist_ok=True)
    #         if optimizer_dir is not None:
    #             optimizer_dir.mkdir(parents=True, exist_ok=True)

    #     staging_futures: list[Future] = []
    #     upload_futures: list[Future] = []

    #     # Use cpu_offload=True so that get_*_state_dict copies tensors to CPU
    #     # synchronously (no extra GPU memory).  dcp.async_save() then only
    #     # needs to asynchronise the disk I/O, which benefits slow storage
    #     # (NFS/S3) without risking OOM on large models.
    #     _options = StateDictOptions(
    #         cpu_offload=True,
    #         ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params,
    #     )

    #     with profile_time_and_memory(f"[DCP Collect Model State Dict for {model_dir}]"):
    #         model_state = get_model_state_dict(self.model, options=_options)

    #     with profile_time_and_memory(f"[DCP async_save(model) for {model_dir}]"):
    #         async_save_kwargs: dict[str, Any] = {
    #             "checkpoint_id": model_dir,
    #             "process_group": self._async_checkpoint_pg,
    #         }
    #         if self._stager is not None:
    #             async_save_kwargs["storage_writer"] = _CachingStagingWriter(
    #                 str(model_dir), self._stager
    #             )
    #         result = dcp.async_save(model_state, **async_save_kwargs)
    #     if AsyncSaveResponse is not None and isinstance(result, AsyncSaveResponse):
    #         staging_futures.append(result.staging_completion)
    #         upload_futures.append(result.upload_completion)
    #     else:
    #         # Older PyTorch: single future covers both staging and upload
    #         staging_futures.append(result)
    #         upload_futures.append(result)

    #     if optimizer_dir is not None:
    #         # Wait for the model save thread to finish before starting the
    #         # optimizer save.  Both background threads use the same gloo
    #         # process group for collectives (reduce_scatter / all_reduce).
    #         # Concurrent collectives on a single gloo PG from different
    #         # threads cause message interleaving and deadlock.
    #         staging_futures[-1].result()

    #         with profile_time_and_memory(f"[DCP Collect Optimizer State Dict for {optimizer_dir}]"):
    #             shard_optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=_options)

    #         with profile_time_and_memory(f"[DCP async_save(optimizer) for {optimizer_dir}]"):
    #             async_save_kwargs = {
    #                 "checkpoint_id": optimizer_dir,
    #                 "process_group": self._async_checkpoint_pg,
    #             }
    #             if self._stager is not None:
    #                 async_save_kwargs["storage_writer"] = _CachingStagingWriter(
    #                     str(optimizer_dir), self._stager
    #                 )
    #             result = dcp.async_save(shard_optimizer_state_dict, **async_save_kwargs)
    #         if AsyncSaveResponse is not None and isinstance(result, AsyncSaveResponse):
    #             staging_futures.append(result.staging_completion)
    #             upload_futures.append(result.upload_completion)
    #         else:
    #             staging_futures.append(result)
    #             upload_futures.append(result)

    #     return AsyncCheckpointFutures(staging=staging_futures, upload=upload_futures)

    def destroy_async_checkpoint_pg(self) -> None:
        """Destroy the dedicated gloo process group used for async checkpoint."""
        self._stager = None
        if self._async_checkpoint_pg is not None:
            dist.destroy_process_group(self._async_checkpoint_pg)
            self._async_checkpoint_pg = None

    def load_dcp(
        self,
        model_dir: Path,
        optimizer_dir: Path | None = None,
        load_states: bool = True,
        load_args: bool = True,
    ):
        """Load the dcp model from the given directory.

        Args:
            dcp_dir (str): The directory to load the model from.
        """
        _load_options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params
        )
        if self.has_freeze_params:
            _set_options = StateDictOptions(cpu_offload=True, strict=False)
        else:
            _set_options = StateDictOptions(cpu_offload=True, strict=True)
        with profile_time_and_memory(f"[Load DCP Model from {model_dir}]"):
            shard_model_state_dict = get_model_state_dict(self.model, options=_load_options)
            # inplace state_dict
            dcp.load(
                state_dict=shard_model_state_dict,
                checkpoint_id=model_dir,
            )
            set_model_state_dict(self.model, shard_model_state_dict, options=_set_options)

        if optimizer_dir is not None:
            with profile_time_and_memory(f"[Load DCP Optimizer] from {optimizer_dir}"):
                shard_optimizer_state_dict = get_optimizer_state_dict(
                    self.model, self.optimizer, options=_load_options
                )
                dcp.load(
                    state_dict=shard_optimizer_state_dict,
                    checkpoint_id=optimizer_dir,
                )
                if not load_states:
                    logger.info("Not loading optimizer states")
                    shard_optimizer_state_dict["state"] = {}
                if not load_args:
                    logger.info("Not loading arg defaults")
                    param_groups = self.optimizer.state_dict()["param_groups"]
                    # Now we only support one param_group. If we want to support different lr for different parameters,
                    # we may use multiple param_groups like:
                    # [{'params': ['net1.weight', 'net2.weight'], 'lr': 0.001}, {'params': ['net3.weight'], 'lr': 0.002}]
                    # Then we need change the code here
                    assert len(param_groups) == 1, "Only one param_group is supported now"
                    init_defaults = param_groups[0]
                    init_defaults.pop("params")
                    for param_group in cast(List[Dict[str, Any]], shard_optimizer_state_dict["param_groups"]):
                        # param_group is like: {'params': ['net1.weight', 'net2.weight'], 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.01}
                        default_keys = list(filter(lambda x: x != "params", param_group.keys()))
                        for key in default_keys:
                            param_group.pop(key)
                        param_group.update(init_defaults)  # lr, betas, eps, etc.

                set_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    optim_state_dict=shard_optimizer_state_dict,
                    options=_set_options,
                )

    def load_dcp_merged(
        self,
        weights_dir: Path,
        load_states: bool = True,
        load_args: bool = True,
    ) -> None:
        """Load a merged model+optimizer checkpoint saved by :meth:`async_save_dcp_merged`.

        Args:
            weights_dir (Path): Directory containing the merged checkpoint.
            load_states (bool): Whether to load optimizer states (momentum, variance, etc.).
            load_args (bool): Whether to load optimizer hyperparameters (lr, betas, etc.).
        """
        _load_options = StateDictOptions(
            cpu_offload=True, ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params
        )
        if self.has_freeze_params:
            _set_options = StateDictOptions(cpu_offload=True, strict=False)
        else:
            _set_options = StateDictOptions(cpu_offload=True, strict=True)

        with profile_time_and_memory(f"[Load DCP Merged from {weights_dir}]"):
            model_sd = get_model_state_dict(self.model, options=_load_options)
            optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=_load_options)
            merged: dict[str, Any] = {"model": model_sd, "optimizer": optim_sd}

            dcp.load(state_dict=merged, checkpoint_id=weights_dir)

            set_model_state_dict(self.model, merged["model"], options=_set_options)

            shard_optimizer_state_dict = merged["optimizer"]
            if not load_states:
                logger.info("Not loading optimizer states")
                shard_optimizer_state_dict["state"] = {}
            if not load_args:
                logger.info("Not loading arg defaults")
                param_groups = self.optimizer.state_dict()["param_groups"]
                assert len(param_groups) == 1, "Only one param_group is supported now"
                init_defaults = param_groups[0]
                init_defaults.pop("params")
                for param_group in cast(List[Dict[str, Any]], shard_optimizer_state_dict["param_groups"]):
                    default_keys = list(filter(lambda x: x != "params", param_group.keys()))
                    for key in default_keys:
                        param_group.pop(key)
                    param_group.update(init_defaults)

            set_optimizer_state_dict(
                self.model,
                self.optimizer,
                optim_state_dict=shard_optimizer_state_dict,
                options=_set_options,
            )

    def put_model_to_device(self, device: torch.device | str):
        """Put the model to the given device."""
        self.model.to_device(device)
        return

    def put_optimizer_to_device(self, device: torch.device | str):
        """Put the optimizer to the given device."""
        if self.fsdp_cfg.cpu_offload:
            return
        if not self.optimizer.state:
            return
        for state in self.optimizer.state.values():
            if isinstance(state, dict):
                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        state[key] = val.to(device, non_blocking=True)
        DEVICE_MODULE.synchronize()
        return

    def _maybe_precompute_float8_dynamic_scale_for_fsdp(self):
        for model in self.model.modules():
            if isinstance(model, BaseModel) and model.float8_handler is not None:
                model.float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)

    def _get_total_loss(self, model_outputs: ModelOutputs) -> torch.Tensor:
        # TODO: This logic should be moved into the model layer. The model should be responsible
        # for aggregating all losses (CE loss, balancing loss, z loss, etc.) and returning a
        # single total_loss. The engine should only call model.forward() and use the returned
        # total_loss directly, rather than iterating through fields to sum losses here.
        # This would provide better separation of concerns and make the loss computation logic
        # more explicit and maintainable.
        loss = torch.tensor(0.0, device=DEVICE)
        for key in model_outputs.model_fields:
            value = getattr(model_outputs, key)
            if "loss" in key and isinstance(value, torch.Tensor):
                loss += value
        return loss
