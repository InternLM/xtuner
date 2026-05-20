from __future__ import annotations

import importlib
import json
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
)

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss import LogProbContext
from xtuner.v1.model.base import (
    AsyncHFSaveHandle,
    BaseModel,
    BatchForwardInfo,
    DataBatchInfo,
    ModelItem,
    ModelOutputs,
    XTunerBaseModelConfig,
)
from xtuner.v1.patch.xtuner_storage import XtunerCacheWriter
from xtuner.v1.profiler.prober import ProberList
from xtuner.v1.utils import (
    get_device,
    get_logger,
    get_torch_device_module,
    log_rank0,
    profile_time_and_memory,
)
from xtuner.v1.utils.grad_norm import cal_grad_norm


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
            log_rank0.warning(f"{key} not in checkpoint.")
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
        async_hf_export: bool = False,
    ) -> None:
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.fsdp_cfg = fsdp_cfg
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(optim_cfg)
        self.intra_layer_micro_batch = intra_layer_micro_batch
        self._count = 0
        self.has_freeze_params = self.__has_freeze_params()
        self._async_hf_export = async_hf_export
        self._async_checkpoint_pg: dist.ProcessGroup | None = None
        self._async_state_dict_cache: dict[str, Any] | None = None

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
            log_rank0.info(model)
        return model

    def build_optimizer(self, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
        return optim_cfg.build(self.model)

    @property
    def data_replicate_size(self) -> int:
        # todo: consider pp
        return self.fsdp_cfg.tp_size

    @torch.no_grad()
    def forward_only(self, seq_ctx: SequenceContext, loss_ctx: LogProbContext):
        output = self.model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})  # type: ignore[call-overload]
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
            log_rank0.info(f"grad_accumulation_steps: {iters_per_step}")
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
                    loss_ctx=loss_ctx_list,  # type: ignore[arg-type]
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
            log_rank0.warning(f"Gradient norm {grad_norm} is invalid, skipping optimizer step.")
            self.optimizer.zero_grad()
        elif (
            self.optim_cfg.skip_grad_norm_threshold is not None and grad_norm > self.optim_cfg.skip_grad_norm_threshold
        ):
            log_rank0.warning(
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

    def async_save_hf(
        self,
        hf_dir: str,
        save_dtype: torch.dtype = torch.bfloat16,
        cleanup_hf_dirs: Sequence[str | Path] = (),
    ) -> AsyncHFSaveHandle:
        return self.model.async_save_hf(
            hf_dir=hf_dir,
            save_dtype=save_dtype,
            cleanup_hf_dirs=cleanup_hf_dirs,
        )

    def wait_async_hf(self, handle: AsyncHFSaveHandle | None = None) -> Path | None:
        return self.model.wait_async_hf(handle)

    def _get_dcp_state_dict(
        self,
        *,
        cpu_offload: bool,
        save_optimizer: bool = True,
    ) -> dict[str, Any]:
        options = StateDictOptions(
            cpu_offload=cpu_offload,
            ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params,
        )
        state_dict: dict[str, Any] = {}

        with profile_time_and_memory("[DCP Collect Model State Dict]"):
            state_dict["model"] = get_model_state_dict(self.model, options=options)

        if save_optimizer:
            with profile_time_and_memory("[DCP Collect Optimizer State Dict]"):
                state_dict["optimizer"] = get_optimizer_state_dict(self.model, self.optimizer, options=options)

        return state_dict

    def save_dcp(
        self,
        weights_dir: Path,
        save_optimizer: bool = True,
    ) -> None:
        if dist.get_rank() == 0:
            weights_dir.mkdir(parents=True, exist_ok=True)

        state_dict = self._get_dcp_state_dict(cpu_offload=True, save_optimizer=save_optimizer)

        with profile_time_and_memory(f"[DCP save for {weights_dir}]"):
            dcp.save(
                state_dict,
                checkpoint_id=weights_dir,
            )

    def _get_async_checkpoint_pg(self) -> dist.ProcessGroup:
        if self._async_checkpoint_pg is None:
            # dcp.async_save() performs collectives from a background thread.
            # Keep those gloo collectives off the training NCCL process group
            # to avoid cross-thread communication conflicts.
            self._async_checkpoint_pg = dist.new_group(backend="gloo")
        return self._async_checkpoint_pg

    @staticmethod
    def _is_async_checkpoint_daemon_init_error(exc: BaseException) -> bool:
        message = str(exc)
        return (
            "EADDRINUSE" in message
            or "address already in use" in message
            or "Checkpoint background process is dead" in message
        )

    def async_save_dcp(
        self,
        weights_dir: Path,
        save_optimizer: bool = True,
    ) -> Future:
        async_checkpoint_pg = self._get_async_checkpoint_pg()

        # Match async HF export semantics: write the DCP payload into a
        # temporary .incomplete directory and commit it only after every rank's
        # async_save future has completed.
        incomplete_dir = weights_dir.with_name(f"{weights_dir.name}.incomplete")
        if weights_dir.exists():
            raise FileExistsError(f"Checkpoint directory already exists: {weights_dir}")
        if dist.get_rank() == 0:
            if incomplete_dir.exists():
                shutil.rmtree(incomplete_dir)
            incomplete_dir.mkdir(parents=True, exist_ok=True)

        # XtunerCacheWriter.stage() creates its staging cache directly in POSIX
        # shared memory (/dev/shm). PyTorch's ForkingPickler detects
        # shared-memory tensors and sends fd handles (no data copy) to the
        # checkpoint process, avoiding PSS amplification.
        # The shm cache is reused across checkpoints via self._async_state_dict_cache.
        state_dict = self._get_dcp_state_dict(cpu_offload=False, save_optimizer=save_optimizer)
        storage_writer = self._build_async_storage_writer(incomplete_dir, save_optimizer=save_optimizer)

        if dist.get_rank() == 0:
            logger.info(f"[DCP async_save for {weights_dir}] async_checkpointer_type=process")

        t0 = time.time()

        def start_async_save() -> Future:
            async_save_kwargs: dict[str, Any] = {}
            state_dict_saver = importlib.import_module("torch.distributed.checkpoint.state_dict_saver")
            async_checkpointer_type = getattr(state_dict_saver, "AsyncCheckpointerType", None)
            if async_checkpointer_type is not None:
                async_save_kwargs["async_checkpointer_type"] = async_checkpointer_type.PROCESS

            with profile_time_and_memory(f"[DCP async_save for {weights_dir}]"):
                return cast(Any, dcp.async_save)(
                    state_dict,
                    checkpoint_id=incomplete_dir,
                    storage_writer=storage_writer,
                    process_group=async_checkpoint_pg,
                    **async_save_kwargs,
                )

        dcp_future = start_async_save()

        def commit_async_save() -> None:
            nonlocal dcp_future
            # Retry only PyTorch DCP daemon init port races, such as
            # EADDRINUSE from TCPStore. Other checkpoint failures still raise.
            max_daemon_init_attempts = 3
            for attempt in range(1, max_daemon_init_attempts + 1):
                try:
                    dcp_future.result()
                    break
                except BaseException as exc:
                    if attempt == max_daemon_init_attempts or not self._is_async_checkpoint_daemon_init_error(exc):
                        elapsed = time.time() - t0
                        logger.error(f"[DCP async_save for {weights_dir}] failed after {elapsed:.2f}s: {exc}")
                        logger.error(traceback.format_exc())
                        raise

                    if dist.get_rank() == 0:
                        logger.warning(
                            "[DCP async_save for %s] checkpoint daemon init failed on attempt %s/%s, retrying: %s",
                            weights_dir,
                            attempt,
                            max_daemon_init_attempts,
                            exc,
                        )
                        if incomplete_dir.exists():
                            shutil.rmtree(incomplete_dir)
                        incomplete_dir.mkdir(parents=True, exist_ok=True)
                    dist.barrier(group=async_checkpoint_pg)
                    dcp_future = start_async_save()

            dist.barrier(group=async_checkpoint_pg)
            if dist.get_rank() == 0:
                incomplete_dir.rename(weights_dir)
            dist.barrier(group=async_checkpoint_pg)

            # Propagate the staging cache created by XtunerCacheWriter.stage()
            # so the next checkpoint reuses the same buffers.
            self._async_state_dict_cache = storage_writer.state_dict_cache

            elapsed = time.time() - t0
            logger.info(f"[DCP async_save for {weights_dir}] finished in {elapsed:.2f}s")

        commit_executor = ThreadPoolExecutor(max_workers=1)
        commit_future = commit_executor.submit(commit_async_save)
        commit_future.add_done_callback(lambda _: commit_executor.shutdown(wait=False))
        return commit_future

    def _build_async_storage_writer(self, weights_dir: Path, *, save_optimizer: bool) -> XtunerCacheWriter:
        # XtunerCacheWriter.stage() builds the staging cache in POSIX shared
        # memory so ForkingPickler can transfer tensors to the daemon via fd
        # handles (no copy).
        # XTuner creates one writer per checkpoint path; carry the cache across
        # writers via self._async_state_dict_cache to avoid re-allocation.
        # Keep cache_staged_state_dict=True to preserve steady-state performance.

        if dist.get_rank() == 0:
            logger.info("[DCP async_save] XtunerCacheWriter cache_staged_state_dict=True")

        storage_writer = XtunerCacheWriter(weights_dir, cache_staged_state_dict=True)
        storage_writer.state_dict_cache = self._async_state_dict_cache
        return storage_writer

    def destroy_async_checkpoint_pg(self) -> None:
        """Destroy the dedicated gloo process group used for async
        checkpoint."""
        self._async_state_dict_cache = None
        if self._async_checkpoint_pg is not None:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group(self._async_checkpoint_pg)
            self._async_checkpoint_pg = None

    def __del__(self) -> None:
        try:
            self.destroy_async_checkpoint_pg()
        except Exception:
            pass

    def load_dcp(
        self,
        weights_dir: Path,
        load_states: bool = True,
        load_args: bool = True,
    ) -> None:
        """Load a DCP checkpoint saved in the merged weights format.

        If the checkpoint does not contain optimizer states, only model weights will be loaded regardless of
        load_states/load_args settings.
        """
        load_optimizer = load_states or load_args
        state_dict = self._get_dcp_state_dict(cpu_offload=True, save_optimizer=load_optimizer)

        if self.has_freeze_params:
            set_options = StateDictOptions(cpu_offload=True, strict=False)
        else:
            set_options = StateDictOptions(cpu_offload=True, strict=True)

        with profile_time_and_memory(f"[Load DCP from {weights_dir}]"):
            dcp.load(state_dict=state_dict, checkpoint_id=weights_dir)

            set_model_state_dict(self.model, state_dict["model"], options=set_options)

            if not load_optimizer:
                return

            optimizer_state_dict = state_dict["optimizer"]
            if not load_states:
                logger.info("Not loading optimizer states")
                optimizer_state_dict["state"] = {}
            if not load_args:
                logger.info("Not loading arg defaults")
                param_groups = self.optimizer.state_dict()["param_groups"]
                # Now we only support one param_group. If we want to support different lr for different parameters,
                # we may use multiple param_groups like:
                assert len(param_groups) == 1, "Only one param_group is supported now"
                init_defaults = param_groups[0]
                init_defaults.pop("params")
                for param_group in cast(List[Dict[str, Any]], optimizer_state_dict["param_groups"]):
                    default_keys = list(filter(lambda x: x != "params", param_group.keys()))
                    for key in default_keys:
                        param_group.pop(key)
                    param_group.update(init_defaults)

            set_optimizer_state_dict(
                self.model,
                self.optimizer,
                optim_state_dict=optimizer_state_dict,
                options=set_options,
            )

    def put_model_to_device(self, device: torch.device | str):
        """Put the model to the given device."""
        self.model.to_device(device)
        return

    def put_optimizer_to_device(self, device: torch.device | str):
        """Put the optimizer to the given device."""
        if self.fsdp_cfg.cpu_offload or self.optim_cfg.swap_optimizer:
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
