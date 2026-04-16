import json
import os
import hashlib
import threading
from concurrent.futures import wait
from pathlib import Path
from typing import Any, Dict, List, cast

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
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
)

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.loss import LogProbContext
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
    ) -> None:
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.fsdp_cfg = fsdp_cfg
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(optim_cfg)
        self.intra_layer_micro_batch = intra_layer_micro_batch
        self._count = 0
        self.has_freeze_params = self.__has_freeze_params()
        self._async_model_tensor_cache = {}
        self._async_optimizer_tensor_cache = {}
        self._async_hf_tensor_cache = {}
        self._last_async_snapshot = None
        self._last_async_hf_snapshot = None

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

    @staticmethod
    def _async_hf_writer_status_filename(rank: int, world_size: int) -> str:
        return f"async-hf-writer-status-rank-{rank:05d}-of-{world_size:05d}.json"

    @staticmethod
    def _async_rank_filename(rank: int, world_size: int) -> str:
        return f"shard-rank-{rank:05d}-of-{world_size:05d}.pt"

    @staticmethod
    def _async_writer_status_filename(rank: int, world_size: int) -> str:
        return f"async-writer-status-rank-{rank:05d}-of-{world_size:05d}.json"

    @staticmethod
    def _compute_file_hash(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _resolve_async_rank_file(cls, ckpt_dir: Path, rank: int, world_size: int) -> Path:
        preferred = ckpt_dir / cls._async_rank_filename(rank, world_size)
        if preferred.exists():
            return preferred

        legacy = ckpt_dir / f"rank_{rank:05d}.pt"
        if legacy.exists():
            return legacy

        raise FileNotFoundError(f"Async checkpoint shard not found for rank={rank} under {ckpt_dir}")

    def _allocate_cpu_buffer_like(self, tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()

        cpu_tensor = tensor.detach().to("cpu")
        if tensor.is_cuda:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor

    def _get_or_update_cpu_tensor(
        self,
        tensor: torch.Tensor,
        cache: dict[tuple[Any, ...], torch.Tensor],
        path: tuple[Any, ...],
    ) -> torch.Tensor:
        detached = tensor.detach()
        if isinstance(detached, DTensor):
            detached = detached.to_local().detach()

        cached = cache.get(path)
        if cached is None or (
            cached.shape != detached.shape
            or cached.dtype != detached.dtype
            or cached.layout != detached.layout
            or cached.stride() != detached.stride()
        ):
            cached = self._allocate_cpu_buffer_like(detached)
            cache[path] = cached

        cached.copy_(detached, non_blocking=detached.is_cuda)
        return cached

    def _copy_state_dict_to_cpu_snapshot(
        self,
        obj: Any,
        cache: dict[tuple[Any, ...], torch.Tensor],
        path: tuple[Any, ...] = (),
    ) -> Any:
        if torch.is_tensor(obj):
            return self._get_or_update_cpu_tensor(obj, cache, path)

        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                result[key] = self._copy_state_dict_to_cpu_snapshot(value, cache, path + (("dict", key),))
            return result

        if isinstance(obj, list):
            return [
                self._copy_state_dict_to_cpu_snapshot(value, cache, path + (("list", idx),))
                for idx, value in enumerate(obj)
            ]

        if isinstance(obj, tuple):
            return tuple(
                self._copy_state_dict_to_cpu_snapshot(value, cache, path + (("tuple", idx),))
                for idx, value in enumerate(obj)
            )

        return obj

    def prepare_async_hf_snapshot(
        self,
        hf_dir: Path,
        save_dtype: torch.dtype = torch.bfloat16,
        safetensors_prefix: str = "model",
        verify_hash: bool = False,
    ) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        status_file = hf_dir.parent / self._async_hf_writer_status_filename(rank, world_size)

        hf_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        file_to_names: list[tuple[str, list[str]]] = []
        weight_map: dict[str, str] = {}

        for safetensor_name, name_list, hf_tensor_list in self.model._iter_hf_save_chunks(
            save_dtype=save_dtype,
            safetensors_prefix=safetensors_prefix,
            device=DEVICE,
        ):
            cached_names: list[str] = []
            for name, hf_tensor in zip(name_list, hf_tensor_list):
                cache_key = (("root", "hf"), ("name", name))
                self._get_or_update_cpu_tensor(
                    hf_tensor,
                    cache=self._async_hf_tensor_cache,
                    path=cache_key,
                )
                cached_names.append(name)
                weight_map[name] = safetensor_name
            if cached_names:
                file_to_names.append((safetensor_name, cached_names))
            del hf_tensor_list

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._last_async_hf_snapshot = {
            "rank": rank,
            "hf_dir": hf_dir,
            "file_to_names": file_to_names,
            "weight_map": weight_map,
            "status_file": status_file,
            "verify_hash": verify_hash,
        }

    def write_async_hf_snapshot(self) -> None:
        if self._last_async_hf_snapshot is None:
            raise RuntimeError("No async HF snapshot prepared")

        snapshot = self._last_async_hf_snapshot
        status_file: Path = snapshot["status_file"]
        verify_hash = bool(snapshot.get("verify_hash", False))
        local_status: dict[str, Any] = {
            "rank": snapshot["rank"],
            "ok": True,
            "error": "",
            "hashes": {},
            "weight_map": {},
        }

        try:
            hf_dir = cast(Path, snapshot["hf_dir"])
            file_to_names = cast(list[tuple[str, list[str]]], snapshot["file_to_names"])
            weight_map = cast(dict[str, str], snapshot["weight_map"])
            written_files: list[str] = []
            for filename, names in file_to_names:
                tensors: dict[str, torch.Tensor] = {}
                for name in names:
                    cache_key = (("root", "hf"), ("name", name))
                    cached_tensor = cast(torch.Tensor | None, self._async_hf_tensor_cache.get(cache_key))
                    if cached_tensor is None:
                        raise RuntimeError(f"Missing cached async HF tensor for key: {name}")
                    tensors[name] = cached_tensor
                self.model._write_hf_save_plan({"hf_dir": hf_dir, "save_tasks": [(filename, tensors)]})
                if tensors:
                    written_files.append(filename)

            local_status["weight_map"] = weight_map
            if verify_hash:
                for filename in written_files:
                    local_status["hashes"][filename] = self._compute_file_hash(hf_dir / filename)
        except Exception as exc:
            local_status["ok"] = False
            local_status["error"] = str(exc)
            with status_file.open("w") as f:
                f.write(json.dumps(local_status, indent=2))
            raise

        with status_file.open("w") as f:
            f.write(json.dumps(local_status, indent=2))

    def prepare_async_dcp_snapshot(
        self,
        model_dir: Path,
        optimizer_dir: Path | None = None,
        verify_hash: bool = False,
    ) -> None:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        status_file = model_dir.parent / self._async_writer_status_filename(rank, world_size)

        if rank == 0:
            model_dir.mkdir(parents=True, exist_ok=True)
            if optimizer_dir is not None:
                optimizer_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        options = StateDictOptions(cpu_offload=False, ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params)

        model_state = get_model_state_dict(self.model, options=options)
        model_snapshot = self._copy_state_dict_to_cpu_snapshot(
            model_state,
            cache=self._async_model_tensor_cache,
            path=(("root", "model"),),
        )

        optimizer_snapshot = None
        if optimizer_dir is not None:
            optimizer_state = get_optimizer_state_dict(self.model, self.optimizer, options=options)
            optimizer_snapshot = self._copy_state_dict_to_cpu_snapshot(
                optimizer_state,
                cache=self._async_optimizer_tensor_cache,
                path=(("root", "optimizer"),),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._last_async_snapshot = {
            "rank": rank,
            "world_size": world_size,
            "model_dir": model_dir,
            "optimizer_dir": optimizer_dir,
            "model_state": model_snapshot,
            "optimizer_state": optimizer_snapshot,
            "status_file": status_file,
            "verify_hash": verify_hash,
        }

    def write_async_dcp_snapshot(self) -> None:
        if self._last_async_snapshot is None:
            raise RuntimeError("No async DCP snapshot prepared")

        snapshot = self._last_async_snapshot
        rank = snapshot["rank"]
        world_size = snapshot["world_size"]
        verify_hash = bool(snapshot.get("verify_hash", False))
        status_file: Path = snapshot["status_file"]
        local_status: dict[str, Any] = {"rank": rank, "ok": True, "error": "", "hashes": {}}

        try:
            model_file = snapshot["model_dir"] / self._async_rank_filename(rank, world_size)
            torch.save(snapshot["model_state"], model_file)
            if not model_file.exists() or model_file.stat().st_size <= 0:
                raise RuntimeError(f"model shard missing or empty after save: {model_file}")
            if verify_hash:
                local_status["hashes"][f"model/{model_file.name}"] = self._compute_file_hash(model_file)

            if snapshot["optimizer_dir"] is not None and snapshot["optimizer_state"] is not None:
                optimizer_file = snapshot["optimizer_dir"] / self._async_rank_filename(rank, world_size)
                torch.save(snapshot["optimizer_state"], optimizer_file)
                if not optimizer_file.exists() or optimizer_file.stat().st_size <= 0:
                    raise RuntimeError(f"optimizer shard missing or empty after save: {optimizer_file}")
                if verify_hash:
                    local_status["hashes"][f"optimizer/{optimizer_file.name}"] = self._compute_file_hash(optimizer_file)
        except Exception as exc:
            local_status["ok"] = False
            local_status["error"] = str(exc)
            with status_file.open("w") as f:
                f.write(json.dumps(local_status, indent=2))
            raise

        with status_file.open("w") as f:
            f.write(json.dumps(local_status, indent=2))

    def _restore_state_dict_from_local_checkpoint(self, template: Any, loaded: Any) -> Any:
        if torch.is_tensor(template):
            if isinstance(template, DTensor):
                local_tensor = template.to_local()
                local_tensor.copy_(loaded.to(device=local_tensor.device, dtype=local_tensor.dtype))
                return template

            template.copy_(loaded.to(device=template.device, dtype=template.dtype))
            return template

        if isinstance(template, dict):
            return {
                key: self._restore_state_dict_from_local_checkpoint(value, loaded[key])
                for key, value in template.items()
            }

        if isinstance(template, list):
            return [
                self._restore_state_dict_from_local_checkpoint(template_item, loaded_item)
                for template_item, loaded_item in zip(template, loaded, strict=True)
            ]

        if isinstance(template, tuple):
            return tuple(
                self._restore_state_dict_from_local_checkpoint(template_item, loaded_item)
                for template_item, loaded_item in zip(template, loaded, strict=True)
            )

        return loaded

    # TODO: Support async save
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
        with profile_time_and_memory(f"[DCP Checkpoint to {model_dir}]"):
            model_state = get_model_state_dict(self.model, options=_options)
            dcp.save(
                model_state,
                checkpoint_id=model_dir,
            )

        with profile_time_and_memory(f"[DCP Checkpoint to {optimizer_dir}]"):
            if optimizer_dir is not None:
                shard_optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=_options)
                dcp.save(
                    shard_optimizer_state_dict,
                    checkpoint_id=optimizer_dir,
                )

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

    def load_async_checkpoint(
        self,
        model_dir: Path,
        optimizer_dir: Path | None = None,
        load_states: bool = True,
        load_args: bool = True,
        verify_hash: bool = False,
        verify_hash_global: bool = True,
        hash_algo: str | None = None,
        expected_shard_hashes: dict[str, str] | None = None,
        strict_topology: bool = True,
        expected_world_size: int | None = None,
        expected_topology: dict[str, int] | None = None,
        runtime_topology: dict[str, int] | None = None,
    ):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if strict_topology:
            if expected_world_size is None:
                raise RuntimeError("Async checkpoint metadata missing world_size, strict topology check cannot proceed")
            if expected_world_size != world_size:
                raise RuntimeError(
                    f"Async checkpoint world_size mismatch: checkpoint={expected_world_size}, runtime={world_size}"
                )

            if expected_topology is not None and runtime_topology is not None:
                for key in ("dp_size", "sp_size", "tp_size"):
                    if key in expected_topology and key in runtime_topology:
                        if int(expected_topology[key]) != int(runtime_topology[key]):
                            raise RuntimeError(
                                "Async checkpoint topology mismatch: "
                                f"{key} checkpoint={expected_topology[key]}, runtime={runtime_topology[key]}"
                            )

        model_file = self._resolve_async_rank_file(model_dir, rank, world_size)
        optimizer_file = self._resolve_async_rank_file(optimizer_dir, rank, world_size) if optimizer_dir else None

        if verify_hash:
            local_ok = True
            local_error = ""
            if hash_algo != "sha256":
                local_ok = False
                local_error = f"Unsupported async checkpoint hash algorithm: {hash_algo}"
            elif expected_shard_hashes is None:
                local_ok = False
                local_error = "Missing shard_hashes in async checkpoint metadata"
            else:
                try:
                    model_key = f"model/{model_file.name}"
                    expected_model_hash = expected_shard_hashes.get(model_key)
                    actual_model_hash = self._compute_file_hash(model_file)
                    if not expected_model_hash:
                        local_ok = False
                        local_error = f"Missing expected hash for {model_key}"
                    elif actual_model_hash != expected_model_hash:
                        local_ok = False
                        local_error = (
                            f"Hash mismatch for {model_key}: expected={expected_model_hash}, actual={actual_model_hash}"
                        )

                    if local_ok and optimizer_file is not None:
                        optimizer_key = f"optimizer/{optimizer_file.name}"
                        expected_optimizer_hash = expected_shard_hashes.get(optimizer_key)
                        actual_optimizer_hash = self._compute_file_hash(optimizer_file)
                        if not expected_optimizer_hash:
                            local_ok = False
                            local_error = f"Missing expected hash for {optimizer_key}"
                        elif actual_optimizer_hash != expected_optimizer_hash:
                            local_ok = False
                            local_error = (
                                f"Hash mismatch for {optimizer_key}: "
                                f"expected={expected_optimizer_hash}, actual={actual_optimizer_hash}"
                            )
                except Exception as exc:
                    local_ok = False
                    local_error = f"Compute hash failed: {exc}"

            if verify_hash_global:
                local_status = {"rank": rank, "ok": local_ok, "error": local_error}
                all_status: list[dict[str, Any]] = [None for _ in range(world_size)]  # type: ignore[list-item]
                dist.all_gather_object(all_status, local_status)
                if not all(status["ok"] for status in all_status):
                    failed = ", ".join(
                        f"rank={status['rank']}({status['error']})" for status in all_status if not status["ok"]
                    )
                    raise RuntimeError(f"Async checkpoint load hash verification failed: {failed}")
            elif not local_ok:
                raise RuntimeError(f"Async checkpoint load hash verification failed on rank={rank}: {local_error}")

        _load_options = StateDictOptions(
            cpu_offload=False, ignore_frozen_params=self.model_cfg.dcp_ignore_frozen_params
        )
        if self.has_freeze_params:
            _set_options = StateDictOptions(cpu_offload=False, strict=False)
        else:
            _set_options = StateDictOptions(cpu_offload=False, strict=True)

        with profile_time_and_memory(f"[Load Async Checkpoint Model from {model_dir}]"):
            shard_model_state_dict = get_model_state_dict(self.model, options=_load_options)
            loaded_model_state = torch.load(model_file, map_location="cpu")
            shard_model_state_dict = self._restore_state_dict_from_local_checkpoint(
                shard_model_state_dict, loaded_model_state
            )
            set_model_state_dict(self.model, shard_model_state_dict, options=_set_options)

        if optimizer_dir is not None:
            with profile_time_and_memory(f"[Load Async Checkpoint Optimizer] from {optimizer_dir}"):
                shard_optimizer_state_dict = get_optimizer_state_dict(
                    self.model, self.optimizer, options=_load_options
                )
                assert optimizer_file is not None
                loaded_optimizer_state = torch.load(optimizer_file, map_location="cpu")
                shard_optimizer_state_dict = self._restore_state_dict_from_local_checkpoint(
                    shard_optimizer_state_dict, loaded_optimizer_state
                )
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
