import json
import os
import threading
import time
from concurrent.futures import Future, wait
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from mmengine import mkdir_or_exist
from safetensors import safe_open
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _has_foreach_support,
)

from xtuner.v1.config import FSDPConfig, OptimConfig, TransformerConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model.base import BaseModel, ModelItem
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


@contextmanager
def profile_time_and_memory(desc):
    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f"{desc} Elapsed time {cost_time:.2f} seconds, peak gpu memory {max_memory / 1024**3:.1f}G")


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
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
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
    float8_handler: Optional[Float8Handler]

    def __init__(
        self,
        model_cfg: TransformerConfig,
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

    def build_model(self) -> BaseModel:
        with torch.device("meta"):
            model = self.model_cfg.build()

        self.float8_handler = None
        if self.model_cfg.float8_cfg is not None and self.model_cfg.float8_cfg.enable_float8:
            self.float8_handler = Float8Handler(
                scaling_granularity_gemm=self.model_cfg.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=self.model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
            )
        model = model.fully_shard(self.fsdp_cfg, self.float8_handler)

        if dist.get_rank() == 0:
            logger.info(model)

        if self.float8_handler:
            self.float8_handler.build_reduce_mesh(model, cast(DeviceMesh, model.fsdp_mesh))
        return model

    def build_optimizer(self, optim_cfg: OptimConfig) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]

        trainable_parameters_names = self.model.trainable_parameters()
        trainable_names = [name for name, _ in trainable_parameters_names]
        untrainable_names = []
        num_total_requires_grad = 0
        num_total = 0
        for name, params_ in self.model.named_parameters():
            num_total += params_.numel()
            num_total_requires_grad += params_.numel() if name in trainable_names else 0
            if name not in trainable_names:
                untrainable_names.append(name)

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad // 1e6}M, total parameters: {num_total // 1e6}M"
            )
            logger.info(f"Untrainable parameters names: {untrainable_names}")
        return optim_cfg.build(params)

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

    def train_step(self, data_batches: list[ModelItem]):
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
        """
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

        loss_log = {}
        other_log = {}
        intra_layer_micro_batch = self.intra_layer_micro_batch
        assert len(data_batches) % intra_layer_micro_batch == 0, (
            f"data_batches length {len(data_batches)} is not divisible by intra_layer_micro_batch {intra_layer_micro_batch}"
        )
        iters_per_step = self.grad_accumulation_steps(len(data_batches))

        moe_need_update_bias = (
            isinstance(getattr(self.model_cfg, "router", None), NoAuxRouterConfig)
            and self.model_cfg.router.router_bias_update_speed > 0
        )
        if moe_need_update_bias:
            tokens_per_expert_global_for_bias = torch.tensor(0, device=DEVICE)

        step_loss = torch.tensor(0.0, device=DEVICE)
        step_llm_loss = torch.tensor(0.0, device=DEVICE)
        step_balancing_loss: torch.Tensor | None = None
        step_z_loss: torch.Tensor | None = None
        step_consumed_tokens = torch.tensor(0.0, device=DEVICE)

        if self._count == 0:
            logger.info(f"grad_accumulation_steps: {iters_per_step}")
            self._count += 1

        for i in range(0, len(data_batches), intra_layer_micro_batch):
            data_batch = data_batches[i : i + intra_layer_micro_batch]
            seq_ctx_list = []
            loss_ctx_list = []
            for data in data_batch:
                seq_ctx = data["seq_ctx"]
                loss_ctx = data["loss_ctx"]
                seq_ctx_list.append(seq_ctx)
                loss_ctx_list.append(loss_ctx)
                step_consumed_tokens += seq_ctx.mask.sum()

            if self.intra_layer_micro_batch == 1:
                output = self.model(seq_ctx=seq_ctx_list[0], loss_ctx=loss_ctx_list[0])
            else:
                # For intra_layer_micro_batch > 1, we need to handle the data batches differently.
                # Here we assume that the model can handle a list of seq_ctx and loss_ctx.
                output = self.model(
                    seq_ctx=seq_ctx_list,
                    loss_ctx=loss_ctx_list,
                )

            # llm loss has been global averaged
            llm_loss = output["loss"]
            step_llm_loss += llm_loss.detach().clone()

            loss = llm_loss
            if "balancing_loss" in output:
                loss = loss + output["balancing_loss"] / iters_per_step
                step_balancing_loss = (
                    output["balancing_loss"]
                    if step_balancing_loss is None
                    else step_balancing_loss + output["balancing_loss"]
                )
            if "z_loss" in output:
                loss = loss + output["z_loss"] / iters_per_step
                step_z_loss = output["z_loss"] if step_z_loss is None else step_z_loss + output["z_loss"]

            if moe_need_update_bias:
                assert "tokens_per_expert_global" in output, "tokens_per_expert_global is required for bias update."
                tokens_per_expert_global_for_bias += output["tokens_per_expert_global"]

            del output
            loss.backward()
            step_loss += loss.detach().clone()

        if moe_need_update_bias:
            avg_count_load = tokens_per_expert_global_for_bias.float().mean(1)
            max_load_i, _ = torch.max(tokens_per_expert_global_for_bias, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            maxvio = maxvio_all_layers.mean()
            self.model.update_bias(tokens_per_expert_global_for_bias, avg_count_load)  # type: ignore
            other_log["maxvio"] = maxvio.item()

        reduced_llm_loss = step_llm_loss
        dist.all_reduce(reduced_llm_loss.div_(dist.get_world_size()))

        loss_log["total_loss"] = step_loss.item()
        loss_log["reduced_llm_loss"] = reduced_llm_loss.item()
        if step_balancing_loss is not None:
            reduced_balancing_loss = step_balancing_loss
            dist.all_reduce(reduced_balancing_loss.div_(dist.get_world_size()))
            loss_log["reduced_balancing_loss"] = reduced_balancing_loss.item()
        if step_z_loss is not None:
            reduced_z_loss = step_z_loss
            dist.all_reduce(reduced_z_loss.div_(dist.get_world_size()))
            loss_log["reduced_z_loss"] = reduced_z_loss.item()
        other_log["consumed_tokens"] = step_consumed_tokens.item()
        return loss_log, other_log

    def from_hf(self, hf_path: str | Path, strict: bool = False):
        self.model.from_hf(hf_path=hf_path, strict=strict)

    def init_model_weights(self):
        self.model.init_weights()

    @staticmethod
    def group_tensors_by_device_mesh_and_placements(tensors: List[torch.Tensor]):
        grouped_tensors: Dict[Tuple[DeviceMesh, Tuple[Placement, ...]], List[torch.Tensor]] = {}
        for tensor in tensors:
            assert isinstance(tensor, DTensor)
            key = (tensor.device_mesh, tensor.placements)
            if key in grouped_tensors:
                grouped_tensors[key].append(tensor)
            else:
                grouped_tensors[key] = [tensor]
        return grouped_tensors

    def cal_total_norm(self, tensors: List[DTensor], norm_type: float = 2.0, foreach: Optional[bool] = None):
        norm_type = float(norm_type)
        if len(tensors) == 0:
            return torch.tensor(0.0)

        device_mesh: DeviceMesh = tensors[0].device_mesh
        placements = tensors[0].placements
        device = tensors[0].device
        norms: Tuple[DTensor, ...]
        if (foreach is None and _has_foreach_support(tensors, device)) or (  # type: ignore
            foreach and _device_has_foreach_support(device)
        ):
            norms = torch._foreach_norm(tensors, norm_type)  # type: ignore
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms = tuple(torch.linalg.vector_norm(g, norm_type) for g in tensors)

        local_norm = torch.linalg.vector_norm(
            torch.stack([norm.to_local() for norm in norms]), norm_type, dtype=torch.float32
        )
        if norm_type == 2:
            local_norm_squared = local_norm**2
            for i, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    # When using ep + fsdp, the placement corresponding to fsdp mesh is _StridedShard
                    # isinstance(_StridedShard, Shard) is True
                    dist.all_reduce(local_norm_squared, group=device_mesh.get_group(i))
                elif isinstance(placement, Replicate):
                    pass
                else:
                    raise ValueError(f"Unsupported placement type {placement} in clip_grad_norm")
            global_norm = local_norm_squared**0.5
        else:
            raise NotImplementedError
        return global_norm

    def clip_grad_norm(self):
        self.model.scale_and_reduce_grad()
        params = self.model.trainable_parameters()
        grads = [p.grad for _, p in params if p.grad is not None]
        grouped_grads = self.group_tensors_by_device_mesh_and_placements(grads)
        total_norms = []
        for grads in grouped_grads.values():
            total_norm = self.cal_total_norm(grads, norm_type=2.0, foreach=True)
            total_norms.append(total_norm)
        grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=torch.float32)
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
        return grad_norm

    def step_optimizer(self, grad_norm):
        """Step the optimizer to update the model parameters."""
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm

    @staticmethod
    def clean_param_name(name: str) -> str:
        if "._checkpoint_wrapped_module." in name:
            name = name.replace("._checkpoint_wrapped_module.", ".")
        if "._orig_mod." in name:
            name = name.replace("._orig_mod.", ".")
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
        dcp_dir: str,
        is_snapshot: bool,
        checkpoint_drop_optimizer: bool,
        dcp_task_coordinator: Optional[CPUThreadTaskCoordinator] = None,
    ) -> List[Future]:
        """Save the dcp model to the given directory.

        Args:
            dcp_dir (str): The directory to save the model.
        """
        if is_snapshot:
            checkpoint_drop_optimizer = False
        logger.info(f"[DCP Checkpoint] Start saving PT checkpoint to {dcp_dir} .........")
        rank = dist.get_rank()
        if rank == 0:
            mkdir_or_exist(dcp_dir)

        with profile_time_and_memory("[DCP Checkpoint]"):
            if dcp_task_coordinator is not None:
                dcp_task_coordinator.wait()

            # todo: save lr scheduler state dict
            # if rank == 0:
            #     torch.save(self.lr_scheduler.state_dict(), os.path.join(dcp_dir, "lr_scheduler.pt"))

            _options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)

            if not checkpoint_drop_optimizer:
                shard_optimizer_state_dict = get_optimizer_state_dict(self.model, self.optimizer, options=_options)
                shard_optimizer_state_dict = {"optimizer": shard_optimizer_state_dict}  # type: ignore
                optimizer_dcp_dir = os.path.join(dcp_dir, "optimizer")

                # Must set different gloo group, otherwise empty save
                _optimizer_gloo_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
                optimizer_handle = dcp.async_save(
                    shard_optimizer_state_dict,
                    checkpoint_id=optimizer_dcp_dir,
                    process_group=_optimizer_gloo_group,
                )

            shard_model_state_dict = get_model_state_dict(self.model, options=_options)
            model_state_dict = {"model": shard_model_state_dict}
            model_dcp_dir = os.path.join(dcp_dir, "model")
            # Must set different gloo group, otherwise empty save
            _model_gloo_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=20))
            model_handle = dcp.async_save(
                model_state_dict, checkpoint_id=model_dcp_dir, process_group=_model_gloo_group
            )

            futures = [model_handle]
            if not checkpoint_drop_optimizer:
                futures.append(optimizer_handle)

        return futures

    def load_dcp(self, dcp_dir: str, skip_load_optimizer: bool):
        """Load the dcp model from the given directory.

        Args:
            dcp_dir (str): The directory to load the model from.
        """
        logger.info(f"Load dcp checkpoint in {dcp_dir}")
        with profile_time_and_memory("[Load DCP]"):
            _load_options = StateDictOptions(cpu_offload=True, ignore_frozen_params=True)
            _set_options = StateDictOptions(cpu_offload=True, strict=False)
            if not skip_load_optimizer:
                shard_optimizer_state_dict = get_optimizer_state_dict(
                    self.model, self.optimizer, options=_load_options
                )
                logger.info("====== Loading optimizer state dict ======")
                optimizer_state_dict = {"optimizer": shard_optimizer_state_dict}
                dcp.load(
                    state_dict=optimizer_state_dict,
                    checkpoint_id=os.path.join(dcp_dir, "optimizer"),
                )
                set_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    optim_state_dict=optimizer_state_dict["optimizer"],
                    options=_set_options,
                )

            # todo: resume lr scheduler
            # if not skip_load_scheduler:
            #     logger.info("====== Loading scheduler state dict ======")
            #     lr_scheduler_state_dict = torch.load(os.path.join(dcp_dir, "lr_scheduler.pt"))
            #     self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

            shard_model_state_dict = get_model_state_dict(self.model, options=_load_options)
            logger.info("====== Loading model state dict ======")
            model_state_dict = {"model": shard_model_state_dict}
            # inplace state_dict
            dcp.load(
                state_dict=model_state_dict,
                checkpoint_id=os.path.join(dcp_dir, "model"),
            )
            set_model_state_dict(self.model, model_state_dict["model"], options=_set_options)

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
