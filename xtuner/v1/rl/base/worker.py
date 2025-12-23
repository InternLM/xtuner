import json
import math
import os
from itertools import chain
from pathlib import Path
from typing import Dict, List, TypeAlias, TypedDict, cast

import ray
import requests
import torch
import torch.distributed as dist
import tqdm
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import (
    VisionComposeTrainEngine,
)
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLForConditionalGeneration
from xtuner.v1.ray.base import SingleAcceleratorWorker
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.rl.utils import gather_logprobs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import (
    ParallelConfigException,
    get_device,
    get_logger,
    get_torch_device_module,
    monkey_unpatch_torch_reductions,
    ray_method,
)
from xtuner.v1.utils.load_spec import LoadEnum

from ..loss_fn import kl_penalty
from .loss import BaseRLLossConfig, RLLossContextInputItem
from .rollout_is import merge_rollout_is_metrics


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def serialize_state_dict(state_dict: dict) -> str:
    """Serialize state dict to str.

    The consumer should use it on same node. As the producer and consumer may
    have different GPU visibility, we use reduce_tensor instead of ForkingPickler.dumps
    to fix the device_id when loading the serialized tensor.

    Args:
        state_dict (dict[str, torch.Tensor]): state dict to serialize.
    Returns:
        str: serialized state dict.
    """
    import base64
    from io import BytesIO
    from multiprocessing.reduction import ForkingPickler

    from torch.multiprocessing.reductions import reduce_tensor

    data = [(k, reduce_tensor(v)) for k, v in state_dict.items()]
    buf = BytesIO()
    ForkingPickler(buf).dump(data)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class WorkerConfig(BaseModel):
    """Training worker configuration for XTuner RL.

    Configuration for RL training workers managing model training, optimization,
    and distributed computing in reinforcement learning workflows.

    Args:
        model_cfg (TransformerConfig): Model architecture configuration.
        optim_cfg (OptimConfig): Optimizer configuration for training.
        loss_cfg (BaseRLLossConfig): Loss function configuration for RL training.
        lr_cfg (LRConfig): Learning rate scheduler configuration.
        fsdp_cfg (FSDPConfig): Fully Sharded Data Parallel configuration.
        load_from (str | Path): Path to load the main model from.
        optimizer_steps (int): Number of optimizer steps per training iteration. Defaults to 1.
        sp_size (int): Sequence parallel size for distributed training. Defaults to 1.
        pack_max_length (int): Maximum sequence length for data packing.
        ref_load_from (str | Path | None): Path to load reference model from.
            If None, uses same as load_from. Defaults to None.
        ref_model_fsdp_cfg (FSDPConfig | None): FSDP configuration for reference model.
            Defaults to None.
        log_dir (str | Path | None): Directory for training logs. Defaults to None.

    **Examples:**

    Example configuration for Basic worker::

        config = WorkerConfig(
            model_cfg=TransformerConfig(model_name="llama2-7b"),
            optim_cfg=OptimConfig(optimizer="adamw"),
            loss_cfg=PPOLossConfig(),
            lr_cfg=LRConfig(lr=1e-5),
            fsdp_cfg=FSDPConfig(),
            load_from="meta-llama/Llama-2-7b-hf",
            pack_max_length=2048
        )

    .. note::
       When ``use_kl_loss=True`` in loss_cfg, a reference model will be loaded
       for KL divergence computation during training.
    """

    model_config = ConfigDict(title="Worker config", extra="forbid", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: BaseRLLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path  # TODO: 把 actor 和 ref 配置分离
    optimizer_steps: int = 1
    sp_size: int = 1
    pack_max_length: int
    ref_load_from: str | Path | None = None
    ref_model_fsdp_cfg: FSDPConfig | None = None
    log_dir: str | Path | None = None
    update_weight_bucket_size_in_gb: float = 0.5  # 512MB


class WorkerInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.LongTensor
    advantages: torch.Tensor
    rollout_logprobs: torch.Tensor | None


class TrainingWorker(SingleAcceleratorWorker):
    _SAVE_OPTIMIZER_DIR = "optimizer"
    _SAVE_MODEL_DIR = "model"

    def __init__(
        self,
        worker_cfg: WorkerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(worker_cfg, rank, master_addr, master_port, world_size, accelerator)
        self.config = cast(WorkerConfig, self.config)
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        self.rank = rank

        # TODO: add lr scheduler
        log_dir = worker_cfg.log_dir
        self.log_dir = None
        if log_dir is not None:
            self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        else:
            self.logger = get_logger()

        if not worker_cfg.fsdp_cfg.torch_compile:
            worker_cfg.model_cfg.compile_cfg = False
        self._engine = self._build_engine(worker_cfg)

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss:
            self._has_ref = True
            if worker_cfg.ref_load_from is None:
                worker_cfg.ref_load_from = worker_cfg.load_from
            self._ref_model = self._build_ref_model(
                worker_cfg.model_cfg, worker_cfg.ref_load_from, worker_cfg.ref_model_fsdp_cfg
            )

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]
        self._optimizer_steps = worker_cfg.optimizer_steps

        # Used to update weight to rollout engine
        self.rollout_device_mesh: DeviceMesh | None = None
        self.rollout_url: str | None = None
        self.rollout_cfg_info: dict = dict()
        self.endpoints: dict[str, str] = dict()
        self.endpoints["update_weights"] = "update_weights"


    def _build_engine(self, worker_cfg: WorkerConfig) -> TrainEngine | VisionComposeTrainEngine:
        if isinstance(worker_cfg.model_cfg, BaseComposeConfig):
            engine = VisionComposeTrainEngine(
                optim_cfg=worker_cfg.optim_cfg,
                fsdp_cfg=worker_cfg.fsdp_cfg,
                model_cfg=worker_cfg.model_cfg,
            )
        else:
            engine = TrainEngine(  # type: ignore
                optim_cfg=worker_cfg.optim_cfg,
                fsdp_cfg=worker_cfg.fsdp_cfg,
                model_cfg=worker_cfg.model_cfg,
            )

        if worker_cfg.load_from is not None:
            engine.from_hf(worker_cfg.load_from)

        if engine.model.compile_cfg is not None and self.rank == 0:
            self.logger.info(f"The `compile_cfg` of model is {json.dumps(engine.model.compile_cfg, indent=4)}")
        return engine

    def _build_ref_model(
        self,
        ref_model_cfg: TransformerConfig | BaseComposeConfig,
        load_from: str | Path,
        ref_model_fsdp_cfg: FSDPConfig | None = None,
    ):
        # TODO: 需要重构，使得能更优雅的兼容 mllm
        model: BaseComposeModel | XtunerBaseModel
        with torch.device("meta"):
            model = ref_model_cfg.build()

        if isinstance(ref_model_cfg, BaseComposeConfig):
            assert ref_model_cfg.text_config.float8_cfg is None, "BaseComposeConfig does not support float8"
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model.language_model.fully_shard(ref_model_fsdp_cfg)  # type: ignore
            model.vision_tower.fully_shard(ref_model_fsdp_cfg)  # type: ignore
            model.multi_modal_projector.fully_shard(ref_model_fsdp_cfg)  # type: ignore
            model = model.fully_shard(ref_model_fsdp_cfg)
            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
        else:
            ref_model_cfg = cast(TransformerConfig, ref_model_cfg)
            if ref_model_cfg.float8_cfg is not None and ref_model_cfg.float8_cfg.enable_float8:
                float8_handler = Float8Handler(
                    scaling_granularity_gemm=ref_model_cfg.float8_cfg.scaling_granularity_gemm,
                    scaling_granularity_grouped_gemm=ref_model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
                )
            else:
                float8_handler = None
            if ref_model_fsdp_cfg is None:
                ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
            model = model.fully_shard(ref_model_fsdp_cfg, float8_handler)  # type: ignore

            model.from_hf(hf_path=load_from)
            model.eval()  # type: ignore
            if float8_handler is not None:
                # As the ref model is not updated, we only compute params' scales once
                float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)  # type: ignore
        model.to_device("cpu")  # type: ignore
        DEVICE_MODULE.empty_cache()
        return model

    def _init_data_mesh(
        self,
        sp_size: int,
    ):
        world_size = dist.get_world_size()
        if world_size % sp_size != 0:
            raise ParallelConfigException(
                f"Found sp_size {sp_size}, world_size {world_size}."
                "sequence parallel size must be a divisor of world size."
            )
        dp_size = world_size // sp_size

        # TODO: fsdp_config could be None
        device = str(DEVICE) if not self.config.fsdp_cfg.cpu_offload else "cpu"

        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        return data_mesh

    def compute_actor_logprobs(
        self, seq_ctx_list: list[SequenceContext], loss_ctx_input_list: list[RLLossContextInputItem]
    ) -> list[RLLossContextInputItem]:
        for seq_ctx, loss_ctx_input in zip(seq_ctx_list, loss_ctx_input_list):
            output = self._engine.forward_only(seq_ctx=seq_ctx)
            loss_ctx_input.old_logprobs = gather_logprobs(output["logits"], loss_ctx_input.shifted_labels)
        return loss_ctx_input_list

    def compute_ref_logprobs(
        self, seq_ctx_list: list[SequenceContext], loss_ctx_input_list: list[RLLossContextInputItem]
    ) -> list[RLLossContextInputItem]:
        assert self._has_ref
        self._ref_model.to_device(DEVICE)
        for seq_ctx, loss_ctx_input in zip(seq_ctx_list, loss_ctx_input_list):
            with torch.no_grad():
                ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx=None)
            ref_logprobs = gather_logprobs(ref_output["logits"], loss_ctx_input.shifted_labels)
            loss_ctx_input.ref_logprobs = ref_logprobs
        self._ref_model.to_device("cpu")
        return loss_ctx_input_list

    def _update_other_log(self, other_log: dict):
        from xtuner.v1.model.utils import ModelForwardExtraLogInfo

        extra_info = other_log.get("extra_info", {})
        if isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info_dict = extra_info.get()
        else:
            extra_info_updated = ModelForwardExtraLogInfo(extra_info)
            extra_info_dict = extra_info_updated.get()
        other_log["extra_info"] = extra_info_dict
        return other_log

    @ray_method
    def fit(self, data_batches: list[WorkerInputItem], rollout_idx: int):
        # NOTE: sglang会清除logger handle, 重新创建
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        loss_cfg = self.config.loss_cfg
        num_batches = len(data_batches)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps)
        if num_batches < self._optimizer_steps:
            self.logger.info(
                f"Optimizer only step once because num_batches {num_batches} < optimizer_steps {self._optimizer_steps}."
            )

        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_input_list: list[RLLossContextInputItem] = []
        rollout_logprobs_list: list[torch.Tensor | None] = []
        # convert dummy padding experts to real size

        language_cfg = (
            self.config.model_cfg.text_config
            if isinstance(self.config.model_cfg, BaseComposeConfig)
            else self.config.model_cfg
        )

        for data in data_batches:
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None:
                if not isinstance(pixel_values, torch.Tensor):
                    assert isinstance(pixel_values, list), (
                        f"pixel_values should be list of tensor, got {type(pixel_values)}"
                    )
                    pixel_values = [ray.get(pixel_obf) for pixel_obf in pixel_values]
                    pixel_values = torch.cat(pixel_values, dim=0)
                    seq_ctx.pixel_values = pixel_values

            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                if isinstance(rollout_routed_experts, list):
                    # list[n,l,e]
                    out_rollout_routed_expert = []
                    for rollout_routed_expert in rollout_routed_experts:
                        if isinstance(rollout_routed_expert, torch.Tensor):
                            rollout_routed_experts_tensor = torch.randint(
                                low=0,
                                high=language_cfg.n_routed_experts,
                                size=(
                                    rollout_routed_expert.size(0),
                                    language_cfg.num_hidden_layers,
                                    language_cfg.num_experts_per_tok,
                                ),
                            )
                            out_rollout_routed_expert.append(rollout_routed_experts_tensor)
                        else:
                            rollout_routed_expert_refs = rollout_routed_expert
                            rollout_routed_expert = ray.get(rollout_routed_expert_refs)
                            # free obj store explicitly
                            ray._private.internal_api.free(rollout_routed_expert_refs)
                            out_rollout_routed_expert.append(torch.as_tensor(rollout_routed_expert, dtype=torch.long))

                    seq_ctx.rollout_routed_experts = torch.cat(out_rollout_routed_expert, dim=0)  # max_len,l,e
                else:
                    assert isinstance(rollout_routed_experts, torch.Tensor), (
                        f"padding experts should be a dummy tensor, bug got {type(rollout_routed_experts)}"
                    )
                    rollout_routed_experts_tensor = torch.randint(
                        low=0,
                        high=language_cfg.n_routed_experts,
                        size=(
                            self.config.pack_max_length,
                            language_cfg.num_hidden_layers,
                            language_cfg.num_experts_per_tok,
                        ),
                    )
                    seq_ctx.rollout_routed_experts = rollout_routed_experts_tensor

                assert seq_ctx.input_ids is not None, "input_ids is None"
                assert seq_ctx.rollout_routed_experts.size(0) == seq_ctx.input_ids.size(1)

            seq_ctx = data["seq_ctx"].to(DEVICE)
            rollout_logprobs = data.get("rollout_logprobs", None)
            if rollout_logprobs is not None:
                rollout_logprobs = rollout_logprobs.to(DEVICE)
                rollout_logprobs_list.append(rollout_logprobs)
            loss_ctx_input = RLLossContextInputItem(
                shifted_labels=data["shifted_labels"],
                advantages=data["advantages"],
                rollout_logprobs=rollout_logprobs,
            ).to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(self.sp_mesh)
                loss_ctx_input = loss_ctx_input.sp_split(self.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx_input_list.append(loss_ctx_input)

        del data_batches

        rank_grad_tokens: torch.Tensor | None = None
        for loss_ctx_input in loss_ctx_input_list:
            mask = loss_ctx_input.shifted_labels != -100
            grad_tokens = mask.sum()
            rank_grad_tokens = grad_tokens if rank_grad_tokens is None else rank_grad_tokens + grad_tokens
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        # old logprobs are inplaced updated in compute_actor_logprobs
        loss_ctx_input_list = self.compute_actor_logprobs(seq_ctx_list, loss_ctx_input_list)
        sum_entropy: torch.Tensor | None = None
        sum_rollout_entropy: torch.Tensor | None = None
        if len(rollout_logprobs_list) > 0:
            assert len(rollout_logprobs_list) == len(loss_ctx_input_list), (
                f"rollout_logprobs_list {len(rollout_logprobs_list)} vs loss_ctx_input_list {len(loss_ctx_input_list)}"
            )

        all_rollout_is_metrics = []
        all_mismatch_metrics = []
        for i, loss_ctx_input in enumerate(loss_ctx_input_list):
            mask = loss_ctx_input.shifted_labels != -100
            entropy = -(cast(torch.Tensor, loss_ctx_input.old_logprobs) * mask).sum()
            sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
            if loss_ctx_input.rollout_logprobs is not None:
                rollout_entropy = -(cast(torch.Tensor, loss_ctx_input.rollout_logprobs) * mask).sum()
                sum_rollout_entropy = (
                    rollout_entropy if sum_rollout_entropy is None else sum_rollout_entropy + rollout_entropy
                )

            if not mask.any():  # all padding tokens, skip
                self.logger.warning(f"Skip batch {i} as all tokens are padding.")
                continue

            if len(rollout_logprobs_list) > 0:
                # calculate importance sampling weights
                cu_seq_lens = seq_ctx_list[i].cu_seq_lens_q
                num_tokens = cu_seq_lens[1:] - cu_seq_lens[:-1]

                rollout_is_weights, rollout_is_mask, mismatch_metrics, rollout_is_metrics = (
                    loss_cfg.rollout_is.compute_rollout_importance_weights_and_metrics(
                        old_log_prob=loss_ctx_input.old_logprobs,
                        rollout_log_prob=rollout_logprobs_list[i],
                        num_tokens=num_tokens,
                        response_mask=mask,
                    )
                )
                loss_ctx_input.shifted_labels[~rollout_is_mask.bool()] = -100  # update loss mask
                loss_ctx_input.is_weights = rollout_is_weights
                all_rollout_is_metrics.append(rollout_is_metrics)
                all_mismatch_metrics.append(mismatch_metrics)

        logger_msg = f"Rollout {rollout_idx}: "

        if len(all_mismatch_metrics) > 0:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if len(mismatch_metrics) > 0:
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"

        if len(all_rollout_is_metrics) > 0:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if len(rollout_is_metrics) > 0:
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"
        self.logger.info(logger_msg)

        if self._has_ref:
            # ref logprobs are inplaced updated in compute_actor_logprobs
            loss_ctx_input_list = self.compute_ref_logprobs(seq_ctx_list, loss_ctx_input_list)
            kl_div_sum: torch.Tensor | None = None
            for loss_ctx_input in loss_ctx_input_list:
                mask = loss_ctx_input.shifted_labels != -100
                kl_div = kl_penalty(
                    cast(torch.Tensor, loss_ctx_input.old_logprobs),
                    cast(torch.Tensor, loss_ctx_input.ref_logprobs),
                    loss_weights=mask,
                    kl_penalty="low_var_kl",
                )
                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div

            kl_div_sum = cast(torch.Tensor, kl_div_sum)
            dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
            avg_kl_div = kl_div_sum / global_grad_tokens if global_grad_tokens > 0 else 0
            self.logger.info(f"Rollout {rollout_idx}: avg KL divergence: {avg_kl_div:.4f}")

        for i in range(0, len(seq_ctx_list), iters_per_step):
            batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
            batches_loss_ctx_input = loss_ctx_input_list[i : i + iters_per_step]

            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(batches_loss_ctx_input, loss_cfg)
            engine_input = []
            for seq_ctx, loss_kwargs in zip(batches_seq_ctx, batches_loss_kwargs):
                loss_ctx = LossContext(
                    loss_cfg=loss_cfg,
                    loss_kwargs=loss_kwargs,
                )
                engine_input.append(
                    ModelItem(
                        seq_ctx=seq_ctx,
                        loss_ctx=loss_ctx,
                    )
                )

            loss_log, other_log = self._engine.train_step(
                data_batches=engine_input,
            )
            other_log = self._update_other_log(other_log)  # type: ignore[arg-type]
            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)
            log_info = dict()  # type: ignore[var-annotated]
            log_info.update(loss_log)
            for k, v in other_log.items():
                if k == "extra_info":
                    for extra_k, extra_v in v.items():
                        log_info[extra_k] = extra_v.item() if isinstance(extra_v, torch.Tensor) else extra_v
                else:
                    log_info[k] = v.item() if isinstance(v, torch.Tensor) else v
            log_info["grad_norm"] = grad_norm.item()
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in log_info.items()
            )
            log_str = f"Rollout {rollout_idx} Step {i}: " + log_str
            self.logger.info(log_str)

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        self._engine.save_hf(hf_dir, save_dtype)

    @ray_method
    def get_data_replicate_size(self) -> int:
        """Get the data replicate size for the training worker."""
        # tp and pp will affect the data replicate size in engine
        # sp will affect the data replicate size in worker
        return self._engine.data_replicate_size * self.sp_mesh.size()

    @ray_method
    def get_model_cfg(self):
        model_cfg = self._engine.model_cfg
        return model_cfg

    @ray_method
    def offload_model(self):
        self._engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self.logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        self.logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    @ray_method
    def onload_model(self):
        self._engine.put_model_to_device(DEVICE)

    @ray_method
    def onload_optimizer(self):
        self._engine.put_optimizer_to_device(DEVICE)

    @ray_method
    def update_rollout_info(
        self,
        engine_rank_mesh_array: DeviceMeshRaw,
        server_url_dict: ServiceUrlMap,
        rollout_config: RolloutConfig,
        worker_server_urls_status: Dict[str, bool],
    ):
        """Update the rollout information for the training worker."""
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        if self.rollout_device_mesh is None:
            self.rollout_device_mesh = DeviceMesh(
                "cpu", mesh=engine_rank_mesh_array, mesh_dim_names=("engine_instance", "engine_parallel")
            )
        rollout_server_url = server_url_dict.get(self.rank, "")
        if worker_server_urls_status.get(rollout_server_url, "False") is False:
            self.logger.error(f"Rollout server url {rollout_server_url} is not available.")
            self.rollout_url = None
        else:
            self.rollout_url = rollout_server_url
        self.rollout_cfg_info["tp"] = tp
        self.rollout_cfg_info["ep"] = ep
        self.rollout_cfg_info["api_key"] = rollout_config.api_key
        if os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
            self.rollout_cfg_info["backend"] = "sglang"
        else:
            self.rollout_cfg_info["backend"] = (rollout_config.extra_rollout_config or dict()).get(
                "lmdeploy_backend", "pytorch"
            )

    @ray_method
    def update_weights(self):
        """Update the model weights."""
        if self.rollout_cfg_info.get("backend") == "turbomind":
            self._update_weights_by_layer()
        else:
            self._update_weights_hf_generator()

    def _update_weights_hf_generator(self):
        """Update the model weights."""
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        if isinstance(model.config, BaseComposeConfig):
            dtype = torch.bfloat16
        else:
            if (model.config.float8_cfg is not None) and (model.config.float8_cfg.enable_float8):
                dtype = torch.float8_e4m3fn
            else:
                dtype = torch.bfloat16

        bucket_size = int(self.config.update_weight_bucket_size_in_gb * 1024**3)
        same_gen = model._get_same_hf_param(
            model._group_param_by_load_spec(LoadEnum.SAME), dtype=dtype, device=DEVICE, bucket_size=bucket_size
        )
        fused_gen = model._get_fused_hf_param(
            model._group_param_by_load_spec(LoadEnum.FUSED),
            dtype=dtype,
            device=DEVICE,
            bucket_size=bucket_size,
            update_weights_for_rl=True,
        )
        shard_gen = model._get_shard_hf_param(
            model._group_param_by_load_spec(LoadEnum.SHARD), dtype=dtype, device=DEVICE, bucket_size=bucket_size
        )

        for name_list, fused_param_list in fused_gen:
            state_dict = {name: param.detach() for name, param in zip(name_list, fused_param_list)}
            if model.fsdp_config.ep_size > 1:
                # When ep_size > 1, generator generates part of the fused param on each ep rank in one ep_group.
                # We can all gather them to get full fused param but it would lead to a larger memory usage.
                # So we broadcast the part fused param from each ep rank in ep_group sequentially,
                # and update the part of the fused param sequentially to reduce memory usage.
                ep_mesh: DeviceMesh = model.ep_mesh
                ep_group = ep_mesh.get_group()
                global_rank = dist.get_rank()
                for src_global_rank in dist.get_process_group_ranks(ep_group):
                    broadcast_state_dict = dict()
                    for key, tensor in state_dict.items():
                        obj_to_broadcast = [key, tensor.to("meta")] if global_rank == src_global_rank else [None, None]
                        dist.broadcast_object_list(obj_to_broadcast, src=src_global_rank, group=ep_group)
                        real_key, meta_tensor = obj_to_broadcast
                        buffer = (
                            state_dict[real_key]
                            if global_rank == src_global_rank
                            else torch.empty_like(meta_tensor, device=DEVICE)
                        )
                        dist.broadcast(buffer, src=src_global_rank, group=ep_group)
                        broadcast_state_dict[real_key] = buffer
                    self.request_update_params(broadcast_state_dict, finished=False)
                    del broadcast_state_dict, buffer
            else:
                self.request_update_params(state_dict, finished=False)
            del state_dict, name_list, fused_param_list

        for name_list, param_list in chain(same_gen, shard_gen):
            state_dict = {name: param.detach() for name, param in zip(name_list, param_list)}
            self.request_update_params(state_dict, finished=False)
            del state_dict, name_list, param_list

        if self.rollout_cfg_info["backend"] == "pytorch":
            self.request_update_params({}, finished=True)

        dist.barrier()
        DEVICE_MODULE.empty_cache()
        return

    def _update_weights_by_layer(self):
        """Update the model weights."""
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        if isinstance(model.config, BaseComposeConfig):
            # TODO: support float8 for vision compose model
            dtype = torch.bfloat16
        else:
            if (model.config.float8_cfg is not None) and (model.config.float8_cfg.enable_float8):
                dtype = torch.float8_e4m3fn
            else:
                dtype = torch.bfloat16

        def get_params(tensor_list, name_list, save_dtype):
            _tensor_list, _spec_list = list(zip(*tensor_list))
            fsdp_unshard_tensor_list = model._fsdp_foreach_allgather(_tensor_list, _spec_list)
            if save_dtype == torch.float8_e4m3fn:
                fsdp_unshard_tensor_list, name_list = model._to_float8(
                    fsdp_unshard_tensor_list, name_list, _tensor_list, save_dtype
                )
            return fsdp_unshard_tensor_list, name_list

        saved_list = []
        is_qwen3vl = False
        if isinstance(model.config, BaseComposeConfig):
            language_model = model.language_model
            if isinstance(model, Qwen3VLForConditionalGeneration):
                is_qwen3vl = True
        else:
            language_model = model

        if is_qwen3vl:
            vision_hf_prefix = "model.visual."
            projector_hf_prefix = "model.visual."
        else:
            vision_hf_prefix = "model.vision_tower."
            projector_hf_prefix = "model.multi_modal_projector."

        for i, layer in tqdm.tqdm(language_model.layers.items(), desc="[gather weight]"):
            tensor_list = []
            name_list = []
            for sub_name, param in layer.state_dict().items():
                if isinstance(model.config, BaseComposeConfig):
                    saved_list.append(f"language_model.layers.{i}.{sub_name}")
                else:
                    saved_list.append(f"layers.{i}.{sub_name}")
                local_tensor = param._local_tensor if isinstance(param, DTensor) else param
                local_tensor = local_tensor.bfloat16()
                load_spec = language_model.load_spec_mapping.get(f"layers.{i}.{sub_name}")

                if isinstance(model.config, BaseComposeConfig):
                    name = f"model.language_model.layers.{i}.{sub_name}"
                else:
                    name = f"model.layers.{i}.{sub_name}"

                if ".experts." in name and ".mlp.experts." not in name:
                    name = name.replace(".experts.", ".mlp.experts.")
                if ".gate." in name and ".mlp.gate." not in name:
                    name = name.replace(".gate.", ".mlp.gate.")
                name_list.append(name)
                tensor_list.append((local_tensor, load_spec))
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            self.request_update_params(state_dict)

        for name, param in model.state_dict().items():
            if name in saved_list:
                continue
            local_tensor = param._local_tensor if isinstance(param, DTensor) else param
            local_tensor = local_tensor.bfloat16()
            load_spec = model.load_spec_mapping.get(name)

            if isinstance(model.config, BaseComposeConfig):
                if "vision_tower." in name:
                    name = name.replace("vision_tower.", vision_hf_prefix)
                elif "multi_modal_projector." in name:
                    name = name.replace("multi_modal_projector.", projector_hf_prefix)
                elif name == "language_model.norm.weight":
                    name = "model.language_model.norm.weight"
                elif name == "language_model.embed_tokens.weight":
                    name = "model.language_model.embed_tokens.weight"
                elif name == "language_model.lm_head.weight":
                    name = "lm_head.weight"
            else:
                if name == "norm.weight":
                    name = "model.norm.weight"
                elif name == "embed_tokens.weight":
                    name = "model.embed_tokens.weight"
            tensor_list = [(local_tensor, load_spec)]
            name_list = [name]
            fsdp_unshard_tensor_list, name_list = get_params(tensor_list, name_list, dtype)
            state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
            self.request_update_params(state_dict)

        if self.rollout_cfg_info["backend"] == "pytorch":
            self.request_update_params({}, finished=True)

        dist.barrier()
        DEVICE_MODULE.empty_cache()
        return

    # def update_weights1(self):
    #     """Update the model weights."""
    #     self.endpoints["update_weights"] = "update_weights"
    #     assert self.rollout_device_mesh is not None
    #     time1 = time.time()

    #     model = self._engine.model
    #     DEVICE_MODULE.empty_cache()

    #     if (model.config.float8_cfg is not None) and (model.config.float8_cfg.enable_float8):
    #         dtype = torch.float8_e4m3fn
    #     else:
    #         dtype = torch.bfloat16

    #     fused_params = []
    #     for name, param in model.state_dict().items():
    #         load_spec = model.load_spec_mapping.get(name)
    #         if load_spec.load_enum == LoadEnum.FUSED:
    #             fused_params.append((name, param, load_spec))

    #     # TODO: decouple update_weights from the model structure
    #     bucket_size = 1024**3
    #     safetensor_size = 0
    #     tensor_list: list[tuple[torch.Tensor, LoadSpec]] = []
    #     name_list: list[str] = []
    #     for name, param, load_spec in fused_params:
    #         local_tensor = param._local_tensor if isinstance(param, DTensor) else param
    #         local_tensor = local_tensor.bfloat16()
    #         if safetensor_size + model._get_tensor_size(param, dtype) > bucket_size:
    #             _tensor_list, _spec_list = list(zip(*tensor_list))
    #             fsdp_unshard_tensor_list = model._fsdp_foreach_allgather(_tensor_list, _spec_list)
    #             if dtype == torch.float8_e4m3fn:
    #                 fsdp_unshard_tensor_list, name_list = model._to_float8(
    #                     fsdp_unshard_tensor_list, name_list, _tensor_list, dtype
    #                 )
    #             state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
    #             self.request_update_params(state_dict)
    #             safetensor_size = 0
    #             tensor_list = [(local_tensor, load_spec)]
    #             name_list = ["model." + name.replace(".experts.", ".mlp.experts.")]
    #             continue
    #         safetensor_size += model._get_tensor_size(param, dtype)
    #         tensor_list.append((local_tensor, load_spec))
    #         name_list.append("model." + name.replace(".experts.", ".mlp.experts."))

    #     if tensor_list:
    #         assert len(name_list) == len(tensor_list)
    #         _tensor_list, _spec_list = list(zip(*tensor_list))
    #         fsdp_unshard_tensor_list = model._fsdp_foreach_allgather(_tensor_list, _spec_list)
    #         if dtype == torch.float8_e4m3fn:
    #             fsdp_unshard_tensor_list, name_list = model._to_float8(
    #                 fsdp_unshard_tensor_list, name_list, _tensor_list, dtype
    #             )
    #         state_dict = dict(zip(name_list, fsdp_unshard_tensor_list))
    #         self.request_update_params(state_dict)

    #     same_gen = model._get_same_hf_param(
    #         model._group_param_by_load_spec(LoadEnum.SAME),
    #         dtype=dtype,
    #         device="cuda",
    #         bucket_size=1024**3,
    #     )
    #     for name_list, gathered_tensor_list in tqdm.tqdm(same_gen, desc="[update dense weights]"):
    #         state_dict = dict(zip(name_list, gathered_tensor_list))
    #         self.request_update_params(state_dict)
    #         del state_dict

    #     self.request_update_params({}, finished=True)

    #     dist.barrier()
    #     logger.info(f"update weights time: {time.time() - time1}")
    #     DEVICE_MODULE.empty_cache()
    #     return

    # def update_weights(self):
    #     """Update the model weights."""
    #     self.endpoints["update_weights"] = "update_weights"
    #     assert self.rollout_device_mesh is not None

    #     model = self._engine.model
    #     DEVICE_MODULE.empty_cache()

    #     saved_keys = []
    #     gather_duration = []
    #     weight_duration = []
    #     reshard_duration = []

    #     # update decoder layers
    #     for i, layer in tqdm.tqdm(model.layers.items(), desc="[gather weight]"):
    #         start = time.perf_counter()
    #         layer.unshard()
    #         layer_state_dict = {}

    #         for sub_name, param in layer.named_parameters():
    #             if "_checkpoint_wrapped_module." in sub_name:
    #                 sub_name = sub_name.replace("_checkpoint_wrapped_module.", "")
    #             if isinstance(param, DTensor):
    #                 param = param.to_local()

    #             if isinstance(param, WeightWithDynamicTilewiseFloat8CastTensor):
    #                 param = param._tensor

    #             if isinstance(param, Float8Tensor):
    #                 scale_name = f"model.layers.{i}.{sub_name}_scale_inv"
    #                 assert "fused_w1w3" in sub_name or "fused_w2" in sub_name
    #                 # save scale_inv parameter to state_dict
    #                 scale_tensor = param._scale
    #                 quant_tensor = param._data
    #                 ep_mesh = model.ep_mesh
    #                 if ep_mesh.size() > 1:
    #                     scale_tensor = torch.cat(dist.nn.all_gather(scale_tensor, group=ep_mesh.get_group()), dim=0)
    #                     quant_tensor = torch.cat(dist.nn.all_gather(quant_tensor, group=ep_mesh.get_group()), dim=0)
    #                 layer_state_dict[scale_name] = scale_tensor.detach()
    #                 # set `param` which will be added to state_dict at the bottom of the for-block
    #                 param = quant_tensor

    #             param = param.to(DEVICE)
    #             name = f"model.layers.{i}.{sub_name}"
    #             saved_keys.append(name.replace("model.", ""))
    #             if ".experts." in name and ".mlp." not in name:
    #                 name = name.replace(".experts.", ".mlp.experts.")
    #             if ".gate." in name and ".mlp." not in name:
    #                 name = name.replace(".gate.", ".mlp.gate.")
    #             layer_state_dict[name] = param.detach()
    #         gather_duration.append(time.perf_counter() - start)
    #         start = time.perf_counter()
    #         self.request_update_params(layer_state_dict, finished=True)
    #         breakpoint()
    #         weight_duration.append(time.perf_counter() - start)

    #         start = time.perf_counter()
    #         del layer_state_dict
    #         layer.reshard()
    #         reshard_duration.append(time.perf_counter() - start)

    #     if dist.get_rank() == 0:
    #         logger.debug(
    #             f"Rank 0 Gather decoder layers done, total {sum(gather_duration):.2f}s, avg "
    #             f"{sum(gather_duration) / len(gather_duration):.2f}s"
    #         )
    #         logger.debug(
    #             f"Rank 0 migrate/save decoder layers done, total {sum(weight_duration):.2f}s, avg "
    #             f"{sum(weight_duration) / len(weight_duration):.2f}s"
    #         )
    #         logger.debug(
    #             f"Rank 0 reshard decoder layers done, total {sum(reshard_duration):.2f}s, avg "
    #             f"{sum(reshard_duration) / len(reshard_duration):.2f}s"
    #         )

    #     # update other params
    #     model.norm.unshard()
    #     model.lm_head.unshard()
    #     model.embed_tokens.unshard()
    #     others_state_dict = {}
    #     for name, param in model.named_parameters():
    #         if "_checkpoint_wrapped_module." in name:
    #             continue
    #         if name not in saved_keys:
    #             saved_keys.append(name)
    #             if name == "norm.weight":
    #                 name = "model.norm.weight"
    #             if name == "embed_tokens.weight":
    #                 name = "model.embed_tokens.weight"
    #             if isinstance(param, DTensor):
    #                 param = param.to_local()
    #             others_state_dict[name] = param.detach()
    #     self.request_update_params(others_state_dict, finished=True)
    #     model.norm.reshard()
    #     model.lm_head.reshard()
    #     model.embed_tokens.reshard()
    #     del others_state_dict
    #     del param

    #     dist.barrier()
    #     DEVICE_MODULE.empty_cache()
    #     return

    @ray_method
    def request_update_params(self, state_dict, finished=False):
        """Send a request to update the parameters on the rollout workers.

        This method serializes the state dictionary and sends it to the
        appropriate rollout worker via an HTTP request.

        Args:
            state_dict (dict | list): The state dictionary containing the model
                parameters to update.
            finished (bool): A flag indicating whether this is the final
                batch of updates. Defaults to False.
        """
        cpu_mesh = self.rollout_device_mesh["engine_parallel"]
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()
        if self.rollout_url is None:
            self.logger.error(f"rank {self.rank} url in None, cannot update weights and skip")
            return
        if self.rollout_cfg_info["backend"] == "pytorch":
            # TODO(chenchiyu): remove lmdeploy related code
            from lmdeploy.utils import serialize_state_dict

            try:
                from lmdeploy.utils import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            if self.rollout_cfg_info["backend"] == "pytorch" and self.rollout_cfg_info["tp"] > 1:
                serialized_data = [None] * self.rollout_cfg_info["tp"]
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(state_dict.items()))
                    metadata = flattened_tensor_bucket.get_metadata()
                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    tp_serialized_data = serialize_state_dict(flattened_tensor_data)
                else:
                    tp_serialized_data = serialize_state_dict(state_dict)
                dist.gather_object(
                    tp_serialized_data,
                    serialized_data if dist.get_rank() == head_rank else None,
                    dst=head_rank,
                    group=cpu_group,
                )
            elif self.rollout_cfg_info["backend"] == "pytorch":
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=list(state_dict.items()))
                    metadata = flattened_tensor_bucket.get_metadata()
                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    serialized_data = serialize_state_dict(flattened_tensor_data)
                else:
                    serialized_data = serialize_state_dict(state_dict)
            else:
                # for turbomind backend, only head_rank should serialize data
                serialized_data = serialize_state_dict(state_dict) if dist.get_rank() == head_rank else None
        else:
            # sglang
            from sglang.srt.utils import MultiprocessingSerializer
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

            try:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

                use_flattened_tensor_bucket = True
            except Exception:
                use_flattened_tensor_bucket = False

            # NOTE: xtuner目前去掉sglang的patch也不会出问题，但为了保险起见，还是保留patch逻辑，并且在update_weights结束后unpatch
            monkey_patch_torch_reductions()
            state_dict = state_dict.items()
            if self.rollout_cfg_info["tp"] == 1:
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_dict)
                    metadata = flattened_tensor_bucket.get_metadata()

                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                else:
                    serialized_data = MultiprocessingSerializer.serialize(state_dict, output_str=True)

                serialized_data = [serialized_data]
            else:
                serialized_data = [None] * self.rollout_cfg_info["tp"]
                if use_flattened_tensor_bucket:
                    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=state_dict)
                    metadata = flattened_tensor_bucket.get_metadata()

                    flattened_tensor_data = {
                        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                        "metadata": metadata,
                    }
                    tp_serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                    dist.gather_object(
                        tp_serialized_data,
                        serialized_data if dist.get_rank() == head_rank else None,
                        dst=head_rank,
                        group=cpu_group,
                    )
                else:
                    tp_serialized_data = MultiprocessingSerializer.serialize(state_dict, output_str=True)
                    dist.gather_object(
                        tp_serialized_data,
                        serialized_data if dist.get_rank() == head_rank else None,
                        dst=head_rank,
                        group=cpu_group,
                    )

        if dist.get_rank() == head_rank:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.rollout_cfg_info['api_key']}",
            }
            if self.rollout_cfg_info["backend"] == "sglang":
                payload = {
                    "serialized_named_tensors": serialized_data,
                    "flush_cache": False,
                }
                try:
                    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

                    use_flattened_tensor_bucket = True
                except Exception:
                    use_flattened_tensor_bucket = False
                if use_flattened_tensor_bucket:
                    payload["load_format"] = "flattened_bucket"

                url = f"{self.rollout_url}/update_weights_from_tensor"
                response = requests.post(url, json=payload or {})
                response.raise_for_status()
            else:
                data = dict(serialized_named_tensors=serialized_data, finished=finished)
                try:
                    from lmdeploy.utils import FlattenedTensorBucket

                    use_flattened_tensor_bucket = True
                except Exception:
                    use_flattened_tensor_bucket = False

                if use_flattened_tensor_bucket:
                    data["load_format"] = "flattened_bucket"
                response = requests.post(
                    f"{self.rollout_url}/{self.endpoints['update_weights']}", headers=headers, json=data
                )
            assert response.status_code == 200, f"response.status_code = {response.status_code}"

        if finished:
            dist.barrier(group=cpu_group)

        monkey_unpatch_torch_reductions()
        return

    @ray_method
    def save_dcp(self, checkpoint_path: Path | str):
        """Save the DCP checkpoint of the training worker."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        optimizer_path = checkpoint_path / self._SAVE_OPTIMIZER_DIR
        model_path = checkpoint_path / self._SAVE_MODEL_DIR

        # Save model and optimizer
        self._engine.save_dcp(
            model_dir=model_path,
            optimizer_dir=optimizer_path,
        )

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training worker from the checkpoint."""
        resume_from = load_checkpoint_cfg.checkpoint_path
        if resume_from is None:
            return
        if isinstance(resume_from, str):
            resume_from = Path(resume_from)
        self.logger.info(f"Resume from checkpoint: {resume_from}")

        if not resume_from.exists():
            raise FileNotFoundError(f"Checkpoint path {resume_from} does not exist.")

        model_path = resume_from / self._SAVE_MODEL_DIR
        optimizer_path = (
            resume_from / self._SAVE_OPTIMIZER_DIR
            if load_checkpoint_cfg.load_optimizer_states or load_checkpoint_cfg.load_optimizer_args
            else None
        )

        self._engine.load_dcp(
            model_dir=model_path,
            optimizer_dir=optimizer_path,
            load_states=load_checkpoint_cfg.load_optimizer_states,
            load_args=load_checkpoint_cfg.load_optimizer_args,
        )

    @ray_method
    def ready(self) -> bool:
        return True


TrainingWorkerClass = ActorClass[TrainingWorker]
TrainingWorkerProxy = ActorProxy[TrainingWorker]
