import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Sequence, TypedDict, cast


if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

import numpy as np
import ray
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass, ActorProxy
from torch.distributed.device_mesh import init_device_mesh
from typing_extensions import NotRequired

from transformers import AutoTokenizer
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.engine.train_engine import LossLog, OtherLog, TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import (
    VisionComposeTrainEngine,
)
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import CELossConfig
from xtuner.v1.loss.ce_loss import CELossContext
from xtuner.v1.model.base import BaseModel as XtunerBaseModel
from xtuner.v1.model.base import ModelItem, TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.rl.loss import BaseRLLossConfig, BaseRLLossContext, kl_penalty
from xtuner.v1.rl.utils import SingleAcceleratorWorker, gather_logprobs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import (
    XTUNER_DETERMINISTIC,
    ParallelConfigException,
    get_device,
    get_logger,
    get_torch_device_module,
    ray_method,
)

from ..rollout_is import merge_rollout_is_metrics
from .update_weighter import UpdateWeighter


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


def calculate_entropy(
    shifted_labels_list: Sequence[torch.Tensor],
    old_logprobs_list: Sequence[torch.Tensor | None],
    global_grad_tokens: torch.Tensor,
) -> torch.Tensor | None:
    if len(old_logprobs_list) == 0 or old_logprobs_list[0] is None:
        return None
    sum_entropy: torch.Tensor | None = None
    for i, shifted_labels in enumerate(shifted_labels_list):
        mask = shifted_labels != -100
        assert old_logprobs_list[i] is not None
        entropy = -(cast(torch.Tensor, old_logprobs_list[i]) * mask).sum()
        sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
    sum_entropy = cast(torch.Tensor, sum_entropy)
    dist.all_reduce(sum_entropy, op=dist.ReduceOp.SUM)
    avg_sum_entropy = sum_entropy / global_grad_tokens if global_grad_tokens > 0 else torch.tensor(0.0)
    return avg_sum_entropy


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
    seed: None | int = None  # if None, use RLTrainer seed

    # sft config
    sft_dataloader_cfg: DataloaderConfig | None = None
    sft_global_batch_size: int = -1
    rollout_steps_per_sft: int = 1
    sft_loss_cfg: CELossConfig = CELossConfig()

    def build(self, placement_group: "PlacementGroup"):
        """Build training workers and controller from this config and placement
        group."""
        # import here to avoid circular import
        from xtuner.v1.rl.trainer.controller import TrainingController
        from xtuner.v1.rl.utils import AutoAcceleratorWorkers

        TrainingWorkerCls = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                }
            }
        )(TrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorkerCls, self, placement_group)
        ray.wait([w.ready.remote() for w in train_workers])
        return TrainingController(workers=train_workers)


class WorkerInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.LongTensor
    advantages: torch.Tensor
    rollout_logprobs: torch.Tensor | None


class RLOtherLog(TypedDict):
    maxvio: NotRequired[float]
    step_consumed_tokens: int
    step_consumed_img_tokens: NotRequired[float]
    efficient_attn_ratio: float
    max_ratio: NotRequired[float]
    loss: NotRequired[float]
    grad_norm: NotRequired[float]


class WorkerTrainLogItem(TypedDict):
    loss_log: LossLog
    rl_other_log: RLOtherLog


class WorkerLogItem(TypedDict):
    train_entropy: float
    rollout_entropy: NotRequired[float]
    mismatch_metrics: NotRequired[dict[str, float]]
    rollout_is_metrics: NotRequired[dict[str, float]]
    train_metrics: List[WorkerTrainLogItem]
    sft_train_metrics: NotRequired[dict[str, float]]


class TrainingWorker(SingleAcceleratorWorker, UpdateWeighter):
    _SAVE_OPTIMIZER_DIR = "optimizer"
    _SAVE_MODEL_DIR = "model"
    _SAVE_SFT_DATALOADER_DIR = "sft_dataloader"
    _SAVE_SFT_TRAIN_STATE_PATH = "sft_train_state.json"

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

        self._set_deterministic()
        self._set_random_seed(worker_cfg.seed)

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]

        self._init_sft(worker_cfg)

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

        self._optimizer_steps = worker_cfg.optimizer_steps

        self._init_update_weighter()

    def _init_sft(self, worker_cfg: WorkerConfig):
        self._sft_dataloader_config = worker_cfg.sft_dataloader_cfg
        self._sft_dataloader: Dataloader | None = None
        self._sft_dataloader_iter: Iterable | None = None
        self._sft_loss_cfg: CELossConfig | None = None
        self._rollout_steps_per_sft = worker_cfg.rollout_steps_per_sft

        self._rollout_step = 0
        self._sft_cur_epoch = 0
        self._sft_total_consumed_samples = 0
        self._sft_total_consumed_tokens = 0

        if self._sft_dataloader_config is not None:
            assert worker_cfg.sft_global_batch_size > 0, "sft_global_batch_size must be greater than 0"
            assert worker_cfg.seed is not None, "seed must be set when sft_dataloader_config is not None"
            tokenizer = AutoTokenizer.from_pretrained(worker_cfg.load_from, trust_remote_code=True)
            self._sft_dataloader = self._sft_dataloader_config.build(
                tokenizer=tokenizer,
                dp_mesh=self.data_mesh["dp"],
                global_batch_size=worker_cfg.sft_global_batch_size,
                micro_batch_size=1,
                seed=worker_cfg.seed,
            )
            self.logger.info(f"Sft Dataloader len: {len(self._sft_dataloader)}")

            sft_loss_cfg = worker_cfg.sft_loss_cfg
            if worker_cfg.sft_loss_cfg is None:
                sft_loss_cfg = CELossConfig()
            self._sft_loss_cfg = sft_loss_cfg

    def _set_deterministic(self):
        if XTUNER_DETERMINISTIC:
            self.logger.info("Setting deterministic algorithms of TrainingWorker.")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: None | int):
        set_random_seed(seed)

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
            model = model.fully_shard(ref_model_fsdp_cfg)  # type: ignore

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
        self,
        seq_ctx_list: list[SequenceContext],
        shifted_labels_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # precompute float8 dynamic scale only once
        self._engine.maybe_precompute_float8_dynamic_scale_for_fsdp()
        old_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            output = self._engine.forward_only(seq_ctx=seq_ctx)
            old_logprobs = gather_logprobs(output["logits"], shifted_labels)
            old_logprobs_list.append(old_logprobs)
        return old_logprobs_list

    def compute_ref_logprobs(
        self, seq_ctx_list: list[SequenceContext], shifted_labels_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert self._has_ref
        self._ref_model.to_device(DEVICE)
        ref_logprobs_list: list[torch.Tensor] = []
        for seq_ctx, shifted_labels in zip(seq_ctx_list, shifted_labels_list):
            with torch.no_grad():
                ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx=None)
            ref_logprobs = gather_logprobs(ref_output["logits"], shifted_labels)
            ref_logprobs_list.append(ref_logprobs)
        self._ref_model.to_device("cpu")
        return ref_logprobs_list

    def _get_rl_other_log(self, other_log: OtherLog) -> RLOtherLog:
        from xtuner.v1.model.utils import ModelForwardExtraLogInfo

        extra_info: ModelForwardExtraLogInfo | dict = other_log.get("extra_info", {})
        if isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info_dict = extra_info.get()
        else:
            extra_info_updated = ModelForwardExtraLogInfo(extra_info)
            extra_info_dict = extra_info_updated.get()

        for k, v in extra_info_dict.items():
            if isinstance(v, torch.Tensor):
                extra_info_dict[k] = v.item()

        rl_other_log: RLOtherLog = {
            "maxvio": other_log.get("maxvio", 0.0),
            "step_consumed_tokens": other_log["step_consumed_tokens"],
            "step_consumed_img_tokens": float(other_log.get("step_consumed_img_tokens", 0.0)),
            "efficient_attn_ratio": other_log["efficient_attn_ratio"],
            "max_ratio": extra_info_dict.get("max_ratio", 0.0),
            "loss": extra_info_dict.get("loss", 0.0),
        }
        return rl_other_log

    def _add_rollout_routed_experts(
        self, seq_ctx: SequenceContext, rollout_routed_experts: torch.Tensor | list[torch.Tensor | ray.ObjectRef]
    ):
        language_cfg = (
            self.config.model_cfg.text_config
            if isinstance(self.config.model_cfg, BaseComposeConfig)
            else self.config.model_cfg
        )

        to_free_routed_expert_refs: list[ray.ObjectRef] = []
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
                    if self.sp_mesh is None or self.sp_mesh.size() == 1:
                        ray._private.internal_api.free(rollout_routed_expert_refs)
                    else:
                        if self.sp_mesh.get_local_rank() == 0:
                            # only free once of sp mesh
                            to_free_routed_expert_refs.append(rollout_routed_expert_refs)
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

        if self.sp_mesh is not None and self.sp_mesh.size() > 1:
            dist.barrier()
            for free_routed_expert_refs in to_free_routed_expert_refs:
                ray._private.internal_api.free(free_routed_expert_refs)
            del to_free_routed_expert_refs

    @ray_method
    def fit(self, data_batches: list[WorkerInputItem], rollout_idx: int) -> WorkerLogItem:
        # NOTE: sglang会清除logger handle, 重新创建
        self.logger = get_logger(log_dir=self.log_dir, tag="TrainingWorker")
        loss_cfg: BaseRLLossConfig = self.config.loss_cfg
        num_batches = len(data_batches)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps)
        if num_batches < self._optimizer_steps:
            self.logger.info(
                f"Optimizer only step once because num_batches {num_batches} < optimizer_steps {self._optimizer_steps}."
            )

        # Update seq_ctx: pixel_values, rollout_routed_experts
        # Init loss_ctx: shifted_labels, advantages, rollout_logprobs
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_list: list[BaseRLLossContext] = []
        for data in data_batches:
            # update seq_ctx
            seq_ctx = data["seq_ctx"]
            pixel_values = seq_ctx.pixel_values
            if pixel_values is not None:
                if not isinstance(pixel_values, np.ndarray):
                    assert isinstance(pixel_values, list), (
                        f"pixel_values should be list of tensor, got {type(pixel_values)}"
                    )
                    pixel_values = [ray.get(pixel_obf) for pixel_obf in pixel_values]
                    pixel_values = [torch.as_tensor(pixel_value) for pixel_value in pixel_values]
                    pixel_values = torch.cat(pixel_values, dim=0)
                    seq_ctx.pixel_values = pixel_values
                else:
                    raise NotImplementedError("The case where pixel_values is a numpy array is not implemented yet.")

            rollout_routed_experts = seq_ctx.rollout_routed_experts
            if rollout_routed_experts is not None:
                self._add_rollout_routed_experts(seq_ctx, rollout_routed_experts)

            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(self.sp_mesh)

            # init loss_ctx
            shifted_labels = data["shifted_labels"].to(DEVICE)
            advantages = data["advantages"].to(DEVICE)
            rollout_logprobs = data.get("rollout_logprobs", None)
            rollout_logprobs = rollout_logprobs.to(DEVICE) if rollout_logprobs is not None else None
            loss_ctx = loss_cfg.build(
                self.sp_mesh, shifted_labels=shifted_labels, advantages=advantages, rollout_logprobs=rollout_logprobs
            )

            seq_ctx_list.append(seq_ctx)
            loss_ctx_list.append(loss_ctx)

        del data_batches

        # When sp_mesh.size() > 1, get the sp_split shifted_labels and rollout_logprobs
        shifted_labels_list = [loss_ctx.loss_kwargs.shifted_labels for loss_ctx in loss_ctx_list]
        rollout_logprobs_list = [loss_ctx.loss_kwargs.rollout_logprobs for loss_ctx in loss_ctx_list]

        # compute old logprobs
        old_logprobs_list = self.compute_actor_logprobs(seq_ctx_list, shifted_labels_list)
        for old_logprobs, loss_ctx in zip(old_logprobs_list, loss_ctx_list):
            loss_ctx.loss_kwargs.old_logprobs = old_logprobs

        worker_log_item: WorkerLogItem = {"train_entropy": 0.0, "train_metrics": [], "sft_train_metrics": {}}
        logger_msg = f"Rollout {rollout_idx}: "

        # compute entropy
        rank_grad_tokens: torch.Tensor | None = None
        for shifted_labels in shifted_labels_list:
            mask = shifted_labels != -100
            grad_tokens = mask.sum()
            rank_grad_tokens = grad_tokens if rank_grad_tokens is None else rank_grad_tokens + grad_tokens
        rank_grad_tokens = cast(torch.Tensor, rank_grad_tokens)
        global_grad_tokens = rank_grad_tokens
        dist.all_reduce(global_grad_tokens, op=dist.ReduceOp.SUM)

        avg_sum_entropy = calculate_entropy(shifted_labels_list, old_logprobs_list, global_grad_tokens)
        avg_rollout_entropy = calculate_entropy(shifted_labels_list, rollout_logprobs_list, global_grad_tokens)

        assert avg_sum_entropy is not None
        worker_log_item["train_entropy"] = avg_sum_entropy.item()
        logger_msg += f"avg entropy: {avg_sum_entropy:.4f}"
        if avg_rollout_entropy is not None:
            worker_log_item["rollout_entropy"] = avg_rollout_entropy.item()
            logger_msg += f", avg rollout entropy: {avg_rollout_entropy:.4f}"

        # compute rollout importance sampling metrics
        all_rollout_is_metrics = []
        all_mismatch_metrics = []
        for i, loss_ctx in enumerate(loss_ctx_list):
            if loss_ctx.loss_kwargs.rollout_logprobs is not None:
                # calculate importance sampling weights
                num_tokens = seq_ctx_list[i].seq_lens_q
                mismatch_metrics, rollout_is_metrics = loss_ctx.compute_rollout_is(self.sp_mesh, num_tokens)
                all_rollout_is_metrics.append(rollout_is_metrics)
                all_mismatch_metrics.append(mismatch_metrics)

        if len(all_mismatch_metrics) > 0:
            mismatch_metrics = merge_rollout_is_metrics(all_mismatch_metrics, DEVICE)
            if len(mismatch_metrics) > 0:
                worker_log_item["mismatch_metrics"] = mismatch_metrics
                logger_msg += f"\n rollout mismatch metrics:\n{json.dumps(mismatch_metrics, indent=4)}"

        if len(all_rollout_is_metrics) > 0:
            rollout_is_metrics = merge_rollout_is_metrics(all_rollout_is_metrics, DEVICE)
            if len(rollout_is_metrics) > 0:
                worker_log_item["rollout_is_metrics"] = rollout_is_metrics
                logger_msg += f"\n rollout importance sampling metrics:\n{json.dumps(rollout_is_metrics, indent=4)}"

        if self.rank == 0:
            self.logger.info(logger_msg)

        # compute reference logprobs
        ref_logprobs_list: list[torch.Tensor] | None = None
        if self._has_ref:
            ref_logprobs_list = self.compute_ref_logprobs(seq_ctx_list, shifted_labels_list)

            for i, loss_ctx in enumerate(loss_ctx_list):
                loss_ctx.loss_kwargs.ref_logprobs = ref_logprobs_list[i]

            kl_div_sum: torch.Tensor | None = None
            for i, shifted_labels in enumerate(shifted_labels_list):
                mask = shifted_labels != -100
                kl_div = kl_penalty(
                    cast(torch.Tensor, old_logprobs_list[i]),
                    cast(torch.Tensor, ref_logprobs_list[i]),
                    loss_weights=mask,
                    kl_penalty="low_var_kl",
                )
                kl_div_sum = kl_div if kl_div_sum is None else kl_div_sum + kl_div

            kl_div_sum = cast(torch.Tensor, kl_div_sum)
            dist.all_reduce(kl_div_sum, op=dist.ReduceOp.SUM)
            avg_kl_div = kl_div_sum / global_grad_tokens if global_grad_tokens > 0 else 0
            self.logger.info(f"Rollout {rollout_idx}: avg KL divergence: {avg_kl_div:.4f}")

        # compute batched loss context
        batched_loss_ctx_list: list[BaseRLLossContext] = []
        LossContext = loss_cfg.loss_ctx_cls
        for i in range(0, len(loss_ctx_list), iters_per_step):
            batches_loss_ctx = loss_ctx_list[i : i + iters_per_step]
            batches_loss_ctx = LossContext.build_batches(batches_loss_ctx)
            batched_loss_ctx_list.extend(batches_loss_ctx)

        # train optimizer steps
        for i in range(0, len(seq_ctx_list), iters_per_step):
            batches_seq_ctx = seq_ctx_list[i : i + iters_per_step]
            batches_loss_ctx = batched_loss_ctx_list[i : i + iters_per_step]

            engine_input = [
                ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
                for seq_ctx, loss_ctx in zip(batches_seq_ctx, batches_loss_ctx)
            ]

            loss_log, other_log = self._engine.train_step(
                data_batches=engine_input,
            )
            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)
            rl_other_log = self._get_rl_other_log(other_log)  # type: ignore[arg-type]
            rl_other_log["grad_norm"] = grad_norm.item()
            worker_log_item["train_metrics"].append(WorkerTrainLogItem(loss_log=loss_log, rl_other_log=rl_other_log))

            log_info = {**loss_log, **rl_other_log}
            log_str = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in log_info.items()
            )
            log_str = f"Rank{self.rank} Rollout {rollout_idx} Step {i}: " + log_str
            self.logger.info(log_str)

        self._rollout_step += 1
        if self._sft_dataloader is not None and self._rollout_step % self._rollout_steps_per_sft == 0:
            loss_log = self._fit_sft()
            worker_log_item["sft_train_metrics"] = loss_log

        return worker_log_item

    def _fit_sft(self):
        self.logger.info(f"Train SFT after {self._rollout_step} RL steps")
        if self._sft_dataloader_iter is None:
            self._sft_dataloader_iter = iter(self._sft_dataloader)

        time_before_get_data = time.time()
        data_batch = self._next_sft_data_batch()
        time_before_train_step = time.time()
        data_time = time_before_train_step - time_before_get_data
        DEVICE_MODULE.reset_peak_memory_stats()
        cur_sample_num = len(data_batch)

        loss_log, other_log, grad_norm = self._train_one_step_sft(data_batch)

        time_after_train_step = time.time()
        step_time = time_after_train_step - time_before_train_step
        step_consumed_tokens = other_log["step_consumed_tokens"]

        self._sft_total_consumed_samples += self._reduce_number_across_rank(cur_sample_num)
        reduced_step_consumed_tokens = self._reduce_number_across_rank(step_consumed_tokens)
        self._sft_total_consumed_tokens += reduced_step_consumed_tokens

        self._sft_log_step(
            loss_log=loss_log,
            local_step_consumed_tokens=step_consumed_tokens,
            step_consumed_tokens=reduced_step_consumed_tokens,
            total_consumed_tokens=self._sft_total_consumed_tokens,
            data_time=data_time,
            step_time=step_time,
            grad_norm=grad_norm,
            efficient_attn_ratio=other_log["efficient_attn_ratio"],
        )

        # to return sft log
        loss_log["grad_norm"] = grad_norm.item()
        loss_log["data_time"] = data_time
        loss_log["step_time"] = step_time
        loss_log["tgs"] = step_consumed_tokens / step_time
        loss_log["efficient_attn_ratio"] = other_log["efficient_attn_ratio"]
        return loss_log

    def _next_sft_data_batch(self):
        try:
            data = next(self._sft_dataloader_iter)  # type: ignore[assignment]
        except StopIteration:
            self._sft_cur_epoch += 1
            self._sft_dataloader.set_epoch(self._sft_cur_epoch)
            self._sft_dataloader_iter = iter(self._sft_dataloader)
            data = next(self._sft_dataloader_iter)
        return data

    def _train_one_step_sft(self, data_batch):
        seq_ctx_list: list[SequenceContext] = []
        loss_cfg: CELossConfig = self._sft_loss_cfg
        loss_ctx_list: list[CELossContext] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"].to(DEVICE)
            if self.sp_mesh.size() > 1:
                seq_ctx = seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
            seq_ctx_list.append(seq_ctx)
            loss_ctx = loss_cfg.build(shifted_labels=data["shifted_labels"], sp_mesh=self.sp_mesh)
            loss_ctx_list.append(loss_ctx)

        del data_batch

        cu_seq_lens_list = [seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list]
        loss_ctx_list = CELossContext.build_batches(
            loss_ctx_list, cu_seq_lens_list=cu_seq_lens_list, sp_mesh=self.sp_mesh
        )

        engine_input = [
            ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx) for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]

        loss_log, other_log = self._engine.train_step(engine_input)
        grad_norm = self._engine.clip_grad_norm()
        self._engine.step_optimizer(grad_norm)
        return loss_log, other_log, grad_norm

    def _sft_log_step(
        self,
        loss_log: LossLog,
        local_step_consumed_tokens: int,
        step_consumed_tokens: int,
        total_consumed_tokens: int,
        data_time: float,
        step_time: float,
        grad_norm: float,
        efficient_attn_ratio: float,
    ):
        tgs = local_step_consumed_tokens / step_time
        loss_log_list = [f"{k}: {v:.8f}" for k, v in loss_log.items()]
        loss_log_str = ", ".join(loss_log_list)

        max_memory = DEVICE_MODULE.max_memory_allocated()  # type: ignore[attr-defined]
        reserved_memory = DEVICE_MODULE.max_memory_reserved()  # type: ignore[attr-defined]

        self.logger.info(
            f"Rank{self.rank} Step {self._rollout_step}: data_time: {data_time:.4f} time: {step_time:.4f} "
            f"text_tokens: {local_step_consumed_tokens} "
            f"step_consumed_tokens: {step_consumed_tokens} "
            f"total_consumed_tokens: {total_consumed_tokens} "
            f"efficient_attn_ratio: {efficient_attn_ratio:.4f} "
            f"{loss_log_str} "
            f"grad_norm: {grad_norm:.8f} "
            f"max_memory: {max_memory / (1024**3):.2f} GB "
            f"reserved_memory: {reserved_memory / (1024**3):.2f} GB "
            f"tgs: {tgs:.1f} "
        )

    def _reduce_number_across_rank(self, rank_number: int) -> int:
        _gathered_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(_gathered_list, rank_number)
        reduced_number = sum(_gathered_list)  # type: ignore[arg-type]
        return reduced_number

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
    def save(self, checkpoint_path: Path | str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training worker."""
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        optimizer_path = checkpoint_path / self._SAVE_OPTIMIZER_DIR
        model_path = checkpoint_path / self._SAVE_MODEL_DIR

        # Save model and optimizer
        self._engine.save_dcp(
            model_dir=model_path,
            optimizer_dir=None if no_save_optimizer else optimizer_path,
        )

        # Save sft dataloader
        if self.rank == 0 and self._sft_dataloader is not None:
            sft_dataloader_path = checkpoint_path / self._SAVE_SFT_DATALOADER_DIR
            dataloader_state = self._sft_dataloader.get_state_dict(self._sft_total_consumed_samples)
            torch.save(dataloader_state, sft_dataloader_path)

            train_state_path = checkpoint_path / self._SAVE_SFT_TRAIN_STATE_PATH
            with train_state_path.open("w") as f:
                f.write(
                    json.dumps(
                        {
                            "cur_step": self._rollout_step,
                            "cur_epoch": self._sft_cur_epoch,
                            "total_consumed_samples": self._sft_total_consumed_samples,
                            "total_consumed_tokens": self._sft_total_consumed_tokens,
                        }
                    )
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

        # Resume sft dataloader
        sft_dataloader_path = resume_from / self._SAVE_SFT_DATALOADER_DIR
        if self._sft_dataloader is not None:
            if not sft_dataloader_path.exists():
                raise FileNotFoundError(f"Dataloader path {sft_dataloader_path} does not exist.")
            dataloader_state = torch.load(sft_dataloader_path, map_location=DEVICE)
            self._sft_dataloader.load_state_dict(dataloader_state)
            self.logger.info(f"Resume sft dataloader from {sft_dataloader_path}")

            train_state_path = resume_from / self._SAVE_SFT_TRAIN_STATE_PATH
            if not train_state_path.exists():
                raise FileNotFoundError(f"Train state path {train_state_path} does not exist.")
            with train_state_path.open("r") as f:
                train_state = json.loads(f.read())
                self._rollout_step = train_state["cur_step"]
                self._sft_cur_epoch = train_state["cur_epoch"]
                self._sft_total_consumed_samples = train_state["total_consumed_samples"]
                self._sft_total_consumed_tokens = train_state["total_consumed_tokens"]
                self.logger.info(f"Resume sft train state from {train_state_path}")

    @ray_method
    def ready(self) -> bool:
        return True


TrainingWorkerClass = ActorClass[TrainingWorker]
TrainingWorkerProxy = ActorProxy[TrainingWorker]
