import math
import os
import time
from typing import Dict, List, TypeAlias, TypedDict, cast

import ray
import requests
import torch
import torch.distributed as dist
import tqdm
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from xtuner.utils.device import get_device, get_torch_device
from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.float8_tensor import Float8Tensor
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTilewiseFloat8CastTensor
from xtuner.v1.ray.accelerator import SingleAcceleratorWorker
from xtuner.v1.ray.config import RolloutConfig
from xtuner.v1.rl.loss_context import LossContext
from xtuner.v1.rl.utils import gather_logprobs, sp_split
from xtuner.v1.utils import ParallelConfigException, get_logger

from ..loss_context import LossContextInputItem
from .config import WorkerConfig
from .engine import GRPOTrainEngine


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
DEVICE = get_device()
DEVICE_MODULE = get_torch_device()
logger = get_logger()


class WorkerInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.LongTensor
    advantages: torch.Tensor


class TrainingWorker(SingleAcceleratorWorker):
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
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        self._engine = self._build_engine(worker_cfg)

        self._has_ref = False
        if worker_cfg.loss_cfg.use_kl_loss:
            self._has_ref = True
            self._ref_model = self._build_ref_model(worker_cfg.model_cfg, worker_cfg.ref_model_fsdp_cfg)

        self.data_mesh = self._init_data_mesh(sp_size=worker_cfg.sp_size)
        self.sp_mesh = self.data_mesh["sp"]
        self._optimizer_steps = worker_cfg.optimizer_steps

        self.loss_ctx = LossContext(loss_cfg=worker_cfg.loss_cfg)

        # Used to update weight to rollout engine
        self.rank = rank
        self.rollout_device_mesh: DeviceMesh | None = None
        self.rollout_url: str | None = None
        self.rollout_cfg_info: dict = dict()
        self.endpoints: dict[str, str] = dict()
        self.endpoints["update_weights"] = "update_weights"
        # TODO: add lr scheduler

    def _build_engine(self, worker_cfg: WorkerConfig):
        pass

    def _build_ref_model(self, ref_model_cfg: TransformerConfig, ref_model_fsdp_cfg: FSDPConfig | None = None):
        with torch.device("meta"):
            model = ref_model_cfg.build()
        if ref_model_cfg.float8_cfg is not None and ref_model_cfg.float8_cfg.enable_float8:
            float8_handler = Float8Handler(
                scaling_granularity_gemm=ref_model_cfg.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=ref_model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
            )
        else:
            float8_handler = None
        if ref_model_fsdp_cfg is None:
            ref_model_fsdp_cfg = FSDPConfig(recompute_ratio=0, cpu_offload=False, requires_grad=False)
        model = model.fully_shard(ref_model_fsdp_cfg, float8_handler)
        model.eval()
        if float8_handler is not None:
            # As the ref model is not updated, we only compute params' scales once
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)
        model.to_device("cpu")
        DEVICE_MODULE.empty_cache()  # type: ignore
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

    def _put_data_to_device(
        self,
        data: WorkerInputItem,
        device: torch.device | str,
    ) -> WorkerInputItem:
        for key, value in data.items():
            if isinstance(value, (torch.Tensor, SequenceContext)):
                data[key] = value.to(device)  # type: ignore
        return data

    def fit(self, data_batches: list[WorkerInputItem]):
        num_batches = len(data_batches)
        iters_per_step = math.ceil(num_batches / self._optimizer_steps)
        if num_batches < self._optimizer_steps:
            logger.info(
                f"Optimizer only step once because num_batches {num_batches} < optimizer_steps {self._optimizer_steps}."
            )

        for data in data_batches:
            self._put_data_to_device(data, DEVICE)
            if self.sp_mesh.size() > 1:
                data["seq_ctx"] = data["seq_ctx"].split(self.sp_mesh)

        if self._has_ref:
            self._ref_model.to_device(DEVICE)
            for data in data_batches:
                seq_ctx = data["seq_ctx"]
                with torch.no_grad():
                    if isinstance(self.config.model_cfg, MoEConfig):
                        ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx=None)
                    else:
                        ref_output = self._ref_model(seq_ctx=seq_ctx, loss_ctx=None)
                ref_logprobs = gather_logprobs(ref_output["logits"], data["shifted_labels"])
                cast(LossContextInputItem, data)["ref_logprobs"] = ref_logprobs
            self._ref_model.to_device("cpu")

        for data in data_batches:
            seq_ctx = data["seq_ctx"]
            shifted_labels = data["shifted_labels"]
            if self.sp_mesh.size() > 1:
                shifted_labels = sp_split(shifted_labels, sp_mesh=self.sp_mesh, split_dim=1, padding_value=-100)
            output = self._engine.forward_only(seq_ctx=seq_ctx)
            old_logprobs = gather_logprobs(output["logits"], shifted_labels)
            cast(LossContextInputItem, data)["old_logprobs"] = old_logprobs

        for i in range(0, len(data_batches), iters_per_step):
            data_batch = data_batches[i : i + iters_per_step]
            engine_input = self.loss_ctx.build_list_ctx(
                data_batch=cast(list[LossContextInputItem], data_batch),
            )
            loss_log, other_log = self._engine.train_step(
                data_batches=engine_input,
            )
            grad_norm = self._engine.clip_grad_norm()
            self._engine.step_optimizer(grad_norm)
            logger.info(f"grad_norm: {grad_norm.item()}")

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        self._engine.save_hf(hf_dir, save_dtype)

    def get_data_replicate_size(self) -> int:
        """Get the data replicate size for the training worker."""
        # tp and pp will affect the data replicate size in engine
        # sp will affect the data replicate size in worker
        return self._engine.data_replicate_size * self.sp_mesh.size()

    def offload_model(self):
        self._engine.put_model_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        logger.info(
            f"Offloaded model to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    def offload_optimizer(self):
        """Offload the optimizer of the training worker."""
        self._engine.put_optimizer_to_device("cpu")
        DEVICE_MODULE.empty_cache()
        logger.info(
            f"Offloaded optimizer to CPU. Current allocate {DEVICE_MODULE.memory_allocated() / (1024**2)} MB, "
            f"reserved: {DEVICE_MODULE.memory_reserved() / (1024**2)} MB"
        )

    def onload_model(self):
        self._engine.put_model_to_device(DEVICE)

    def onload_optimizer(self):
        self._engine.put_optimizer_to_device(DEVICE)

    def update_rollout_info(
        self, engine_mesh_list: DeviceMeshRaw, server_url_dict: ServiceUrlMap, rollout_config: RolloutConfig
    ):
        """Update the rollout information for the training worker."""
        tp = rollout_config.tensor_parallel_size
        ep = rollout_config.expert_parallel_size
        assert tp == 1 or ep == 1, "Either tensor parallel size or engine parallel size must be 1."
        self.rollout_device_mesh = DeviceMesh(
            "cpu", mesh=engine_mesh_list, mesh_dim_names=("engine_instance", "engine_parallel")
        )
        self.rollout_url = server_url_dict.get(self.rank, "")
        self.rollout_cfg_info["tp"] = tp
        self.rollout_cfg_info["ep"] = ep
        self.rollout_cfg_info["api_key"] = rollout_config.api_key
        self.rollout_cfg_info["backend"] = (rollout_config.extra_rollout_config or dict()).get(
            "lmdeploy_backend", "pytorch"
        )

    def update_weights(self):
        """Update the model weights."""
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self._engine.model
        DEVICE_MODULE.empty_cache()

        saved_keys = []
        gather_duration = []
        weight_duration = []
        reshard_duration = []

        # update decoder layers
        for i, layer in tqdm.tqdm(model.layers.items(), desc="[gather weight]"):
            start = time.perf_counter()
            layer.unshard()
            layer_state_dict = {}

            for sub_name, param in layer.named_parameters():
                if "_checkpoint_wrapped_module." in sub_name:
                    sub_name = sub_name.replace("_checkpoint_wrapped_module.", "")
                if isinstance(param, DTensor):
                    param = param.to_local()

                if isinstance(param, WeightWithDynamicTilewiseFloat8CastTensor):
                    param = param._tensor

                if isinstance(param, Float8Tensor):
                    scale_name = f"model.layers.{i}.{sub_name}_scale_inv"
                    assert "fused_w1w3" in sub_name or "fused_w2" in sub_name
                    # save scale_inv parameter to state_dict
                    scale_tensor = param._scale
                    quant_tensor = param._data
                    ep_mesh = model.ep_mesh
                    if ep_mesh.size() > 1:
                        scale_tensor = torch.cat(dist.nn.all_gather(scale_tensor, group=ep_mesh.get_group()), dim=0)
                        quant_tensor = torch.cat(dist.nn.all_gather(quant_tensor, group=ep_mesh.get_group()), dim=0)
                    layer_state_dict[scale_name] = scale_tensor.detach()
                    # set `param` which will be added to state_dict at the bottom of the for-block
                    param = quant_tensor

                param = param.to(DEVICE)
                name = f"model.layers.{i}.{sub_name}"
                saved_keys.append(name.replace("model.", ""))
                if ".experts." in name and ".mlp." not in name:
                    name = name.replace(".experts.", ".mlp.experts.")
                if ".gate." in name and ".mlp." not in name:
                    name = name.replace(".gate.", ".mlp.gate.")
                layer_state_dict[name] = param.detach()
            gather_duration.append(time.perf_counter() - start)
            start = time.perf_counter()
            self.request_update_params(layer_state_dict)
            weight_duration.append(time.perf_counter() - start)

            start = time.perf_counter()
            del layer_state_dict
            layer.reshard()
            reshard_duration.append(time.perf_counter() - start)

        if dist.get_rank() == 0:
            logger.info(
                f"Rank 0 Gather decoder layers done, total {sum(gather_duration):.2f}s, avg "
                f"{sum(gather_duration) / len(gather_duration):.2f}s"
            )
            logger.info(
                f"Rank 0 migrate/save decoder layers done, total {sum(weight_duration):.2f}s, avg "
                f"{sum(weight_duration) / len(weight_duration):.2f}s"
            )
            logger.info(
                f"Rank 0 reshard decoder layers done, total {sum(reshard_duration):.2f}s, avg "
                f"{sum(reshard_duration) / len(reshard_duration):.2f}s"
            )

        # update other params
        model.norm.unshard()
        model.lm_head.unshard()
        model.embed_tokens.unshard()
        others_state_dict = {}
        for name, param in model.named_parameters():
            if "_checkpoint_wrapped_module." in name:
                continue
            if name not in saved_keys:
                saved_keys.append(name)
                if name == "norm.weight":
                    name = "model.norm.weight"
                if name == "embed_tokens.weight":
                    name = "model.embed_tokens.weight"
                if isinstance(param, DTensor):
                    param = param.to_local()
                others_state_dict[name] = param.detach()
        self.request_update_params(others_state_dict, finished=True)
        model.norm.reshard()
        model.lm_head.reshard()
        model.embed_tokens.reshard()
        del others_state_dict
        del param

        dist.barrier()
        DEVICE_MODULE.empty_cache()
        return

    def request_update_params(self, state_dict, finished=False):
        cpu_mesh = self.rollout_device_mesh["engine_parallel"]
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()

        if self.rollout_cfg_info["backend"] == "pytorch" and self.rollout_cfg_info["tp"] > 1:
            serialized_data = [None] * self.rollout_cfg_info["tp"]
            tmp_serialized_data = serialize_state_dict(state_dict)
            dist.gather_object(
                tmp_serialized_data,
                serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
        else:
            serialized_data = serialize_state_dict(state_dict)

        if dist.get_rank() == head_rank:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.rollout_cfg_info['api_key']}",
            }
            data = dict(serialized_named_tensors=serialized_data, finished=finished)
            response = requests.post(
                f"{self.rollout_url}/{self.endpoints['update_weights']}", headers=headers, json=data
            )
            assert response.status_code == 200, f"response.status_code = {response.status_code}"

        if finished:
            dist.barrier(group=cpu_group)
        return


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


@ray.remote(
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        }
    },
)
class GRPOTrainingWorker(TrainingWorker):
    def _build_engine(self, worker_cfg: WorkerConfig) -> GRPOTrainEngine:
        engine = GRPOTrainEngine(
            optim_cfg=worker_cfg.optim_cfg,
            fsdp_cfg=worker_cfg.fsdp_cfg,
            model_cfg=worker_cfg.model_cfg,
        )
        if worker_cfg.load_from is not None:
            engine.from_hf(worker_cfg.load_from)
        return engine
