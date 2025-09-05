import time
from typing import Dict, List, TypeAlias

import ray
import requests
import torch
import torch.distributed as dist
import tqdm
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from xtuner.v1.config.trainer import TrainerConfig
from xtuner.v1.float8.float8_tensor import Float8Tensor
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTilewiseFloat8CastTensor
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils import get_device, get_torch_device_module

from ..accelerator import SingleAcceleratorWorker
from ..config import RolloutConfig


DeviceMeshRaw: TypeAlias = List[List[int]]  # A list of lists representing device mesh indices
ServiceUrlMap: TypeAlias = Dict[int, str]  # A dictionary mapping service names to their URLs
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


@ray.remote(
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        }
    },
)
class TrainingWorker(SingleAcceleratorWorker):
    """Worker class for training tasks.

    This class extends `SingleAcceleratorWorker` to provide functionalities
    specific to training, such as fitting the model and updating weights for
    rollout workers.

    Args:
        config (TrainerConfig): The configuration for the trainer.
        rank (int): The rank of this worker in the distributed setup.
        master_addr (str): The address of the master worker.
        master_port (int): The port of the master worker.
        world_size (int): The total number of workers.
        accelerator (str): The type of accelerator to use (e.g., "GPU").
            Defaults to "GPU".
    """

    def __init__(
        self,
        config: TrainerConfig,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "GPU",
    ):
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)
        # Additional initialization for training can be added here
        self.rank = rank
        self.endpoints: dict[str, str] = dict()
        self.config = config
        self.trainer = Trainer.from_config(config)
        self.rollout_device_mesh: DeviceMesh | None = None
        self.rollout_url: str | None = None
        self.rollout_cfg_info: dict = dict()
        self.endpoints["update_weights"] = "update_weights"

    def get_data_replicate_size(self) -> int:
        """Get the data parallel size for the training worker.

        Returns:
            int: The data parallel size, which is 1 for this worker.
        """
        return 1

    def fit(self):
        """Starts the training process by calling the trainer's `fit`
        method."""
        self.trainer.fit()

    def update_rollout_info(
        self, engine_mesh_list: DeviceMeshRaw, server_url_dict: ServiceUrlMap, rollout_config: RolloutConfig
    ):
        """Update the rollout information for the training worker.

        This method sets up the device mesh and other configuration details
        needed to communicate with the rollout workers.

        Args:
            engine_mesh_list (DeviceMeshRaw): A list of lists representing the
                device mesh for the rollout engines.
            server_url_dict (ServiceUrlMap): A dictionary mapping service names
                to their URLs.
            rollout_config (RolloutConfig): The configuration for the rollout.
        """
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
        """Update the model weights for the rollout workers.

        This method gathers the model weights, handles different tensor types (including Float8Tensors), and sends the
        updated weights to the rollout workers.
        """
        self.endpoints["update_weights"] = "update_weights"
        assert self.rollout_device_mesh is not None

        model = self.trainer._engine.model
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
            self.trainer.logger.info(
                f"Rank 0 Gather decoder layers done, total {sum(gather_duration):.2f}s, avg "
                f"{sum(gather_duration) / len(gather_duration):.2f}s"
            )
            self.trainer.logger.info(
                f"Rank 0 migrate/save decoder layers done, total {sum(weight_duration):.2f}s, avg "
                f"{sum(weight_duration) / len(weight_duration):.2f}s"
            )
            self.trainer.logger.info(
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
        dist.barrier()
        model.norm.reshard()
        model.lm_head.reshard()
        model.embed_tokens.reshard()

        DEVICE_MODULE.empty_cache()
        return

    def request_update_params(self, state_dict, finished=False):
        """Send a request to update the parameters on the rollout workers.

        This method serializes the state dictionary and sends it to the
        appropriate rollout worker via an HTTP request.

        Args:
            state_dict (dict): The state dictionary containing the model
                parameters to update.
            finished (bool): A flag indicating whether this is the final
                batch of updates. Defaults to False.
        """
        cpu_mesh = self.rollout_device_mesh["engine_parallel"]
        cpu_group = cpu_mesh.get_group()
        head_rank = cpu_mesh.mesh[0].item()

        # TODO(chenchiyu): remove lmdeploy related code
        from lmdeploy.utils import serialize_state_dict

        if self.rollout_cfg_info["backend"] == "pytorch" and self.rollout_cfg_info["tp"] > 1:
            serialized_data = [None] * self.rollout_cfg_info["tp"]
            dist.gather_object(
                serialize_state_dict(state_dict),
                serialized_data if dist.get_rank() == head_rank else None,
                dst=head_rank,
                group=cpu_group,
            )
        elif self.rollout_cfg_info["backend"] == "pytorch":
            serialized_data = serialize_state_dict(state_dict)
        else:
            # for turbomind backend, only head_rank should serialize data
            serialized_data = serialize_state_dict(state_dict) if dist.get_rank() == head_rank else None

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
