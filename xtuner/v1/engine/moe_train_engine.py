import json
import math
import os
import types
from typing import Dict, List, Optional, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmengine import mkdir_or_exist
from packaging import version
from safetensors.torch import save_file
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._functional_collectives import all_reduce
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp
from tqdm import tqdm

from xtuner.v1.config import AdamWConfig, Float8Config, FSDPConfig, LRConfig, MoEConfig, MoELossConfig, OptimConfig
from xtuner.v1.engine.dense_train_engine import DenseTrainEngine, HFCheckpointLoader

# todo: 如何 import
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear
from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear
from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.loss import BalancingLoss, ZLoss
from xtuner.v1.model import build_model
from xtuner.v1.model.moe.moe import MoE

# from xpuyu.models.auto import AutoFullyShardModel
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEBlock
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.router import GreedyRouterConfig, NoAuxRouterConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module
from xtuner.v1.utils.compile import maybe_compile


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class MoETrainEngine(DenseTrainEngine):
    model_cfg: MoEConfig
    model: MoE
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    ep_mesh: DeviceMesh
    fsdp_mesh: DeviceMesh
    checkpoint_loader: Optional[HFCheckpointLoader] = None
    float8_handler: Optional[Float8Handler] = None

    def __init__(
        self,
        *,
        model_cfg: MoEConfig,
        moe_loss_cfg: MoELossConfig,
        optim_cfg: OptimConfig,
        lr_cfg: LRConfig,
        fsdp_cfg: FSDPConfig,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        self.balancing_loss = BalancingLoss(moe_loss_cfg=moe_loss_cfg)
        self.z_loss = ZLoss(moe_loss_cfg=moe_loss_cfg)

    def trainable_moe_parameters(self):
        requried_grad_moe_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and ".experts." in name:
                requried_grad_moe_params.append(param)
        return requried_grad_moe_params

    def trainable_non_moe_parameters(self):
        requried_grad_non_moe_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and ".experts." not in name:
                requried_grad_non_moe_params.append(param)
        return requried_grad_non_moe_params

    def init_device_mesh(self, device: str = "cuda"):
        world_size = dist.get_world_size()

        experts_fsdp_size = world_size // self.fsdp_cfg.ep_size

        if self.fsdp_cfg.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                device,
                (experts_fsdp_size, self.fsdp_cfg.ep_size),
                mesh_dim_names=(f"{self.fsdp_cfg.mesh_prefix}.fsdp", f"{self.fsdp_cfg.mesh_prefix}.ep"),
            )
            self.ep_mesh = model_mesh[f"{self.fsdp_cfg.mesh_prefix}.ep"]
            self.fsdp_mesh = model_mesh[f"{self.fsdp_cfg.mesh_prefix}.fsdp"]
        else:
            assert self.fsdp_cfg.ep_size == 1, "Currently, HSDP requires expert parallel size to be 1"
            # We can not init ep_mesh and fsdp_mesh like this.
            # This will lead to "RuntimeError: Cannot create a submesh from a submesh."
            # in FSDPParam.shard_mesh, as fsdp_mesh is not the root mesh. The root mesh is model_mesh.
            # So we have to init the ep_mesh and fsdp_mesh separately.
            # model_mesh = init_device_mesh(
            #     device,
            #     (
            #         experts_fsdp_size // self.fsdp_cfg.hsdp_sharding_size,
            #         self.fsdp_cfg.hsdp_sharding_size,
            #         self.fsdp_cfg.ep_size,
            #     ),
            #     mesh_dim_names=(
            #         f"{self.fsdp_cfg.mesh_prefix}.hsdp_replicate",
            #         f"{self.fsdp_cfg.mesh_prefix}.hsdp_shard",
            #         f"{self.fsdp_cfg.mesh_prefix}.ep",
            #     ),
            # )
            # self.ep_mesh = model_mesh[f"{self.fsdp_cfg.mesh_prefix}.ep"]
            # self.fsdp_mesh = model_mesh[
            #     (f"{self.fsdp_cfg.mesh_prefix}.hsdp_replicate", f"{self.fsdp_cfg.mesh_prefix}.hsdp_shard")
            # ]
            self.ep_mesh = init_device_mesh(
                device, (world_size, 1), mesh_dim_names=("_", f"{self.fsdp_cfg.mesh_prefix}.ep")
            )[f"{self.fsdp_cfg.mesh_prefix}.ep"]
            self.fsdp_mesh = init_device_mesh(
                device,
                (
                    experts_fsdp_size // self.fsdp_cfg.hsdp_sharding_size,
                    self.fsdp_cfg.hsdp_sharding_size,
                ),
                mesh_dim_names=(
                    f"{self.fsdp_cfg.mesh_prefix}.hsdp_replicate",
                    f"{self.fsdp_cfg.mesh_prefix}.hsdp_shard",
                ),
            )

    def apply_ep(self, model: nn.Module):
        for module in model.modules():
            if isinstance(module, GroupedLinear):
                weight = nn.Parameter(distribute_tensor(module.weight, self.ep_mesh, [Shard(0)]))
                module.register_parameter("weight", weight)

    def replicate_other_params(self, model: nn.Module):
        def traverse(module):
            if isinstance(module, MoEBlock):
                return
            for name, param in module.named_parameters(recurse=False):
                dist_param = nn.Parameter(distribute_tensor(param, self.ep_mesh, [Replicate()]))
                module.register_parameter(name, dist_param)
            for child in module.children():
                traverse(child)

        traverse(model)

    def prepare_for_experts_parallel(self, model: nn.Module):
        self.apply_ep(model)
        self.replicate_other_params(model)

    def load_moe_param(self, param: DTensor, hf_keys: List):
        ep_size = self.ep_mesh.size()
        ep_rank = self.ep_mesh.get_local_rank()
        # use mesh_dim=-1 to support hsdp
        fsdp_size = self.fsdp_mesh.size(mesh_dim=-1)
        fsdp_rank = self.fsdp_mesh.get_local_rank(mesh_dim=-1)
        ne = self.model_cfg.n_routed_experts
        is_fused_w1w3 = len(hf_keys) == 2 * ne
        dout = 2 * self.model_cfg.moe_intermediate_size if is_fused_w1w3 else self.model_cfg.hidden_size
        # dout = param.shape[0] // ne
        # after apply experts parallel:
        # start = ep_rank * (ne // ep_size) * dout
        # end = (ep_rank + 1) * (ne // ep_size) * dout
        # after apply fsdp:
        len_rank = torch.tensor([param._local_tensor.shape[0]], device=DEVICE, dtype=torch.int32)
        len_global = torch.zeros(fsdp_size, device=DEVICE, dtype=torch.int32)
        dist.all_gather_into_tensor(len_global, len_rank, group=self.fsdp_mesh.get_group(mesh_dim=-1))
        len_global_cumsum = torch.cumsum(len_global, dim=0)
        len_global_cumsum = [0] + len_global_cumsum.tolist()  # type: ignore
        # the local tensor is located at [start, end) of the global tensor
        start = ep_rank * (ne // ep_size) * dout + len_global_cumsum[fsdp_rank]
        end = ep_rank * (ne // ep_size) * dout + len_global_cumsum[fsdp_rank + 1]

        if start == end:
            # The entire local param is fsdp padded tensor
            return param

        idx_start = start // (dout // (1 + int(is_fused_w1w3)))
        idx_end = end // (dout // (1 + int(is_fused_w1w3)))
        weights_list = []
        if idx_end > len(hf_keys) or (idx_end == len(hf_keys) and end % (dout // (1 + int(is_fused_w1w3))) > 0):
            assert (
                self.float8_handler is not None
                and self.float8_handler.scaling_granularity_gemm == ScalingGranularity.TILEWISE
            )
        if idx_start >= len(hf_keys):
            # The entire local param is fsdp padded tensor
            param._local_tensor.copy_(0.0)  # type: ignore
            return param
        if idx_start == idx_end:
            weight = self.checkpoint_loader.load(hf_keys[idx_start]).to(DEVICE)  # type: ignore
            weight_start = start % (dout // (1 + int(is_fused_w1w3)))
            weight_end = end % (dout // (1 + int(is_fused_w1w3)))
            weights_list.append(weight[weight_start:weight_end])
        else:
            weight = self.checkpoint_loader.load(hf_keys[idx_start]).to(DEVICE)  # type: ignore
            weight_start = start % (dout // (1 + int(is_fused_w1w3)))
            weights_list.append(weight[weight_start:])
            # use min(idx_end, len(hf_keys)) to support padding
            for idx in range(idx_start + 1, min(idx_end, len(hf_keys))):
                weight = self.checkpoint_loader.load(hf_keys[idx]).to(DEVICE)  # type: ignore
                weights_list.append(weight)
            weight_end = end % (dout // (1 + int(is_fused_w1w3)))
            if weight_end > 0 and idx_end < len(hf_keys):
                weight = self.checkpoint_loader.load(hf_keys[idx_end]).to(DEVICE)  # type: ignore
                weights_list.append(weight[:weight_end])
        weights = torch.cat(weights_list, dim=0)
        assert weights.shape[0] <= param._local_tensor.shape[0]
        non_pad_len = weights.shape[0]
        param._local_tensor[:non_pad_len].copy_(weights)
        param._local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        return param

    def load_dense_param(self, param: DTensor, hf_key: str):
        # use mesh_dim=-1 to support hsdp
        fsdp_size = self.fsdp_mesh.size(mesh_dim=-1)
        fsdp_rank = self.fsdp_mesh.get_local_rank(mesh_dim=-1)
        len_rank = torch.tensor([param._local_tensor.shape[0]], device=DEVICE, dtype=torch.int32)
        len_global = torch.zeros(fsdp_size, device=DEVICE, dtype=torch.int32)
        dist.all_gather_into_tensor(len_global, len_rank, group=self.fsdp_mesh.get_group(mesh_dim=-1))
        len_global_cumsum = torch.cumsum(len_global, dim=0)
        len_global_cumsum = [0] + len_global_cumsum.tolist()  # type: ignore
        # the local tensor is located at [start, end) of the global tensor
        start = len_global_cumsum[fsdp_rank]
        end = len_global_cumsum[fsdp_rank + 1]
        if start == end:
            # The entire local param is fsdp padded tensor
            return param

        weight = self.checkpoint_loader.load(hf_key).to(DEVICE)  # type: ignore
        if end > weight.shape[0]:
            assert (
                self.float8_handler is not None
                and self.float8_handler.scaling_granularity_gemm == ScalingGranularity.TILEWISE
            )
            non_pad_len = weight.shape[0] - start
            if non_pad_len > 0:
                param._local_tensor[:non_pad_len].copy_(weight[start:])
                param._local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
            else:
                param._local_tensor.copy_(0.0)  # type: ignore
        else:
            param._local_tensor.copy_(weight[start:end])
        return param

    def maybe_compile_layers(self):
        if self.fsdp_cfg.torch_compile:
            if self.fsdp_cfg.compile_targets is None:
                if self.ep_mesh.size() > 1:
                    # all_to_all_single_autograd in TorchAll2AllDispatcher.dispatch can not be compiled even if the fullgraph=False
                    # ref: https://github.com/pytorch/pytorch/issues/155205
                    # todo: decorate MoEDecoderLayer.forward with @torch.compile(fullgraph=False) when the bug is fixed
                    # so that we do not need to remove the compile target
                    maybe_compile.remove_compile_target(
                        "xtuner.v1.module.decoder_layer.moe_decoder_layer.MoEDecoderLayer.forward"
                    )
            else:
                maybe_compile.clear_compile_targets()
                for target in self.fsdp_cfg.compile_targets:
                    maybe_compile.set_compile_target(target)
        else:
            maybe_compile.clear_compile_targets()

    def fully_shard(self, model, model_prefix: str = ""):
        if self.fsdp_cfg.requires_grad:
            for module in model.modules():
                for p_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        param_fp32 = torch.nn.Parameter(param.to(dtype=torch.float32))
                        setattr(module, p_name, param_fp32)
        else:
            for param in model.parameters():
                param.requires_grad = False

        if self.ep_mesh.size() > 1:
            self.prepare_for_experts_parallel(model)

        model.rotary_emb = model.build_rotary_embedding(self.model_cfg)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.fsdp_cfg.param_dtype, reduce_dtype=self.fsdp_cfg.reduce_dtype
        )
        num_recompute_layers = int(self.model_cfg.num_hidden_layers * self.fsdp_cfg.recompute_ratio)
        layer_prefix = f"{model_prefix}layers"

        self.maybe_compile_layers()

        for layer_idx, layer in tqdm(model.layers.items(), desc="[FSDP Sharding]"):
            layer_idx = int(layer_idx)
            layer.to_empty(device=DEVICE_MODULE.current_device())
            if layer_idx < num_recompute_layers - 1:
                layer = ptd_checkpoint_wrapper(
                    layer,
                    preserve_rng_state=False,
                    checkpoint_impl=CheckpointImpl.REENTRANT,
                )

            model.layers[str(layer_idx)] = layer
            if layer_idx >= len(model.layers) - 1:
                reshard_after_forward = False
            else:
                reshard_after_forward = self.fsdp_cfg.reshard_after_forward
            fully_shard(
                layer,
                mesh=self.fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if self.fsdp_cfg.cpu_offload else None,
            )

            if self.checkpoint_loader:
                # load ckpt
                for key, value in layer.state_dict().items():
                    full_key = f"{layer_prefix}.{layer_idx}.{key}"
                    full_key = self.clean_param_name(full_key)
                    hf_keys = model.to_hf_key_list(full_key)
                    # inplace load
                    if ".experts." in full_key:
                        self.load_moe_param(value, hf_keys)
                    else:
                        self.load_dense_param(value, hf_keys)

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            for layer_cur, layer_next in zip(
                list(model.layers.values())[:-1],
                list(model.layers.values())[1:],
            ):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

        model.embed_tokens.to_empty(device=DEVICE)
        fully_shard(
            model.embed_tokens,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_cfg.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_cfg.cpu_offload else None,
        )
        if self.checkpoint_loader:
            self.load_dense_param(model.embed_tokens.weight, f"{model_prefix}model.embed_tokens.weight")

        model.norm.to_empty(device=DEVICE)
        fully_shard(
            model.norm,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_cfg.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_cfg.cpu_offload else None,
        )
        if self.checkpoint_loader:
            self.load_dense_param(model.norm.weight, f"{model_prefix}model.norm.weight")

        model.lm_head.to_empty(device=DEVICE)
        fully_shard(
            model.lm_head,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_cfg.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_cfg.cpu_offload else None,
        )
        if self.checkpoint_loader:
            self.load_dense_param(model.lm_head.weight, f"{model_prefix}lm_head.weight")

        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=self.fsdp_cfg.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if self.fsdp_cfg.cpu_offload else None,
        )
        model.set_modules_to_forward_prefetch([model.embed_tokens, model.layers["0"]])

        # if self.float8_handler is not None and self.float8_handler.enabled:
        #     self.float8_handler.build_reduce_mesh_devided_64(self.fsdp_mesh, self.ep_mesh)
        #     self.float8_handler.build_reduce_mesh_mapping(model, self.fsdp_mesh, self.ep_mesh)

        # todo: remove
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(
                module, (TileWiseFloat8Linear, TensorWiseFloat8Linear)
            ):
                module.forward = types.MethodType(MoETrainEngine.patched_linear_forward, module)  # type: ignore
            elif isinstance(module, nn.Embedding):
                module.forward = types.MethodType(MoETrainEngine.patched_emb_forward, module)  # type: ignore
            elif isinstance(module, RMSNorm):
                module.forward = types.MethodType(MoETrainEngine.patched_rms_norm_forward, module)  # type: ignore
        return model

    @staticmethod
    def patched_linear_forward(self, input):
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)

    @staticmethod
    def patched_rms_norm_forward(self, input):
        if hasattr(self, "weight"):
            if isinstance(self.weight, DTensor):
                w = self.weight.to_local()
            else:
                w = self.weight
        else:
            if isinstance(self.norm.weight, DTensor):
                w = self.norm.weight.to_local()
            else:
                w = self.norm.weight
        return F.rms_norm(input, w.shape, w, self.variance_epsilon)

    @staticmethod
    def patched_emb_forward(self, input):
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
        else:
            w = self.weight
        return F.embedding(
            input,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def build_model(self) -> nn.Module:
        # self.init_device_mesh()
        with torch.device("meta"):
            model = build_model(self.model_cfg, self.ep_mesh)

        if self.model_cfg.float8_cfg is not None and self.model_cfg.float8_cfg.enable_float8:
            self.float8_handler = Float8Handler(
                scaling_granularity_gemm=self.model_cfg.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=self.model_cfg.float8_cfg.scaling_granularity_grouped_gemm,
            )
            self.float8_handler.pad_for_fsdp(model, self.fsdp_mesh)

        return self.fully_shard(model)

    def init_model(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
            elif isinstance(module, GroupedLinear):
                # Initialize the weight of GroupedLinear
                module.weight.data.normal_(mean=0.0, std=0.02)

    @torch.no_grad()
    def cal_tokens_per_expert(self, router_logits: torch.Tensor):
        scoring_func = self.model_cfg.router.scoring_func
        n_routed_experts = self.model_cfg.n_routed_experts
        num_experts_per_tok = self.model_cfg.num_experts_per_tok
        num_layers = router_logits.shape[0]
        router_logits = router_logits.float()  # (nlayers, seq, ne)
        if scoring_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=-1)
        elif scoring_func == "sigmoid":
            routing_weights = router_logits / torch.sum(router_logits, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown scoring function: {scoring_func}")
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
        selected_experts_flat = selected_experts.view(num_layers, -1)
        offset = torch.arange(num_layers, device=router_logits.device).unsqueeze(1) * n_routed_experts
        selected_experts_offset = selected_experts_flat + offset
        tokens_per_expert_flat = torch.histc(
            selected_experts_offset.view(-1),
            bins=num_layers * n_routed_experts,
            min=0,
            max=num_layers * n_routed_experts,
        )
        tokens_per_expert = tokens_per_expert_flat.view(num_layers, n_routed_experts)  # (nlayers, ne)
        tokens_per_expert_global_for_bias = all_reduce(tokens_per_expert, "sum", dist.group.WORLD)  # type: ignore
        return tokens_per_expert_global_for_bias

    def select_non_pad_router_logits(
        self, router_logits_list: List[List[torch.Tensor]], attn_mask_list: List[torch.Tensor]
    ):
        # router_logits_list [intra_layer_micro_batch, num_layers][seq, num_experts]
        # attn_mask_list [intra_layer_micro_batch, ][1, seq]
        intra_layer_micro_batch = len(router_logits_list)
        num_layers = len(router_logits_list[0])

        router_logits_list_new = []  # [num_layers, intra_layer_micro_batch] -> [num_layers * intra_layer_micro_batch]
        for layer_idx in range(num_layers):
            for micro_batch_idx in range(intra_layer_micro_batch):
                router_logits_list_new.append(router_logits_list[micro_batch_idx][layer_idx])

        router_logits = torch.stack(
            router_logits_list_new, dim=0
        )  # [num_layers * intra_layer_micro_batch, seq, num_experts]
        router_logits = router_logits.view(
            num_layers, -1, router_logits.shape[-1]
        )  # [num_layers, intra_layer_micro_batch * seq, num_experts]
        attn_mask = torch.stack(attn_mask_list, dim=0)  # [intra_layer_micro_batch, 1, seq]
        attn_mask = attn_mask.flatten()
        router_logits = router_logits[:, attn_mask].contiguous().float()  # [num_layers, non_pad_seq, num_experts]
        return router_logits

    @torch.no_grad()
    def update_bias(self, total_expert_counts_pre_iter, expected_loads):
        """Implementation for the following paper:
        Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
        https://arxiv.org/abs/2408.15664

        TODO: refactor it later.
        """
        first_k_dense_replace = self.model_cfg.first_k_dense_replace
        bias_update_speed = self.model_cfg.router.router_bias_update_speed
        n_layer, n_routed_experts = total_expert_counts_pre_iter.size()

        for i_layer in range(n_layer):
            # 前 l 层是 mlp 层，跳过
            e_score_correction_bias = self.model.model.layers[
                first_k_dense_replace + i_layer
            ].mlp.gate.e_score_correction_bias
            expected_load = expected_loads[i_layer]
            current_loads = total_expert_counts_pre_iter[i_layer]

            load_diff = current_loads - expected_load
            update_mask = load_diff != 0  # 只更新需要调整的专家
            updates = torch.where(load_diff > 0, -bias_update_speed, bias_update_speed) * update_mask.float()

            e_score_correction_bias.add_(updates)

    def train_step(self, data_batches: List[Dict], intra_layer_micro_batch: int = 1, sp_mesh: DeviceMesh = None):  # type: ignore
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
            max_length (Optional[int]): The maximum sequence length for padding.
            intra_layer_micro_batch (int): The number of micro-batches for intra-layer all2all overlap.
            sp_mesh (Optional[DeviceMesh]): The device mesh for sequence parallelism.
        """
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

        log = {}
        global_grad_tokens = self.cal_global_grad_tokens(data_batches, sp_mesh)
        assert len(data_batches) % intra_layer_micro_batch == 0, (
            f"data_batches length {len(data_batches)} is not divisible by intra_layer_micro_batch {intra_layer_micro_batch}"
        )
        iters_per_step = len(data_batches) // intra_layer_micro_batch

        need_update_bias = (
            isinstance(self.model_cfg.router, NoAuxRouterConfig) and self.model_cfg.router.router_bias_update_speed > 0
        )
        if need_update_bias:
            tokens_per_expert_global_for_bias = torch.tensor(0, device=DEVICE)

        step_loss = torch.tensor(0.0, device=DEVICE)
        step_llm_loss = torch.tensor(0.0, device=DEVICE)
        step_balancing_loss = torch.tensor(0.0, device=DEVICE)
        step_z_loss = torch.tensor(0.0, device=DEVICE)
        step_consumed_tokens = torch.tensor(0.0, device=DEVICE)

        for i in range(0, len(data_batches), intra_layer_micro_batch):
            data_batch = data_batches[i : i + intra_layer_micro_batch]
            seq_ctx_list = []
            shift_labels_list = []
            for data in data_batch:
                # shift_seq_ctx and labels have been split in data_preprocess if sequence parallelism is enabled
                shift_seq_ctx, shift_labels = self.data_preprocess(data, sp_mesh)
                seq_ctx_list.append(shift_seq_ctx)
                shift_labels_list.append(shift_labels)
                step_consumed_tokens += shift_seq_ctx.mask.sum()

            # llm_loss_list, router_logits_list = self.model(
            #     seq_ctx=seq_ctx_list[0],
            #     labels=shift_labels_list[0],
            # )
            output = self.model(
                seq_ctx=seq_ctx_list[0],
                labels=shift_labels_list[0],
                return_router_results=True,
            )
            llm_loss_list = [output["loss"]]
            router_logits_list = [[val["logits"] for val in output["router_logits"].values()]]

            # global average llm loss
            llm_loss = torch.tensor(0.0, device=DEVICE)
            for loss, labels in zip(llm_loss_list, shift_labels_list):
                rank_grad_tokens = (labels >= 0).sum()
                # tp size == 1
                llm_loss += loss * rank_grad_tokens / global_grad_tokens * dist.get_world_size()
            step_llm_loss += llm_loss.detach().clone()

            # aux_loss = self.cal_aux_loss() # None | dict[str, torch.Tensor]

            router_logits = self.select_non_pad_router_logits(
                router_logits_list, attn_mask_list=[seq_ctx.mask for seq_ctx in seq_ctx_list]
            )

            # aux_loss has been global averaged
            balancing_loss = self.balancing_loss(
                router_logits=router_logits,
                n_routed_experts=self.model_cfg.n_routed_experts,
                num_experts_per_tok=self.model_cfg.num_experts_per_tok,
            )
            z_loss = self.z_loss(router_logits=router_logits)
            loss = llm_loss + (balancing_loss + z_loss) / iters_per_step
            step_balancing_loss += balancing_loss.detach().clone() / iters_per_step
            step_z_loss += z_loss.detach().clone() / iters_per_step
            if dist.get_rank() == 0:
                print(llm_loss, balancing_loss, z_loss, loss)

            if need_update_bias:
                tokens_per_expert_global = self.cal_tokens_per_expert(router_logits)
                tokens_per_expert_global_for_bias += tokens_per_expert_global

            del llm_loss_list, router_logits_list, router_logits

            loss.backward()
            step_loss += loss.detach().clone()

        maxvio = torch.tensor(0.0, device=DEVICE)
        if need_update_bias:
            avg_count_load = tokens_per_expert_global_for_bias.float().mean(1)
            max_load_i, _ = torch.max(tokens_per_expert_global_for_bias, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            maxvio = maxvio_all_layers.mean()
            self.update_bias(tokens_per_expert_global_for_bias, avg_count_load)

        grad_norm = self.step_optimizer()
        self.lr_scheduler.step()

        reduced_llm_loss = step_llm_loss
        dist.all_reduce(reduced_llm_loss.div_(dist.get_world_size()))
        reduced_balancing_loss = step_balancing_loss
        dist.all_reduce(reduced_balancing_loss.div_(dist.get_world_size()))
        reduced_z_loss = step_z_loss
        dist.all_reduce(reduced_z_loss.div_(dist.get_world_size()))

        log["lr"] = self.lr_scheduler.get_last_lr()[0]
        log["total_loss"] = step_loss.item()
        log["reduced_llm_loss"] = reduced_llm_loss.item()
        log["reduced_balancing_loss"] = reduced_balancing_loss.item()
        log["reduced_z_loss"] = reduced_z_loss.item()
        log["maxvio"] = maxvio.item()
        log["grad_norm"] = grad_norm.item()
        log["consumed_tokens"] = step_consumed_tokens.item()
        return log

    def step_optimizer(self):
        """Step the optimizer to update the model parameters."""
        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            self.scale_moe_grad()
            self.reduce_non_moe_grad()
        grad_norm = self.clip_grad_norm()
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm

    def scale_moe_grad(self):
        """Scale the gradients of moe parameters."""
        with torch.no_grad():
            for param in self.trainable_moe_parameters():
                if param.grad is None:
                    continue
                param.grad.div_(self.ep_mesh.size())

    def reduce_non_moe_grad(self):
        """Reduce the gradients of non-moe parameters across ep group.

        Note that fsdp only do the reduce-scatter communication across fsdp group, non-moe parameters are replicated
        across ep group.
        """
        with torch.no_grad():
            for param in self.trainable_non_moe_parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(
                    param.grad.to_local().div_(self.ep_mesh.size()),
                    ReduceOp.SUM,
                    group=self.ep_mesh.get_group(mesh_dim=0),
                )

    def handle_dense_param(self, hf_key: str, param: DTensor | torch.Tensor, layer_idx: str):
        # If ep size is 1, the param is a torch.Tensor, otherwise it is a DTensor.
        state_dict = {}
        index = {}
        index[hf_key] = f"layer-{layer_idx}-dense.safetensors"
        # param: torch.Tensor or DTensor ep Replicate()
        param = param._local_tensor if isinstance(param, DTensor) else param
        state_dict[hf_key] = param.contiguous().to("cpu", non_blocking=True)
        return state_dict, index

    def handle_moe_param(
        self, hf_keys: List[str], param: DTensor | torch.Tensor, expert_saving_device_mesh: DeviceMesh, layer_idx: str
    ):
        state_dict = {}
        index = {}
        is_fused_w1w3 = len(hf_keys) == 2 * self.model_cfg.n_routed_experts

        expert_parallel_saving_size = expert_saving_device_mesh.shape[1]
        expert_parallel_saving_rank = dist.get_rank(expert_saving_device_mesh.get_group(1))

        n_routed_experts = self.model_cfg.n_routed_experts
        n_local_experts = n_routed_experts // expert_parallel_saving_size

        # rank0 needs to record in which file each parameter is stored
        for idx, hf_key in enumerate(hf_keys):
            eid = idx // (1 + int(is_fused_w1w3))
            index[hf_key] = (
                f"layer-{layer_idx}-moe-{eid // n_local_experts}-of-{expert_parallel_saving_size}.safetensors"
            )

        hf_keys = hf_keys[
            expert_parallel_saving_rank * n_local_experts * (1 + int(is_fused_w1w3)) : (
                expert_parallel_saving_rank + 1
            )
            * n_local_experts
            * (1 + int(is_fused_w1w3))
        ]

        # param: torch.Tensor or DTensor ep Shard(0)
        param_local = param._local_tensor if isinstance(param, DTensor) else param
        if self.ep_mesh is None or self.ep_mesh.size() == 1:
            # param: torch.Tensor
            # ep_size == 1, expert parallel is used to save model efficiently
            param_local = distribute_tensor(
                param, expert_saving_device_mesh["expert_parallel_saving"], [Shard(0)]
            )._local_tensor

        # param_local (n_local_experts * (1 + int(is_fused_w1w3)) * dout, din)
        param_list = torch.chunk(param_local, n_local_experts * (1 + int(is_fused_w1w3)), dim=0)
        for key, param in zip(hf_keys, param_list):
            state_dict[key] = param.contiguous().to("cpu", non_blocking=True)
        return state_dict, index

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        """Save the hf model to the given directory.

        Args:
            hf_dir (str): The directory to save the model.
            save_dtype (torch.dtype): The dtype to save the model parameters, bfloat16 or float8.
        """
        mkdir_or_exist(hf_dir)
        DEVICE_MODULE.empty_cache()
        assert save_dtype in [torch.float8_e4m3fn, torch.bfloat16], f"save_dtype {save_dtype} is not supported"

        n_routed_experts = self.model_cfg.n_routed_experts
        world_size = dist.get_world_size()

        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            # If ep is enabled, moe params are sharded across ep_mesh after module.unshard().
            # param.full_tensor() is necessary to get the correct part of the param if we set
            # a different expert_parallel_saving_size. And full_tensor is expensive.
            expert_parallel_saving_size = self.ep_mesh.size()
        else:
            # We use expert_parallel_saving_size gpus to save the parameters of the MoE part in parallel,
            # improving the model-saving speed. However, we aim to avoid setting
            # the degree of parallelism too high, as this could result in
            # saving an excessive number of small files.
            expert_parallel_saving_size = math.gcd(n_routed_experts, world_size)
            if expert_parallel_saving_size > max(32, self.ep_mesh.size()):
                for expert_parallel_saving_size_new in range(expert_parallel_saving_size, 0, -1):
                    if (
                        expert_parallel_saving_size % expert_parallel_saving_size_new == 0
                        and expert_parallel_saving_size_new <= max(32, self.ep_mesh.size())
                    ):
                        expert_parallel_saving_size = expert_parallel_saving_size_new
                        break
        expert_saving_device_mesh = init_device_mesh(
            DEVICE,
            (world_size // expert_parallel_saving_size, expert_parallel_saving_size),
            mesh_dim_names=("expert_replicate_saving", "expert_parallel_saving"),
        )
        expert_replicate_saving_rank = dist.get_rank(expert_saving_device_mesh.get_group(0))
        expert_parallel_saving_rank = dist.get_rank(expert_saving_device_mesh.get_group(1))
        rank = dist.get_rank()

        index = {}
        saved_keys = []

        for layer_idx, layer in tqdm(self.model.layers.items(), desc="[Save HF]"):  # type: ignore
            cast(FSDPModule, layer).unshard()
            layer_dense_state_dict = {}
            layer_moe_state_dict = {}
            suffix_dense = f"layer-{layer_idx}-dense.safetensors"
            suffix_moe = (
                f"layer-{layer_idx}-moe-{expert_parallel_saving_rank}-of-{expert_parallel_saving_size}.safetensors"
            )
            layer_dense_save_path = os.path.join(hf_dir, suffix_dense)
            layer_moe_save_path = os.path.join(hf_dir, suffix_moe)

            for name, param in layer.named_parameters(prefix=f"layers.{layer_idx}"):
                # name = f"model.layers.{layer_idx}.{sub_name}"
                name = self.clean_param_name(name)
                saved_keys.append(name)
                hf_keys = cast(MoE, self.model).to_hf_key_list(name)
                if ".experts." in name:
                    # moe params
                    state_dict_cur, index_cur = self.handle_moe_param(
                        hf_keys=cast(List[str], hf_keys),
                        param=param,
                        expert_saving_device_mesh=expert_saving_device_mesh,
                        layer_idx=layer_idx,
                    )
                    layer_moe_state_dict.update(state_dict_cur)
                else:
                    # dense params
                    state_dict_cur, index_cur = self.handle_dense_param(
                        hf_key=cast(str, hf_keys),
                        param=param,
                        layer_idx=layer_idx,
                    )
                    layer_dense_state_dict.update(state_dict_cur)
                index.update(index_cur)

            for name, buffer in layer.named_buffers(prefix=f"layers.{layer_idx}"):
                # name = f"model.layers.{layer_idx}.{sub_name}"
                name = self.clean_param_name(name)
                hf_key = cast(MoE, self.model).to_hf_key_list(name)
                saved_keys.append(cast(str, hf_key))
                layer_dense_state_dict[hf_key] = buffer.contiguous().to("cpu", non_blocking=True)
                index[hf_key] = suffix_dense

            DEVICE_MODULE.synchronize()
            layer.reshard()

            if rank == 0:
                save_file(layer_dense_state_dict, layer_dense_save_path)
            if expert_replicate_saving_rank == 0 and len(layer_moe_state_dict) > 0:
                save_file(layer_moe_state_dict, layer_moe_save_path)

        others_state_dict = {}
        others_save_path = os.path.join(hf_dir, "others.safetensors")
        for name, param in self.model.named_parameters():
            name = self.clean_param_name(name)
            if name in saved_keys:
                continue
            hf_key = cast(MoE, self.model).to_hf_key_list(name)
            others_state_dict[hf_key] = cast(DTensor, param).full_tensor().contiguous().to("cpu", non_blocking=True)
            index[hf_key] = "others.safetensors"

        for name, buffer in self.model.named_buffers():
            name = self.clean_param_name(name)
            if name in saved_keys:
                continue
            hf_key = cast(MoE, self.model).to_hf_key_list(name)
            others_state_dict[hf_key] = buffer.contiguous().to("cpu", non_blocking=True)
            index[hf_key] = "others.safetensors"

        total_size = sum([param.numel() for param in self.model.parameters()])
        metadata = {"total_size": total_size * 2}
        if rank == 0:
            save_file(others_state_dict, others_save_path)
            with open(os.path.join(hf_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump({"metadata": metadata, "weight_map": index}, f, ensure_ascii=False, indent=4)
