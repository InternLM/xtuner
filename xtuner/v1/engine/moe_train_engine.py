from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig, MoEConfig, MoELossConfig, OptimConfig
from xtuner.v1.config.float8 import Float8Config
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.collator import ColateItem
from xtuner.v1.engine.dense_train_engine import DenseTrainEngine

# todo: 如何 import
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.loss import BalancingLoss, ZLoss
from xtuner.v1.model.moe.moe import MoE

# from xpuyu.models.auto import AutoFullyShardModel
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.router import GreedyRouterConfig, NoAuxRouterConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class MoETrainEngine(DenseTrainEngine):
    model_cfg: MoEConfig
    model: MoE
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
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

    # todo: @yehaochen
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
            ].mlp.gate.e_score_correction_bias  # TODO: (caoweihan) update bias should be a method of `MoE` model
            expected_load = expected_loads[i_layer]
            current_loads = total_expert_counts_pre_iter[i_layer]

            load_diff = current_loads - expected_load
            update_mask = load_diff != 0  # 只更新需要调整的专家
            updates = torch.where(load_diff > 0, -bias_update_speed, bias_update_speed) * update_mask.float()

            e_score_correction_bias.add_(updates)

    def train_step(self, data_batches: List[ColateItem], intra_layer_micro_batch: int = 1, sp_mesh: DeviceMesh = None):  # type: ignore
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
            max_length (Optional[int]): The maximum sequence length for padding.
            intra_layer_micro_batch (int): The number of micro-batches for intra-layer all2all overlap.
            sp_mesh (Optional[DeviceMesh]): The device mesh for sequence parallelism.
        """
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

        for colate_item in data_batches:
            colate_item["seq_ctx"].to(DEVICE)
            colate_item["labels"].to(DEVICE)

        log = {}
        global_grad_tokens = self.cal_global_grad_tokens([i["labels"] for i in data_batches], sp_mesh)
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
            labels_list = []
            for data in data_batch:
                seq_ctx = data["seq_ctx"]
                labels = data["labels"]
                # shift_seq_ctx and labels have been split in data_preprocess if sequence parallelism is enabled
                if sp_mesh:
                    seq_ctx, labels = seq_ctx.split_with_labels(labels, sp_mesh)  # type: ignore

                seq_ctx_list.append(seq_ctx)
                labels_list.append(labels)
                step_consumed_tokens += seq_ctx.mask.sum()

            # llm_loss_list, router_logits_list = self.model(
            #     seq_ctx=seq_ctx_list[0],
            #     labels=shift_labels_list[0],
            # )
            output = self.model(
                seq_ctx=seq_ctx_list[0],
                labels=labels_list[0],  # type: ignore
                return_router_results=True,
            )
            llm_loss_list = [output["loss"]]
            router_logits_list = [[val["logits"] for val in output["router_logits"].values()]]

            # global average llm loss
            llm_loss = torch.tensor(0.0, device=DEVICE)
            for loss, labels in zip(llm_loss_list, labels_list):
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

    # todo: 调用 model 里的 scale grad
    def step_optimizer(self):
        """Step the optimizer to update the model parameters."""
        grad_norm = self.clip_grad_norm()
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return grad_norm

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        """Save the hf model to the given directory.

        Args:
            hf_dir (str): The directory to save the model.
            save_dtype (torch.dtype): The dtype to save the model parameters, bfloat16 or float8.
        """
        self.model.save_hf(hf_dir=hf_dir, save_dtype=save_dtype)
