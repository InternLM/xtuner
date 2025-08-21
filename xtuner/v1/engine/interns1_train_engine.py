from typing import List, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config import FSDPConfig, OptimConfig
from xtuner.v1.engine.moe_train_engine import MoETrainEngine

# todo: 如何 import
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.interns1 import InternS1Config, InternS1ForConditionalGeneration
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class InternS1TrainEngine(MoETrainEngine):
    model_cfg: InternS1Config
    model: InternS1ForConditionalGeneration  # type: ignore
    llm_float8_handler: Float8Handler | None
    vision_float8_handler: Float8Handler | None
    projector_float8_handler: Float8Handler | None

    def __init__(
        self,
        *,
        model_cfg: InternS1Config,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig,
        intra_layer_micro_batch: int = 1,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,  # TODO: 这个 type hint 不合理，原则上不能继承 MoETrainEngine
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        if self.llm_float8_handler:
            self.llm_float8_handler.build_reduce_mesh(
                self.model.language_model, cast(DeviceMesh, self.model.language_model.fsdp_mesh)
            )
        if self.vision_float8_handler:
            self.vision_float8_handler.build_reduce_mesh(
                self.model.vision_tower, cast(DeviceMesh, self.model.vision_tower.fsdp_mesh)
            )
        if self.projector_float8_handler:
            self.projector_float8_handler.build_reduce_mesh(
                self.model.multi_modal_projector, cast(DeviceMesh, self.model.multi_modal_projector.fsdp_mesh)
            )
        self.intra_layer_micro_batch = intra_layer_micro_batch

    def build_model(self) -> InternS1ForConditionalGeneration:
        with torch.device("meta"):
            model = self.model_cfg.build()

        self.llm_float8_handler = None
        self.vision_float8_handler = None
        self.projector_float8_handler = None
        if self.model_cfg.text_config.float8_cfg is not None and self.model_cfg.text_config.float8_cfg.enable_float8:
            self.llm_float8_handler = Float8Handler(
                scaling_granularity_gemm=self.model_cfg.text_config.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=self.model_cfg.text_config.float8_cfg.scaling_granularity_grouped_gemm,
            )
        if (
            self.model_cfg.vision_config.float8_cfg is not None
            and self.model_cfg.vision_config.float8_cfg.enable_float8
        ):
            self.vision_float8_handler = Float8Handler(
                scaling_granularity_gemm=self.model_cfg.vision_config.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=self.model_cfg.vision_config.float8_cfg.scaling_granularity_grouped_gemm,
            )
        if (
            self.model_cfg.projector_config.float8_cfg is not None
            and self.model_cfg.projector_config.float8_cfg.enable_float8
        ):
            self.projector_float8_handler = Float8Handler(
                scaling_granularity_gemm=self.model_cfg.projector_config.float8_cfg.scaling_granularity_gemm,
                scaling_granularity_grouped_gemm=self.model_cfg.projector_config.float8_cfg.scaling_granularity_grouped_gemm,
            )
        model.language_model.fully_shard(self.fsdp_cfg, self.llm_float8_handler)
        model.vision_tower.fully_shard(self.fsdp_cfg, self.vision_float8_handler)
        model.multi_modal_projector.fully_shard(self.fsdp_cfg, self.projector_float8_handler)
        model = model.fully_shard(self.fsdp_cfg)
        return model

    @torch.no_grad()
    def cal_tokens_per_expert(self, router_logits: torch.Tensor):
        scoring_func = self.model_cfg.text_config.outer.scoring_func
        n_routed_experts = self.model_cfg.text_config.n_routed_experts
        num_experts_per_tok = self.model_cfg.text_config.num_experts_per_tok
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

    @torch.no_grad()
    def update_bias(self, total_expert_counts_pre_iter, expected_loads):
        """Implementation for the following paper:
        Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
        https://arxiv.org/abs/2408.15664

        TODO: refactor it later.
        """
        first_k_dense_replace = self.model_cfg.text_config.first_k_dense_replace
        bias_update_speed = self.model_cfg.text_config.router.router_bias_update_speed
        n_layer, n_routed_experts = total_expert_counts_pre_iter.size()

        for i_layer in range(n_layer):
            # 前 l 层是 mlp 层，跳过
            e_score_correction_bias = self.model.language_model.model.layers[
                first_k_dense_replace + i_layer
            ].mlp.gate.e_score_correction_bias  # TODO: (caoweihan) update bias should be a method of `MoE` model
            expected_load = expected_loads[i_layer]
            current_loads = total_expert_counts_pre_iter[i_layer]

            load_diff = current_loads - expected_load
            update_mask = load_diff != 0  # 只更新需要调整的专家
            updates = torch.where(load_diff > 0, -bias_update_speed, bias_update_speed) * update_mask.float()

            e_score_correction_bias.add_(updates)

    def train_step(self, data_batches: List[ModelItem], sp_mesh: DeviceMesh = None):  # type: ignore
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
            sp_mesh (Optional[DeviceMesh]): The device mesh for sequence parallelism.
        """
        if self.llm_float8_handler is not None and self.llm_float8_handler.enabled:
            self.llm_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.language_model)
        if self.vision_float8_handler is not None and self.vision_float8_handler.enabled:
            self.vision_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.vision_tower)
        if self.projector_float8_handler is not None and self.projector_float8_handler.enabled:
            self.projector_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.multi_modal_projector)

        loss_log = {}
        other_log = {}
        intra_layer_micro_batch = self.intra_layer_micro_batch
        assert len(data_batches) % intra_layer_micro_batch == 0, (
            f"data_batches length {len(data_batches)} is not divisible by intra_layer_micro_batch {intra_layer_micro_batch}"
        )
        iters_per_step = self.grad_accumulation_steps(len(data_batches))

        need_update_bias = (
            isinstance(self.model_cfg.text_config.router, NoAuxRouterConfig)
            and self.model_cfg.text_config.router.router_bias_update_speed > 0
        )
        if need_update_bias:
            tokens_per_expert_global_for_bias = torch.tensor(0, device=DEVICE)

        step_loss = torch.tensor(0.0, device=DEVICE)
        step_llm_loss = torch.tensor(0.0, device=DEVICE)
        step_balancing_loss: torch.Tensor | None = None
        step_z_loss: torch.Tensor | None = None
        step_consumed_tokens = torch.tensor(0.0, device=DEVICE)

        for i in range(0, len(data_batches), intra_layer_micro_batch):
            data_batch = data_batches[i : i + intra_layer_micro_batch]
            seq_ctx_list = []
            loss_ctx_list = []
            for data in data_batch:
                seq_ctx = data["seq_ctx"]
                loss_ctx = data["loss_ctx"]

                # shift_seq_ctx and labels have been split in data_preprocess if sequence parallelism is enabled
                if sp_mesh:
                    # TODO(HHA): labels 的 sp 逻辑应该由 loss_ctx 做
                    seq_ctx, labels = seq_ctx.split_with_labels(loss_ctx.loss_froward_item.labels, sp_mesh)  # type: ignore
                    loss_ctx.loss_froward_item.labels = labels  # type: ignore
                    loss_ctx: CELossContext = loss_ctx.split(sp_mesh)  # type: ignore

                seq_ctx_list.append(seq_ctx)
                loss_ctx_list.append(loss_ctx)
                step_consumed_tokens += seq_ctx.mask.sum()

            # todo: support intra_layer_micro_batch
            output = self.model(seq_ctx=seq_ctx_list[0], loss_ctx=loss_ctx_list[0])
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

            if need_update_bias:
                assert "tokens_per_expert_global" in output, "tokens_per_expert_global is required for bias update."
                tokens_per_expert_global_for_bias += output["tokens_per_expert_global"]

            del output
            loss.backward()
            step_loss += loss.detach().clone()

        maxvio = torch.tensor(0.0, device=DEVICE)
        if need_update_bias:
            avg_count_load = tokens_per_expert_global_for_bias.float().mean(1)
            max_load_i, _ = torch.max(tokens_per_expert_global_for_bias, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            maxvio = maxvio_all_layers.mean()
            self.model.language_model.update_bias(tokens_per_expert_global_for_bias, avg_count_load)

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
        other_log["maxvio"] = maxvio.item()
        other_log["consumed_tokens"] = step_consumed_tokens.item()
        return loss_log, other_log
