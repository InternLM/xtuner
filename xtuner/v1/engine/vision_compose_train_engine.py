from pathlib import Path
from typing import List, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoProcessor
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.compose.base import BaseComposeConfig, BaseComposeModel
from xtuner.v1.model.moe.moe import MoEModelOutputs
from xtuner.v1.model.utils import ModelForwardExtraLogInfo
from xtuner.v1.module.router import NoAuxRouterConfig
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from .train_engine import LossLog, OtherLog, TrainEngine


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class VisionComposeTrainEngine(TrainEngine):
    model_cfg: BaseComposeConfig  # type: ignore
    model: BaseComposeModel  # type: ignore
    llm_float8_handler: Float8Handler | None
    vision_float8_handler: Float8Handler | None
    projector_float8_handler: Float8Handler | None

    def __init__(
        self,
        model_cfg: BaseComposeConfig,
        *args,
        **kwargs,
    ) -> None:
        self._processor = None  # only for save
        super().__init__(model_cfg, *args, **kwargs)  # type: ignore

    def build_model(self) -> BaseComposeModel:  # type: ignore
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

        if dist.get_rank() == 0:
            logger.info(model)

        if self.llm_float8_handler:
            self.llm_float8_handler.build_reduce_mesh(
                model.language_model, cast(DeviceMesh, model.language_model.fsdp_mesh)
            )
        if self.vision_float8_handler:
            self.vision_float8_handler.build_reduce_mesh(
                model.vision_tower, cast(DeviceMesh, model.vision_tower.fsdp_mesh)
            )
        if self.projector_float8_handler:
            self.projector_float8_handler.build_reduce_mesh(
                model.multi_modal_projector, cast(DeviceMesh, model.multi_modal_projector.fsdp_mesh)
            )
        return model

    def from_hf(self, hf_path: str | Path, strict: bool = False):
        super().from_hf(hf_path, strict)
        self._processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        super().save_hf(hf_dir, save_dtype)
        if self._processor is not None:
            self._processor.save_pretrained(hf_dir)

    # this method can be called outside, e.g., at the beginning of compute_actor_logprobs or compute_ref_logprobs during rl training
    def maybe_precompute_float8_dynamic_scale_for_fsdp(self):
        if self.llm_float8_handler is not None and self.llm_float8_handler.enabled:
            self.llm_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.language_model)
        if self.vision_float8_handler is not None and self.vision_float8_handler.enabled:
            self.vision_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.vision_tower)
        if self.projector_float8_handler is not None and self.projector_float8_handler.enabled:
            self.projector_float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model.multi_modal_projector)

    def train_step(self, data_batches: List[ModelItem]) -> tuple[LossLog, OtherLog]:
        """Perform a training step with the given data batches and mesh.

        Args:
            data_batches (List[Dict]): The input data batches for the training step.
        """
        self.maybe_precompute_float8_dynamic_scale_for_fsdp()

        loss_log: LossLog = {}  # type: ignore[typeddict-item]
        other_log: OtherLog = {}  # type: ignore[typeddict-item]
        intra_layer_micro_batch = self.intra_layer_micro_batch
        assert len(data_batches) % intra_layer_micro_batch == 0, (
            f"data_batches length {len(data_batches)} is not divisible by intra_layer_micro_batch {intra_layer_micro_batch}"
        )
        iters_per_step = self.grad_accumulation_steps(len(data_batches))

        if self._count == 0:
            logger.info(f"grad_accumulation_steps: {iters_per_step}")
            self._count += 1

        moe_need_update_bias = (
            isinstance(getattr(self.model_cfg.text_config, "router", None), NoAuxRouterConfig)
            and self.model_cfg.text_config.router.router_bias_update_speed > 0
        )
        moe_need_log_maxvio = getattr(self.model_cfg.text_config, "router", None) is not None
        if moe_need_log_maxvio:
            tokens_per_expert_global_for_bias = torch.zeros(
                self.model_cfg.text_config.num_hidden_layers - self.model_cfg.text_config.first_k_dense_replace,
                self.model_cfg.text_config.n_routed_experts,
                dtype=torch.int64,
                device=DEVICE,
            )

        step_loss = torch.tensor(0.0, device=DEVICE)
        step_llm_loss = torch.tensor(0.0, device=DEVICE)
        step_balancing_loss: torch.Tensor | None = None
        step_z_loss: torch.Tensor | None = None
        step_consumed_tokens = torch.tensor(0, device=DEVICE)
        efficient_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)
        total_forward_tokens = torch.tensor(0, device=DEVICE, dtype=torch.long)

        train_engine_extra_info = ModelForwardExtraLogInfo()
        step_consumed_img_tokens = 0.0
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

                if seq_ctx.num_img_tokens is not None:
                    step_consumed_img_tokens += sum(seq_ctx.num_img_tokens)
                    if seq_ctx.sequence_parallel_mesh:
                        step_consumed_img_tokens /= seq_ctx.sequence_parallel_mesh.size()

                num_tokens = seq_ctx.cu_seq_lens_k[1:] - seq_ctx.cu_seq_lens_k[:-1]
                efficient_forward_tokens += (num_tokens.long() ** 2).sum()
                total_forward_tokens += (num_tokens.long().sum()) ** 2

            # todo: support intra_layer_micro_batch
            output = self.model(seq_ctx=seq_ctx_list[0], loss_ctx=loss_ctx_list[0])
            # llm loss has been global averaged
            llm_loss = output["loss"]
            step_llm_loss += llm_loss.detach().clone()

            loss = llm_loss
            if "extra_info" in output:
                train_engine_extra_info.append(output["extra_info"])

            if "balancing_loss" in output:
                output = cast(MoEModelOutputs, output)
                loss = loss + output["balancing_loss"] / iters_per_step
                step_balancing_loss = (
                    output["balancing_loss"]
                    if step_balancing_loss is None
                    else step_balancing_loss + output["balancing_loss"]
                )
            if "z_loss" in output:
                output = cast(MoEModelOutputs, output)
                loss = loss + output["z_loss"] / iters_per_step
                step_z_loss = output["z_loss"] if step_z_loss is None else step_z_loss + output["z_loss"]

            if moe_need_log_maxvio:
                output = cast(MoEModelOutputs, output)
                assert "tokens_per_expert_global" in output, "tokens_per_expert_global is required for bias update."
                tokens_per_expert_global_for_bias += output["tokens_per_expert_global"]

            del output
            loss.backward()
            step_loss += loss.detach().clone()

        if moe_need_log_maxvio:
            avg_count_load = tokens_per_expert_global_for_bias.float().mean(1)
            max_load_i, _ = torch.max(tokens_per_expert_global_for_bias, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            maxvio = maxvio_all_layers.mean()
            if moe_need_update_bias:
                self.model.language_model.update_bias(tokens_per_expert_global_for_bias, avg_count_load)  # type: ignore
            other_log["maxvio"] = maxvio.item()

        reduced_llm_loss = step_llm_loss
        dist.all_reduce(reduced_llm_loss.div_(dist.get_world_size()))

        loss_log["local_loss"] = step_loss.item()
        loss_log["reduced_llm_loss"] = reduced_llm_loss.item()
        if step_balancing_loss is not None:
            reduced_balancing_loss = step_balancing_loss
            dist.all_reduce(reduced_balancing_loss.div_(dist.get_world_size()))
            loss_log["reduced_balancing_loss"] = reduced_balancing_loss.item()
        if step_z_loss is not None:
            reduced_z_loss = step_z_loss
            dist.all_reduce(reduced_z_loss.div_(dist.get_world_size()))
            loss_log["reduced_z_loss"] = reduced_z_loss.item()

        other_log["step_consumed_tokens"] = int(step_consumed_tokens.item())
        other_log["extra_info"] = train_engine_extra_info  # type: ignore[assignment]
        other_log["efficient_attn_ratio"] = (efficient_forward_tokens / total_forward_tokens).item()
        other_log["step_consumed_img_tokens"] = int(step_consumed_img_tokens)

        extra_info = other_log.get("extra_info", {})  # type: ignore

        # TODO: @duanyanhui `extra_info` should be redesigned.
        if not isinstance(extra_info, ModelForwardExtraLogInfo):
            extra_info = ModelForwardExtraLogInfo(extra_info)
        loss_log.update(extra_info.get())

        if "maxvio" in other_log:
            loss_log["maxvio"] = other_log["maxvio"]  # type: ignore
        loss_log["efficient_attn_ratio"] = other_log["efficient_attn_ratio"]  # type: ignore
        return loss_log, other_log
