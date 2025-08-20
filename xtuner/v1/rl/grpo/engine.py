import torch

from xtuner.v1.config import FSDPConfig, MoEConfig, OptimConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.engine.moe_train_engine import MoETrainEngine
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module

from ..loss_context import EngineInputItem


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


class GRPOMoETrainEngine(MoETrainEngine):
    def __init__(
        self,
        *,
        model_cfg: MoEConfig,
        optim_cfg: OptimConfig,
        fsdp_cfg: FSDPConfig,
        intra_layer_micro_batch: int = 1,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            intra_layer_micro_batch=intra_layer_micro_batch,
        )

    @torch.no_grad()
    def forward_only(self, seq_ctx: SequenceContext):
        output = self.model(seq_ctx=seq_ctx, loss_ctx=None, return_router_results=False)
        return output

    def train_step(self, data_batches: list[EngineInputItem]):  # type: ignore
        # TODO: support intra-layer micro-batch
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

        loss_log = {}
        other_log = {}

        total_loss = torch.tensor(0.0, device=DEVICE)
        total_grpo_loss = torch.tensor(0.0, device=DEVICE)
        step_consumed_tokens = torch.tensor(0.0, device=DEVICE)

        for data_batch in data_batches:
            seq_ctx = data_batch["seq_ctx"]
            loss_ctx = data_batch["loss_ctx"]
            output = self.model(seq_ctx=seq_ctx, loss_ctx=loss_ctx, return_router_results=False)
            step_loss = output["loss"]
            step_loss.backward()
            total_loss += step_loss.detach()
            total_grpo_loss += output["loss"].detach()
            step_consumed_tokens += seq_ctx.mask.sum()

        loss_log["total_loss"] = total_loss.item()
        loss_log["total_grpo_loss"] = total_grpo_loss.item()
        other_log["consumed_tokens"] = step_consumed_tokens.item()
        return loss_log, other_log
