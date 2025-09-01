from typing import Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from xtuner.v1.data_proto.sequence_context import SequenceContext


@runtime_checkable
class LossConfigProto(Protocol):
    def build(self) -> "BaseModel":
        """Build the model configuration."""
        raise NotImplementedError


class EngineInputItem(TypedDict):
    seq_ctx: SequenceContext
    loss_ctx: "LossContext"


class LossContextInputItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantages: torch.Tensor
    old_logprobs: torch.Tensor  # old_logprobs in train worker's input is None and will be set afterwards
    ref_logprobs: NotRequired[torch.Tensor | None]


class ForwardItem(TypedDict):
    data_batch: list[LossContextInputItem]
    iter_idx: int


class LossContext(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    loss_cfg: LossConfigProto
    forward_item: ForwardItem | None = None

    loss_module: nn.Module | None = None

    def model_post_init(self, _) -> None:
        self.loss_module = self.loss_cfg.build()

    # 只负责透传 data_batch 至 loss fn / module 里，别的什么都不做，保证通用性
    def build_list_ctx(
        self,
        data_batch: list[LossContextInputItem],
    ) -> list["EngineInputItem"]:
        loss_ctx_list = []
        for i, data in enumerate(data_batch):
            forward_item = ForwardItem(
                data_batch=data_batch,
                iter_idx=i,
            )
            loss_ctx = self.__class__(
                loss_cfg=self.loss_cfg,
                forward_item=forward_item,
            )
            loss_ctx_list.append(loss_ctx)

        seq_ctx_list = [data["seq_ctx"] for data in data_batch]
        ret_data_batch = [
            {"seq_ctx": seq_ctx, "loss_ctx": loss_ctx} for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]
        return cast(list[EngineInputItem], ret_data_batch)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        head_weight: torch.Tensor | None = None,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.forward_item is not None, "forward_item must be set before calling forward"
        return cast(nn.Module, self.loss_module)(hidden_states, head_weight, head_bias, self.forward_item)
