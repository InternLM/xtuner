from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel


if TYPE_CHECKING:
    from xtuner.v1.loss import BalancingLoss, ZLoss
    from xtuner.v1.loss.base_chunk_loss import BaseChunkLoss
    from xtuner.v1.loss.ce_loss import CrossEntropyLoss, LigerCrossEntropyLoss


class BalancingLossConfig(BaseModel):
    balancing_loss_alpha: float = 0.001
    balancing_loss_global_average: bool = True

    def build(self, router_scoring_func) -> "BalancingLoss":
        from xtuner.v1.loss import BalancingLoss

        return BalancingLoss(
            self.balancing_loss_alpha,
            self.balancing_loss_global_average,
            router_scoring_func=router_scoring_func,
        )


class ZLossConfig(BaseModel):
    z_loss_alpha: float = 0.001
    z_loss_global_average: bool = True

    def build(self) -> "ZLoss":
        from xtuner.v1.loss import ZLoss

        return ZLoss(
            self.z_loss_alpha,
            self.z_loss_global_average,
        )


class CELossConfig(BaseModel):
    loss_reduction: Literal["global", "token", "sample", "square"] = "global"
    label_shifted: bool = True
    loss_name: str = "cross_entropy"
    ignore_idx: int = -100
    chunk_size: int = 1024
    chunk_loss_fn: str = "chunk_ce_loss"

    def build(self) -> "CrossEntropyLoss | LigerCrossEntropyLoss | BaseChunkLoss":
        from xtuner.v1.loss.base_chunk_loss import BaseChunkLoss
        from xtuner.v1.loss.ce_loss import CrossEntropyLoss, LigerCrossEntropyLoss
        from xtuner.v1.loss.chunk_ce_loss import ChunkCELoss, chunk_ce_loss

        if self.loss_name == "cross_entropy":
            return CrossEntropyLoss(loss_cfg=self)
        elif self.loss_name == "liger_cross_entropy":
            return LigerCrossEntropyLoss(loss_cfg=self)
        elif self.loss_name == "chunk_cross_entropy":
            if self.chunk_loss_fn == "chunk_ce_loss":
                chunk_loss_fn = chunk_ce_loss
            else:
                raise NotImplementedError
            # TODO: 这个类只能定义基本结构，无法保证任何场景都不自定义
            return BaseChunkLoss(loss_cfg=self, chunk_loss_class=ChunkCELoss, chunk_loss_fn=chunk_loss_fn)
        else:
            raise NotImplementedError
