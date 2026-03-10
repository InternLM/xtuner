from typing import Annotated

from cyclopts import Group, Parameter
from pydantic import BaseModel

from xtuner.v1.rl.advantage.base import AdvantageEstimator


advantage_group = Group("Advantage Estimation", sort_key=2, help="Advantage estimation configuration.")


class BaseAdvantageConfig(BaseModel):
    """Intermediate base for discriminated union."""

    def build(self) -> AdvantageEstimator:
        raise NotImplementedError("Subclasses must implement this method.")


class GRPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.grpo.GRPOEstimator`.

    Attributes:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.grpo import GRPOEstimator

        return GRPOEstimator(eps=self.eps)


class DrGRPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.grpo.DrGRPOEstimator`.

    Attributes:
        max_length (float): Max response length for duration scaling.
            Default 32768.
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    max_length: Annotated[
        float,
        Parameter(group=advantage_group, help="Max response length for duration scaling."),
    ] = 32768
    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.grpo import DrGRPOEstimator

        return DrGRPOEstimator(max_length=self.max_length, eps=self.eps)


class RLOOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for
    :class:`~xtuner.v1.rl.advantage.rloo.RLOOEstimator`."""

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.rloo import RLOOEstimator

        return RLOOEstimator()


class OPOAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.opo.OPOEstimator`.

    Attributes:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-8

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.opo import OPOEstimator

        return OPOEstimator(eps=self.eps)


class PassKAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.passk.PassKEstimator`.

    Attributes:
        k (int): The K in pass@k. Default 4.
        eps (float): Small constant for numerical stability. Default 1e-6.
    """

    k: Annotated[
        int,
        Parameter(group=advantage_group, help="The K in pass@k."),
    ] = 4
    eps: Annotated[
        float,
        Parameter(group=advantage_group, help="Small constant for numerical stability."),
    ] = 1e-6

    def build(self) -> AdvantageEstimator:
        from xtuner.v1.rl.advantage.passk import PassKEstimator

        return PassKEstimator(k=self.k, eps=self.eps)
