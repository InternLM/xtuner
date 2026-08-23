from typing import Annotated

from cyclopts import Group, Parameter
from pydantic import BaseModel, ConfigDict

from xtuner.v1.rl.advantage.base import AdvantageEstimator, TokenLevelAdvantageEstimator


advantage_group = Group("Advantage Estimation", sort_key=2, help="Advantage estimation configuration.")


class BaseAdvantageConfig(BaseModel):
    """Intermediate base for discriminated union."""

    model_config = ConfigDict(extra="forbid")

    def build(self) -> AdvantageEstimator:
        raise NotImplementedError("Subclasses must implement this method.")


class BaseTokenLevelAdvantageConfig(BaseAdvantageConfig):
    """Intermediate base for estimators that require a critic.

    Distinguished from :class:`BaseAdvantageConfig` so the RL trainer can tell,
    from the config alone, whether a value model must be built and whether
    advantages are computed trainer-side or inside the training worker.
    """

    def build(self) -> TokenLevelAdvantageEstimator:  # type: ignore[override]
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


class GAEAdvantageConfig(BaseTokenLevelAdvantageConfig):
    """Configuration for :class:`~xtuner.v1.rl.advantage.gae.GAEEstimator`.

    Selecting this estimator switches the trainer to PPO: a critic must be
    configured, and advantages are computed in the training worker from the
    critic's value predictions rather than from a group reward baseline.

    Attributes:
        gamma (float): Discount factor. Default 1.0, standard for RLHF where
            episodes are short and undiscounted.
        gae_lambda (float): GAE lambda, trading bias against variance. Default 0.95.
        normalize_advantage (bool): Whether to normalize advantages to zero mean
            and unit variance over all action tokens in the global batch.
            Default True.
    """

    gamma: Annotated[
        float,
        Parameter(group=advantage_group, help="Discount factor."),
    ] = 1.0
    gae_lambda: Annotated[
        float,
        Parameter(group=advantage_group, help="GAE lambda, trading bias against variance."),
    ] = 0.95
    normalize_advantage: Annotated[
        bool,
        Parameter(group=advantage_group, help="Normalize advantages across the global batch."),
    ] = True

    def build(self) -> TokenLevelAdvantageEstimator:
        from xtuner.v1.rl.advantage.gae import GAEEstimator

        return GAEEstimator(gamma=self.gamma, gae_lambda=self.gae_lambda)
