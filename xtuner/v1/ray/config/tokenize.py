from __future__ import annotations

from typing import TYPE_CHECKING

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated

from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    from xtuner.v1.ray.rollout.tokenize_controller import TokenizeController


class TokenizeControllerConfig(BaseModel):
    """Configuration for rollout tokenize controller."""

    model_config = ConfigDict(extra="forbid")

    num_ray_actors: Annotated[
        int,
        Parameter(help="Number of ray actors used by tokenize controller. 0 means local tokenize mode."),
    ] = 0
    num_cpus_per_actor: Annotated[
        float,
        Parameter(help="CPU cores allocated for each tokenize ray actor."),
    ] = 1
    num_processes_per_actor: Annotated[
        int,
        Parameter(help="Number of subprocesses inside each tokenize ray actor."),
    ] = 1
    request_timeout: Annotated[
        float,
        Parameter(help="Timeout duration (in seconds) for tokenize requests."),
    ] = 300.0
    enable_spread_scheduling: Annotated[
        bool,
        Parameter(help="Use SPREAD scheduling for tokenize ray actors when actor count > 1."),
    ] = True

    def build(self, tokenizer_path: str) -> TokenizeController:
        from xtuner.v1.ray.rollout.tokenize_controller import TokenizeController

        logger = get_logger(tag="TokenizeControllerConfig")
        if self.num_ray_actors <= 0:
            logger.info("TokenizeController uses local tokenizer mode.")
        return TokenizeController(
            tokenizer_path=tokenizer_path,
            num_ray_actors=self.num_ray_actors,
            num_cpus_per_actor=self.num_cpus_per_actor,
            num_processes_per_actor=self.num_processes_per_actor,
            enable_spread_scheduling=self.enable_spread_scheduling,
        )
