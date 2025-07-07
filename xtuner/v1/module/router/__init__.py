from xtuner.v1.config import MoEConfig

from .greedy import GreedyRouter, GreedyRouterConfig
from .noaux_router import NoAuxRouter, NoAuxRouterConfig
from .protocol import RouterProtocol, RouterResults


def build_router(config: MoEConfig) -> RouterProtocol:
    """Get the router based on the configuration."""
    if isinstance(config.router, NoAuxRouterConfig):
        return NoAuxRouter(config)
    elif isinstance(config.router, GreedyRouterConfig):
        return GreedyRouter(config)
    else:
        raise ValueError(f"Unsupported router type: {type(config.router)}")


__all__ = [
    "RouterProtocol",
    "NoAuxRouter",
    "NoAuxRouterConfig",
    "GreedyRouterConfig",
    "GreedyRouter",
    "build_router",
    "RouterResults",
]
