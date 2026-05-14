from xtuner.v1.rl.utils import register_cpu_resources

from .composed import ComposedJudger, ComposedJudgerConfig, JudgerConfigLike, default_merge_fn
from .native import Judger, JudgerConfig, JudgerPool


#
# Use ``JudgerConfig`` when one sample only needs one concrete judger implementation:
# one reward handler, one judger_name, and one execution mode (local or Ray actors).
#
# Use ``ComposedJudgerConfig`` when one sample may need to be routed to different child
# judgers by ``select_fn``, or when you want to run multiple child judgers and merge their
# outputs with ``merge_fn``.
#
def build_judger(config: JudgerConfigLike) -> Judger:
    if isinstance(config, ComposedJudgerConfig):
        return _build_composite_judger(config)
    return _build_replicated_judger(config)


def _build_replicated_judger(config: JudgerConfig) -> Judger:
    if config.cpu_resources is None:
        return config.build_local()

    register_cpu_resources(
        name=f"judger:{config.judger_name}",
        cpu_resources=config.cpu_resources,
    )

    if config.cpu_resources.num_workers == 1:
        return config._build_remote_judger(cpu_resources=config.cpu_resources)
    return JudgerPool(
        replicas=config._build_remote_judgers(cpu_resources=config.cpu_resources),
        judger_name=config.judger_name,
    )


def _build_composite_judger(config: ComposedJudgerConfig) -> Judger:
    branches: dict[str, Judger] = {}
    for key, branch_config in config.branches.items():
        branches[key] = build_judger(branch_config)
    return ComposedJudger(
        branches=branches,
        select_fn=config.select_fn,
        merge_fn=config.merge_fn or default_merge_fn,
        default_key=config.default_key,
    )
