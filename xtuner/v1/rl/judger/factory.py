from .composed import ComposedJudger, ComposedJudgerConfig, JudgerConfigLike, default_merge_fn
from .native import Judger, JudgerConfig, JudgerPool
from xtuner.v1.rl.utils import get_cpu_resource_manager


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
    external_cpu_allocation = None
    if config.external_cpu is not None:
        external_cpu_manager = get_cpu_resource_manager()
        if external_cpu_manager is None:
            raise ValueError(
                f"Judger {config.judger_name!r} sets external_cpu, "
                "but no CPUResourceManager was provided."
            )
        external_cpu_allocation = external_cpu_manager.register(
            name=f"judger:{config.judger_name}",
            config=config.external_cpu,
        )

    if external_cpu_allocation is None:
        return config.build_local()
    if external_cpu_allocation.num_actors == 1:
        return config._build_remote_judger(external_cpu_allocation=external_cpu_allocation)
    return JudgerPool(
        replicas=config._build_remote_judgers(external_cpu_allocation=external_cpu_allocation),
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
