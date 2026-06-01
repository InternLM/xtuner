from xtuner.v1.rl.utils import CPUActorLauncher, register_cpu_resources

from .composed import ComposedJudger, ComposedJudgerConfig, JudgerConfigLike
from .native import Judger, JudgerActor, JudgerConfig, JudgerPool, RayJudgerProxy, RemoteJudger


#
# Use ``JudgerConfig`` for one concrete reward handler. The built-in
# ``NativeJudger`` path scores one rollout sample at a time.
#
# Use ``ComposedJudgerConfig`` when one sample may need to be routed to child
# judgers by ``RolloutState.data_source``. A data_source string selects one
# branch; a data_source dict selects multiple branches and requires ``merge_fn``.
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
        return _build_remote_judger(config)
    return JudgerPool(
        replicas=_build_remote_judgers(config),
        judger_name=config.judger_name,
    )


def _build_remote_actor(config: JudgerConfig) -> RayJudgerProxy:
    assert config.cpu_resources is not None
    return CPUActorLauncher.build_actor(
        JudgerActor,
        config,
        actor_num_cpus=config.cpu_resources.num_cpus_per_worker,
        actor_memory=config.cpu_resources.cpu_memory_per_worker,
    )


def _build_remote_judger(config: JudgerConfig) -> Judger:
    return RemoteJudger(_build_remote_actor(config), judger_name=config.judger_name)


def _build_remote_judgers(config: JudgerConfig) -> list[Judger]:
    assert config.cpu_resources is not None
    return [_build_remote_judger(config) for _ in range(config.cpu_resources.num_workers)]


def _build_composite_judger(config: ComposedJudgerConfig) -> Judger:
    branches: dict[str, Judger] = {}
    for key, branch_config in config.branches.items():
        branches[key] = build_judger(branch_config)
    return ComposedJudger(
        branches=branches,
        merge_fn=config.merge_fn,
    )
