from ray.util.placement_group import PlacementGroup

from .dispatch import DispatchJudger, JudgerConfigLike, MultiJudgerConfig, default_merge_fn
from .native import Judger, JudgerConfig, JudgerPool


def build_judger(config: JudgerConfigLike, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> Judger:
    if isinstance(config, MultiJudgerConfig):
        return _build_composite_judger(config, pg=pg, start_bundle_idx=start_bundle_idx)
    return _build_replicated_judger(config, pg=pg, start_bundle_idx=start_bundle_idx)


def _build_replicated_judger(config: JudgerConfig, pg: PlacementGroup | None, start_bundle_idx: int) -> Judger:
    if config.num_ray_actors == 0:
        return config.build_local()
    if config.num_ray_actors == 1:
        return config._build_remote_judger(pg=pg, bundle_idx=start_bundle_idx)
    return JudgerPool(
        replicas=config._build_remote_judgers(pg=pg, start_bundle_idx=start_bundle_idx),
        judger_name=config.judger_name,
    )


def _build_composite_judger(
    config: MultiJudgerConfig,
    pg: PlacementGroup | None,
    start_bundle_idx: int,
) -> Judger:
    branches: dict[str, Judger] = {}
    bundle_idx = start_bundle_idx
    for key, branch_config in config.branches.items():
        branches[key] = build_judger(branch_config, pg=pg, start_bundle_idx=bundle_idx)
        bundle_idx += branch_config.get_num_placement_group_bundles()
    return DispatchJudger(
        branches=branches,
        select_fn=config.select_fn,
        merge_fn=config.merge_fn or default_merge_fn,
        default_key=config.default_key,
    )
