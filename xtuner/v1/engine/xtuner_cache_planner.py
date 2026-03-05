from typing import Optional

from torch.distributed.checkpoint import SavePlanner
from torch.distributed.checkpoint import DefaultSavePlanner, SavePlan, Metadata
from torch.distributed.checkpoint.planner_helpers import (
    _compare_save_plans,
    _merge_delta_local_plans,
)


# copy from torch 2.8.0 planner_helpers.py
def _contains_usable_plan(delta_plans: list[SavePlan]) -> bool:
    """
    Check if any delta plan is usable, indicating the plan has changed.

    Args:
        delta_plans (List[SavePlan]): A list of delta plans to check.
    Returns:
        True if any delta plan is usable, False otherwise.
    """
    return any(delta_plan and delta_plan.usable for delta_plan in delta_plans)


class XtunerCacheSavePlanner(DefaultSavePlanner):
    # Metadata for the global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_metadata: dict[str, Metadata] = {}

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_replicated_tensors: Optional[bool] = None,
        dedup_save_to_lowest_rank: bool = False,
        enable_plan_caching: bool = False,
        cache_key_prefix: str = ""
    ) -> None:
        super().__init__(flatten_state_dict, flatten_sharded_tensors, dedup_replicated_tensors, dedup_save_to_lowest_rank, enable_plan_caching)
        self._cached_plans_key: str = cache_key_prefix + self.__class__.__name__

    def _create_global_plan_with_caching(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], list[SavePlan], Metadata]:

        if hasattr(SavePlanner, "_cached_metadata"):
            # adaptor for torch >= 2.8.0
            return super()._create_global_plan_with_caching(all_plans)
        
        # ONLY cache ``_cached_metadata`` in XtunerCacheSavePlanner
        global_plan_delta: list[SavePlan] = []

        if self._cached_plans_key not in SavePlanner._cached_all_plans:
            # Case 1: If the plans are not cached, the cache will be hydrated with the
            # all_plans, global_plans (Deduped), and metadata.

            # Cache the original all_plans
            SavePlanner._cached_all_plans[self._cached_plans_key] = all_plans
            global_plan, metadata = self._create_global_plan(all_plans)
            # Cache the deduped and validated global_plan
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            # Cache the metadata
            XtunerCacheSavePlanner._cached_metadata[self._cached_plans_key] = metadata
            # If plans are not cached, global_plan delta will be the same as global plan.
            return global_plan, global_plan, metadata

        # Case 2: Plans are cached
        if not _contains_usable_plan(all_plans):
            # Case 2.1: Plans are cached and the local plans have NOT changed (No usable plans).
            # Global plan delta will be empty plans to avoid the collective overhead.
            # We can reuse the deduped global plan and metadata from the cache directly.
            global_plan_delta = [SavePlan([], usable=False)] * len(all_plans)
            global_plan = SavePlanner._cached_global_plan[self._cached_plans_key]
            metadata = XtunerCacheSavePlanner._cached_metadata[self._cached_plans_key]
        else:
            # Case 2.2: Plans are cached but the local plans have changed.
            # We will merge the changed local plans with the cached local plans.
            # Updated plans will overwrite the cached plans. New global plan and metadata will be created and cached.
            # Global plan delta will be created by comparing the new global plan with the cached global plan.
            # Only the global plan delta (updated ones) will be sent to the coordinator to avoid the collective overhead.
            merged_plans = _merge_delta_local_plans(
                SavePlanner._cached_all_plans[self._cached_plans_key], all_plans
            )
            # Cache the updated local plans
            SavePlanner._cached_all_plans[self._cached_plans_key] = merged_plans
            global_plan, metadata = self._create_global_plan(merged_plans)

            if self._cached_plans_key in self._cached_global_plan:
                for cached_plan, new_plan in zip(
                    SavePlanner._cached_global_plan[self._cached_plans_key], global_plan
                ):
                    if _compare_save_plans(cached_plan, new_plan):
                        global_plan_delta.append(SavePlan([], usable=False))
                    else:
                        global_plan_delta.append(new_plan)

            # Cache the new global plan and the metadata
            SavePlanner._cached_global_plan[self._cached_plans_key] = global_plan
            XtunerCacheSavePlanner._cached_metadata[self._cached_plans_key] = metadata

        return global_plan_delta, global_plan, metadata