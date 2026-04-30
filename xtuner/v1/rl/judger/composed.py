from __future__ import annotations

from copy import deepcopy
from typing import Callable, TypeAlias

from pydantic import BaseModel, ConfigDict, Field
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import Judger, JudgerConfig


SelectedJudgerKeys: TypeAlias = str | list[str] | None
JudgerSelectFn: TypeAlias = Callable[[RolloutState, dict[str, Judger]], SelectedJudgerKeys]
JudgerMergeFn: TypeAlias = Callable[
    [RolloutState | list[RolloutState], dict[str, RolloutState | list[RolloutState]]],
    RolloutState | list[RolloutState],
]


def default_select_fn(rollout_state: RolloutState, branches: dict[str, Judger]) -> SelectedJudgerKeys:
    """Default branch selector for ``ComposedJudgerConfig``.

    Selection order is intentionally simple:
    1. If ``rollout_state.data_source`` is a string and matches a branch key, use it.
    2. Otherwise return ``None`` and let ``default_key`` or the single-branch fallback decide.

    Users with task-specific routing logic should pass a custom ``select_fn`` instead of extending
    this default heuristic.
    """
    data_source = rollout_state.data_source
    if isinstance(data_source, str) and data_source in branches:
        return data_source

    return None


def default_merge_fn(
    original: RolloutState | list[RolloutState],
    judged: dict[str, RolloutState | list[RolloutState]],
) -> RolloutState | list[RolloutState]:
    """Default merger for ``ComposedJudgerConfig``.

    This merger intentionally does not combine multiple judger scores into a single aggregated value.
    It writes the merged reward as ``{branch_name: score}``, where ``branch_name`` is the selected
    key from ``ComposedJudgerConfig.branches`` and ``score`` is taken from each child judger's
    ``reward["score"]``.

    Supports both single ``RolloutState`` and batched ``list[RolloutState]`` inputs. In the batch
    case, each element in the list represents a different response to the same prompt, and each
    branch's judged result must be a list of the same length.

    Users who need weighted sums, richer reward payloads, or custom post-processing should provide
    their own ``merge_fn``.
    """
    if isinstance(original, list):
        for name, state in judged.items():
            if not isinstance(state, list):
                raise TypeError(
                    f"default_merge_fn: branch {name!r} returned a single RolloutState "
                    "but original is a list. All branches must return lists when input is a list."
                )
            if len(state) != len(original):
                raise ValueError(
                    f"default_merge_fn: branch {name!r} returned {len(state)} states "
                    f"but original has {len(original)} states."
                )
        results: list[RolloutState] = []
        for i, orig in enumerate(original):
            merged = orig.model_copy(deep=True)
            merged.reward = {}
            for name, states in judged.items():
                assert isinstance(states, list)
                state_i: RolloutState = states[i]
                reward = state_i.reward
                if reward is None or "score" not in reward:
                    raise KeyError(f"Default merge_fn requires reward['score'] for branch {name!r}.")
                merged.reward[name] = reward["score"]
            results.append(merged)
        return results
    else:
        merged = original.model_copy(deep=True)
        merged.reward = {}
        for name, state in judged.items():
            if isinstance(state, list):
                raise TypeError(
                    f"default_merge_fn: branch {name!r} returned a list but original is a single RolloutState."
                )
            if state.reward is None or "score" not in state.reward:
                raise KeyError(f"Default merge_fn requires reward['score'] for branch {name!r}.")
            merged.reward[name] = state.reward["score"]
        return merged


class ComposedJudger(Judger):
    def __init__(
        self,
        branches: dict[str, Judger],
        select_fn: JudgerSelectFn = default_select_fn,
        merge_fn: JudgerMergeFn = default_merge_fn,
        default_key: str | None = "default",
    ):
        if not branches:
            raise ValueError("ComposedJudger requires at least one branch.")
        self.branches = branches
        self.select_fn = select_fn
        self.merge_fn = merge_fn
        self.default_key = default_key

    def _resolve_selected_keys(self, rollout_state: RolloutState | list[RolloutState]) -> list[str]:
        if isinstance(rollout_state, list):
            selected = self.select_fn(rollout_state[0], self.branches)
        else:
            selected = self.select_fn(rollout_state, self.branches)

        if selected is None:
            selected_keys: list[str] = []
        elif isinstance(selected, str):
            selected_keys = [selected]
        else:
            selected_keys = list(dict.fromkeys(selected))

        if not selected_keys:
            if self.default_key is not None and self.default_key in self.branches:
                return [self.default_key]
            if len(self.branches) == 1:
                return [next(iter(self.branches))]
            state = rollout_state[0] if isinstance(rollout_state, list) else rollout_state
            raise KeyError(
                f"ComposedJudger could not select a branch for task_name={state.task_name!r}, "
                f"data_source={state.data_source!r}, available={sorted(self.branches)}"
            )
        return selected_keys

    async def judge(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:  # type: ignore[override]
        selected_keys = self._resolve_selected_keys(rollout_state)

        judged: dict[str, RolloutState | list[RolloutState]] = {}
        for key in selected_keys:
            if key not in self.branches:
                raise KeyError(f"Unknown judger branch: {key}, available={sorted(self.branches)}")
            judged[key] = await self.branches[key].judge(deepcopy(rollout_state))
        return self.merge_fn(rollout_state, judged)


class ComposedJudgerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    branches: dict[str, JudgerConfigLike]
    # ``select_fn`` chooses which branch keys should be executed for one sample.
    # Return a single string for single-judger routing, a list of strings for multi-judger execution,
    # or ``None`` to fall back to ``default_key`` / single-branch implicit fallback.
    select_fn: JudgerSelectFn = Field(default=default_select_fn, exclude=True)
    # ``merge_fn`` merges the judged rollout states back into one rollout state.
    # The default implementation does not aggregate scores; it writes ``{branch_name: score}``.
    merge_fn: JudgerMergeFn | None = Field(default=None, exclude=True)
    default_key: str | None = "default"

    def get_num_placement_group_bundles(self) -> int:
        return sum(branch.get_num_placement_group_bundles() for branch in self.branches.values())

    def build(self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> Judger:
        from .factory import build_judger

        return build_judger(self, pg=pg, start_bundle_idx=start_bundle_idx)


JudgerConfigLike: TypeAlias = JudgerConfig | ComposedJudgerConfig

ComposedJudgerConfig.model_rebuild()
