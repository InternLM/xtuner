from __future__ import annotations

import asyncio
from typing import Callable, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import BaseJudger, Judger, JudgerConfig, JudgerOutput


# Merge function contract for multi-branch composed judging:
# - The first argument is the original rollout state.
# - The second argument maps each selected branch key to that branch's raw judger output.
# - The function must return the same shape as the input rollout state with ``reward`` populated.
JudgerMergeFn: TypeAlias = Callable[
    [RolloutState | list[RolloutState], dict[str, JudgerOutput | list[JudgerOutput]]],
    RolloutState | list[RolloutState],
]


class ComposedJudger(Judger):
    def __init__(
        self,
        branches: dict[str, Judger],
        merge_fn: JudgerMergeFn | None = None,
    ):
        super().__init__()
        if not branches:
            raise ValueError("ComposedJudger requires at least one branch.")
        self._validate_branches(branches)
        self.branches = cast(dict[str, Judger], branches)
        # ``merge_fn=None`` is only valid for routing mode, where each sample
        # selects exactly one branch and that branch's reward is passed through.
        # If ``data_source`` selects multiple branches, callers must provide an
        # explicit merge function because reward aggregation is task-specific.
        self.merge_fn = merge_fn

    def _validate_branches(self, branches: dict[str, Judger]) -> None:
        for key, branch in branches.items():
            if isinstance(branch, Judger):
                continue
            if isinstance(branch, BaseJudger):
                # ComposedJudger intentionally composes branches through the
                # Judger payload contract instead of calling arbitrary
                # BaseJudger.judge() implementations. This avoids deep-copying
                # RolloutState for every branch, which would add serialization
                # overhead and can be risky for large or externally owned
                # rollout fields. Without deepcopy, BaseJudger-only branches
                # could concurrently mutate the same RolloutState.
                raise TypeError(
                    "ComposedJudger branch must inherit Judger, not only BaseJudger. "
                    "BaseJudger-only branches implement their own judge flow and are not supported in "
                    f"ComposedJudger yet. branch={key!r}, type={type(branch).__name__}"
                )
            raise TypeError(
                f"ComposedJudger branch must be a Judger instance. branch={key!r}, type={type(branch).__name__}"
            )

    def _select_keys_from_data_source(self, rollout_state: RolloutState) -> list[str]:
        data_source = rollout_state.data_source
        if data_source is None:
            raise ValueError(
                "ComposedJudger requires rollout_state.data_source to route judger branches. "
                f"task_name={rollout_state.task_name!r}, available={sorted(self.branches)}"
            )
        if isinstance(data_source, str):
            if data_source not in self.branches:
                raise KeyError(
                    f"Unknown judger branch from data_source: {data_source!r}, available={sorted(self.branches)}"
                )
            return [data_source]
        if isinstance(data_source, dict):
            if not data_source:
                raise ValueError("ComposedJudger data_source dict must contain at least one judger branch.")
            selected_keys = []
            for key in data_source:
                if not isinstance(key, str):
                    raise TypeError(f"ComposedJudger data_source dict keys must be strings, got {key!r}.")
                if key not in self.branches:
                    raise KeyError(
                        f"Unknown judger branch from data_source: {key!r}, available={sorted(self.branches)}"
                    )
                selected_keys.append(key)
            return selected_keys

        raise TypeError(
            "ComposedJudger data_source must be a branch name string or a dict of branch names "
            f"got {type(data_source).__name__}: {data_source!r}. "
            f"task_name={rollout_state.task_name!r}, available={sorted(self.branches)}"
        )

    async def _judge_branch(
        self,
        key: str,
        rollout_state: RolloutState,
    ) -> tuple[str, JudgerOutput]:
        branch = self.branches[key]
        payload = branch.preprocess(rollout_state)
        output = await branch.judge_payload(payload)
        if isinstance(output, list):
            raise TypeError(f"Branch {key!r} returned a list output for one RolloutState.")
        return key, output

    async def _batch_judge_branch(
        self,
        key: str,
        rollout_states: list[RolloutState],
    ) -> tuple[str, list[JudgerOutput]]:
        branch = self.branches[key]
        payloads = [branch.preprocess(state) for state in rollout_states]
        outputs = await branch.judge_payload(payloads)
        if not isinstance(outputs, list):
            raise TypeError(f"Branch {key!r} returned a single output for a rollout state list.")
        if len(outputs) != len(rollout_states):
            raise ValueError(f"Branch {key!r} returned {len(outputs)} outputs for {len(rollout_states)} states.")
        return key, outputs

    def _postprocess_branch_batch(
        self,
        branch: Judger,
        rollout_states: list[RolloutState],
        outputs: list[JudgerOutput],
    ) -> list[RolloutState]:
        return [branch.postprocess(state, output) for state, output in zip(rollout_states, outputs)]

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        selected_keys = self._select_keys_from_data_source(rollout_state)

        if len(selected_keys) == 1:
            key = selected_keys[0]
            _, output = await self._judge_branch(key, rollout_state)
            branch = self.branches[key]
            return branch.postprocess(rollout_state, output)

        if self.merge_fn is None:
            raise ValueError(
                "ComposedJudger selected multiple branches but merge_fn is not provided. "
                f"selected_keys={selected_keys!r}"
            )

        judged = dict[str, JudgerOutput | list[JudgerOutput]](
            await asyncio.gather(*(self._judge_branch(key, rollout_state) for key in selected_keys))
        )
        merged = self.merge_fn(rollout_state, judged)
        if isinstance(merged, list):
            raise TypeError("ComposedJudger merge_fn returned a list for judge.")
        return merged

    async def batch_judge(self, rollout_states: list[RolloutState]) -> list[RolloutState]:
        if not rollout_states:
            raise ValueError("ComposedJudger requires at least one RolloutState when input is a list.")
        selected_keys = self._select_keys_from_data_source(rollout_states[0])

        if len(selected_keys) == 1:
            key = selected_keys[0]
            branch = self.branches[key]
            _, outputs = await self._batch_judge_branch(key, rollout_states)
            return self._postprocess_branch_batch(branch, rollout_states, outputs)

        if self.merge_fn is None:
            raise ValueError(
                "ComposedJudger selected multiple branches but merge_fn is not provided. "
                f"selected_keys={selected_keys!r}"
            )

        judged = dict[str, JudgerOutput | list[JudgerOutput]](
            await asyncio.gather(*(self._batch_judge_branch(key, rollout_states) for key in selected_keys))
        )
        merged = self.merge_fn(rollout_states, judged)
        if not isinstance(merged, list):
            raise TypeError("ComposedJudger merge_fn returned a single RolloutState for batch_judge.")
        return merged


class ComposedJudgerConfig(BaseModel):
    """Configuration for composing multiple judgers.

    ``ComposedJudgerConfig`` routes rollout states through
    ``RolloutState.data_source``. A string value selects one branch and passes
    that branch output through as ``RolloutState.reward``. A dict value selects
    multiple branches by key and requires ``merge_fn`` to define the final
    reward shape.

    Args:
        branches (dict[str, JudgerConfig | ComposedJudgerConfig]): Mapping from
            branch name to judger configuration. Branch names must match
            ``RolloutState.data_source`` string values or dict keys.
        merge_fn (JudgerMergeFn | None): Function that merges multiple branch
            outputs into the returned rollout state. Required when ``data_source``
            may select more than one branch. Leave as ``None`` only when every
            sample selects exactly one branch.

    **Examples:**

    Example composed judger with two branches::

        config = ComposedJudgerConfig(
            branches={
                "math": GSM8KJudgerConfig(),
                "format": JudgerConfig(judger_name="format", reward_handler=format_reward),
            },
            merge_fn=merge_rewards,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    branches: dict[str, JudgerConfigLike]
    # Branch routing is fixed to ``RolloutState.data_source``:
    # - str selects one branch.
    # - dict keys select multiple branches.
    # ``merge_fn=None`` means single-branch pass-through only. If data_source
    # may select multiple branches, this must be set explicitly.
    merge_fn: JudgerMergeFn | None = Field(default=None, exclude=True)

    def build(self) -> Judger:
        from .factory import build_judger

        return build_judger(self)


JudgerConfigLike: TypeAlias = JudgerConfig | ComposedJudgerConfig

ComposedJudgerConfig.model_rebuild()
