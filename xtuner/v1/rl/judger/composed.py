from __future__ import annotations

import asyncio
from typing import Callable, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import RolloutState

from .native import Judger, JudgerConfig, JudgerOutput


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
        self.branches = branches
        # ``merge_fn=None`` is only valid for routing mode, where each sample
        # selects exactly one branch and that branch's reward is passed through.
        # If ``data_source`` selects multiple branches, callers must provide an
        # explicit merge function because reward aggregation is task-specific.
        self.merge_fn = merge_fn

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

    def _resolve_selected_keys(self, rollout_state: RolloutState | list[RolloutState]) -> list[str]:
        if isinstance(rollout_state, list):
            if not rollout_state:
                raise ValueError("ComposedJudger requires at least one RolloutState when input is a list.")
            return self._select_keys_from_data_source(rollout_state[0])
        return self._select_keys_from_data_source(rollout_state)

    async def _judge_branch(
        self,
        key: str,
        rollout_state: RolloutState | list[RolloutState],
    ) -> tuple[str, JudgerOutput | list[JudgerOutput]]:
        branch = self.branches[key]
        if isinstance(rollout_state, list):  # batch samples judge
            payloads = [branch.preprocess(state) for state in rollout_state]
            return key, await branch.judge_payload(payloads)
        # single sample judge
        payload = branch.preprocess(rollout_state)
        return key, await branch.judge_payload(payload)

    def _postprocess_single_branch(
        self,
        branch: Judger,
        rollout_state: RolloutState | list[RolloutState],
        output: JudgerOutput | list[JudgerOutput],
    ) -> RolloutState | list[RolloutState]:
        if isinstance(rollout_state, list):
            if not isinstance(output, list):
                raise TypeError("Single selected branch returned a single output for a rollout state list.")
            if len(output) != len(rollout_state):
                raise ValueError(
                    f"Single selected branch returned {len(output)} outputs for {len(rollout_state)} states."
                )
            return [branch.postprocess(state, output_i) for state, output_i in zip(rollout_state, output)]

        if isinstance(output, list):
            raise TypeError("Single selected branch returned a list output for one RolloutState.")
        return branch.postprocess(rollout_state, output)

    async def judge(self, rollout_state: RolloutState | list[RolloutState]) -> RolloutState | list[RolloutState]:  # type: ignore[override]
        selected_keys = self._resolve_selected_keys(rollout_state)

        if len(selected_keys) == 1:
            # 处理每次只选择其中一个 branch 的情况
            key = selected_keys[0]
            _, output = await self._judge_branch(key, rollout_state)
            return self._postprocess_single_branch(self.branches[key], rollout_state, output)

        if self.merge_fn is None:
            raise ValueError(
                "ComposedJudger selected multiple branches but merge_fn is not provided. "
                f"selected_keys={selected_keys!r}"
            )

        judged = dict(await asyncio.gather(*(self._judge_branch(key, rollout_state) for key in selected_keys)))
        return self.merge_fn(rollout_state, judged)


class ComposedJudgerConfig(BaseModel):
    """Configuration for composing multiple judgers.

    ``ComposedJudgerConfig`` routes each rollout to one or more branch judgers
    and merges the branch outputs back into a single ``RolloutState``. It is
    useful when different samples in the same task require different reward
    functions or when multiple rewards must be computed together.

    Args:
        branches (dict[str, JudgerConfig | ComposedJudgerConfig]): Mapping from
            branch name to judger configuration.
        merge_fn (JudgerMergeFn | None): Function that merges multiple branch
            outputs into the returned rollout state. Required when ``data_source``
            may select more than one branch.

    **Examples:**

    Example composed judger with two branches::

        config = ComposedJudgerConfig(
            branches={
                "math": GSM8KJudgerConfig(),
                "format": JudgerConfig(judger_name="format", reward_handler=format_reward),
            },
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
