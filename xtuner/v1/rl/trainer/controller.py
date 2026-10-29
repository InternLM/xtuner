from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypedDict, cast

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.profile import timer

from .pack import DPRankPackIndices, RLDataPacker, TokenBatchProto
from .worker import TrainBatchAttr, TrainingWorker, WorkerConfig, WorkerLogItem


if TYPE_CHECKING:
    from xtuner.v1.rl.advantage.base import AdvantageEstimator

TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class _DPRankPlan(TypedDict):
    """Global sample mapping and local pack indices for one DP rank."""

    global_indices: list[int]
    plan: DPRankPackIndices


class _PackPlan(TypedDict):
    """Index-only layout and packing metrics, independent of training
    targets."""

    dp_plans: dict[int, _DPRankPlan]
    plan_log: dict[str, float]


class _GroupRewardInfo(TypedDict):
    """Reward clusters and sample mapping for one original rollout group."""

    rewards: list[float]
    representatives: list[RolloutState]
    cluster_indices: list[int | None]


class _PreparedBatch(TypedDict):
    """Outputs of ``_prepare_rollout_items`` consumed by the fit phases."""

    rollout_items: list[RolloutState]
    reward_groups: list[_GroupRewardInfo]
    batch_attr: TrainBatchAttr
    cluster_rewards: list[float]
    distillation_reward_observations: list[tuple[RolloutState, float]]


def _stat_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values).float() if values else torch.tensor([0.0]).float()


def _summarize_process_group_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "ranks=0"

    count_key = next(
        (key for key in ("suspended", "resumed", "destroyed", "reloaded") if key in results[0]),
        "count",
    )
    counts = [result.get(count_key, 0) for result in results]
    count_summary = f"{counts[0]} on all ranks" if len(set(counts)) == 1 else f"by_rank={counts}"
    skipped_counts = [result.get("skipped", 0) for result in results]
    result_errors = [error for result in results for error in result.get("errors", [])]
    summary = f"ranks={len(results)}, {count_key}={count_summary}"
    if any(skipped_counts):
        skipped_summary = (
            f"{skipped_counts[0]} on all ranks" if len(set(skipped_counts)) == 1 else f"by_rank={skipped_counts}"
        )
        summary += f", skipped={skipped_summary}"
    if result_errors:
        summary += f", errors={len(result_errors)}, first_error={result_errors[0]}"
    return summary


def _state_uses_3d_position_ids(state: RolloutState) -> bool:
    layout = state.extra_fields.get("position_layout")
    if layout is not None:
        return layout == "mrope_3d"
    return state.position_ids is not None and state.position_ids.ndim == 3


def _state_has_rollout_logprobs(state: RolloutState) -> bool:
    flag = state.extra_fields.get("has_rollout_logprobs")
    if flag is None:
        return state.logprobs is not None
    return bool(flag)


def _state_has_routed_experts(state: RolloutState) -> bool:
    flag = state.extra_fields.get("has_routed_experts")
    if flag is None:
        return state.routed_experts is not None
    return bool(flag)


class TrainingController:
    def __init__(
        self,
        workers: list[TrainingWorker],
        advantage_estimator: AdvantageEstimator | None = None,
        distillation: DistillationTrainerAdapter | None = None,
        has_critic: bool = False,
    ) -> None:
        self.workers = workers
        self.advantage_estimator = advantage_estimator
        self.distillation = distillation if distillation is not None else DistillationTrainerAdapter(None)
        self.task_adv_weight = self.distillation.task_adv_weight
        self.logger = get_logger()
        self._gae_reward_scope = self._get_gae_reward_scope(advantage_estimator)
        self._has_critic = has_critic

    @staticmethod
    def _get_gae_reward_scope(
        estimator: AdvantageEstimator | None,
    ) -> Literal["segment", "session"] | None:
        from xtuner.v1.rl.advantage.gae import DecoupledGAEEstimator, GAEEstimator

        if isinstance(estimator, (GAEEstimator, DecoupledGAEEstimator)):
            return estimator.reward_scope
        return None

    def _cluster_group_rewards(
        self, group: list[RolloutState]
    ) -> tuple[list[float], list[RolloutState], list[int | None]]:
        # 按 session 聚类奖励：agentic session 可能拆成多个共享同一 reward 的可训练 segment，
        # 每个 session 只计一次，避免 rewards/* 与 advantage 被 segment 数放大。
        # session_id 只由 agentic loop / XTUNER_DETERMINISTIC 写入；普通 RL 回退到 rollout_id。
        rewards_by_session: dict[Any, float] = {}
        cluster_index_by_session: dict[Any, int] = {}
        session_representatives: list[RolloutState] = []
        cluster_indices: list[int | None] = []
        for state in group:
            if state.reward is None or "score" not in state.reward:
                if self.task_adv_weight > 0:
                    raise ValueError(
                        f"Reward is missing or does not contain 'score' key in data: {state}, "
                        f"but task_adv_weight={self.task_adv_weight} > 0"
                    )
                cluster_indices.append(None)
                continue
            reward = float(state.reward["score"])
            session_key = state.session_id if state.session_id is not None else state.rollout_id
            if session_key not in rewards_by_session:
                cluster_index_by_session[session_key] = len(session_representatives)
                rewards_by_session[session_key] = reward
                session_representatives.append(state)
            cluster_indices.append(cluster_index_by_session[session_key])
        return list(rewards_by_session.values()), session_representatives, cluster_indices

    def _compute_group_advantages(
        self,
        cluster_rewards: list[float],
        session_representatives: list[RolloutState],
        cluster_indices: list[int | None],
    ) -> list[float]:
        """Scatter cluster-level advantage estimates back to group members.

        Args:
            cluster_rewards (list[float]): Session-clustered rewards of the group.
            session_representatives (list[RolloutState]): One representative state per session cluster.
            cluster_indices (list[int | None]): Per-state cluster index; ``None`` marks reward-missing states.

        Returns:
            list[float]: Advantage per state, aligned with the group order. Zero when task advantage
            is disabled or the state carries no reward.
        """
        if self.task_adv_weight != 0 and self.advantage_estimator is not None:
            cluster_advantages = self.advantage_estimator.compute(
                torch.tensor(cluster_rewards, dtype=torch.float32), session_representatives
            )
            return [float(cluster_advantages[index].item()) if index is not None else 0.0 for index in cluster_indices]
        return [0.0] * len(cluster_indices)

    @staticmethod
    def _critic_log_scalars(critic_logs: list[WorkerLogItem] | None) -> dict[str, float]:
        """Average rank-0 critic step metrics into ``data_info`` keys."""
        if not critic_logs:
            return {}
        steps = critic_logs[0].get("train_metrics") or []
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        rename = {"reduced_llm_loss": "critic/value_loss", "grad_norm": "critic/grad_norm"}
        for step in steps:
            for key, value in step.items():
                target = rename.get(key)
                if target is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                sums[target] = sums.get(target, 0.0) + float(value)
                counts[target] = counts.get(target, 0) + 1
        return {key: sums[key] / counts[key] for key in sums}

    def _build_trainer_log_info(
        self,
        prepared: _PreparedBatch,
        pack_plan: _PackPlan,
        log_infos: list[WorkerLogItem],
        advantages: Sequence[float | list[float]] | None,
        advantage_stats: dict[str, float] | None = None,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> dict[str, float]:
        """Assemble the complete ``data_info`` step log from one fit's phases.

        All log-related logic of ``fit`` lives here: data statistics of the prepared
        batch, plan-derived packing metrics and the maxima of the per-worker
        conversion/materialize timings.

        Args:
            prepared (_PreparedBatch): Output of ``_prepare_rollout_items``.
            pack_plan (_PackPlan): Output of ``_build_pack_plan``.
            log_infos (list[WorkerLogItem]): Per-worker training logs of the batch.
            advantages: Final scalar or token advantages in global sample order. Ignored when
                ``advantage_stats`` carries worker-local GAE statistics.
            raw_rewards_sum (float): Producer-side raw reward sum used by ``raw_rewards/mean``.
            raw_rewards_count (int): Producer-side raw reward count used by ``raw_rewards/mean``.

        Returns:
            dict[str, float]: The step statistics dictionary.
        """
        advantages_list: list[float] = []
        prompt_len_list: list[float] = []
        response_len_list: list[float] = []
        tool_turns_list: list[int] = []
        training_tokens = 0
        if advantages is not None and len(advantages) != len(prepared["rollout_items"]):
            raise ValueError(
                f"advantage rows {len(advantages)} do not match rollout items {len(prepared['rollout_items'])}"
            )
        for state_index, state in enumerate(prepared["rollout_items"]):
            advantage = advantages[state_index] if advantages is not None else None
            prompt_len_list.append(int(state.extra_fields["train_prompt_length"]))
            response_len_list.append(int(state.extra_fields["supervised_tokens"]))
            turns = state.extra_fields.get("agent_tool_turns")
            if isinstance(turns, int):
                tool_turns_list.append(turns)
            assert state.num_tokens is not None
            training_tokens += state.num_tokens
            if advantage_stats is None:
                # Advantage metrics count one entry per supervised position, matching the
                # per-position advantage layout built by the worker conversion.
                if advantage is None:
                    continue
                if isinstance(advantage, list):
                    labels = state.labels
                    assert labels is not None
                    advantages_list.extend(value for value, label in zip(advantage, labels[1:]) if label != -100)
                else:
                    advantages_list.extend([advantage] * int(state.extra_fields["supervised_tokens"]))

        # rewards/* report the per-session reward distribution; batch_size/training_samples
        # count the rollout segments included in training.
        rewards_t = _stat_tensor(prepared["cluster_rewards"])
        if advantage_stats is None:
            advantages_t = _stat_tensor(advantages_list)
            advantage_stats = {
                "mean": advantages_t.mean().item(),
                "min": advantages_t.min().item(),
                "max": advantages_t.max().item(),
            }
        prompt_len_t = _stat_tensor(prompt_len_list)
        response_len_t = _stat_tensor(response_len_list)
        training_samples = len(prepared["rollout_items"])

        raw_rewards_mean = raw_rewards_sum / raw_rewards_count if raw_rewards_count > 0 else rewards_t.mean().item()
        info_dict: dict[str, float] = {
            "batch_size": training_samples,
            "training_samples": training_samples,
            "training_tokens": training_tokens,
            "rewards/mean": rewards_t.mean().item(),
            "rewards/min": rewards_t.min().item(),
            "rewards/max": rewards_t.max().item(),
            "raw_rewards/mean": raw_rewards_mean,
            "advantages/mean": advantage_stats["mean"],
            "advantages/min": advantage_stats["min"],
            "advantages/max": advantage_stats["max"],
            "response_len/mean": response_len_t.mean().item(),
            "response_len/min": response_len_t.min().item(),
            "response_len/max": response_len_t.max().item(),
            "response_len/std": response_len_t.std().item(),
            "prompt_len/mean": prompt_len_t.mean().item(),
            "prompt_len/min": prompt_len_t.min().item(),
            "prompt_len/max": prompt_len_t.max().item(),
        }
        info_dict.update(self.distillation.reward_scalars(prepared["distillation_reward_observations"]))
        if tool_turns_list:
            tool_turns_t = torch.tensor(tool_turns_list, dtype=torch.float32)
            info_dict["tool_turns/mean"] = tool_turns_t.mean().item()
            info_dict["tool_turns/min"] = float(tool_turns_t.min().item())
            info_dict["tool_turns/max"] = float(tool_turns_t.max().item())
        info_dict.update(pack_plan["plan_log"])
        # WorkerLogItem is a TypedDict; dynamic-key reads below need a plain dict view.
        worker_logs = cast(list[dict[str, Any]], log_infos)
        for key, target_key in (
            ("packing_conversion_time_s", "packing/conversion_time_s_max"),
            ("packing_pack_time_s", "packing/pack_time_s_max"),
        ):
            worker_values = [log_info[key] for log_info in worker_logs if key in log_info]
            if worker_values:
                info_dict[target_key] = max(worker_values)
        return info_dict

    def _prepare_rollout_items(self, rollout_groups: list[list[RolloutState]], rollout_idx: int) -> _PreparedBatch:
        """Flatten samples and retain reward/group mappings without estimating
        advantages or building training tensors.

        Groups arrive pre-filtered: the replay buffer selects by ``Status.COMPLETED``
        and sample alignment is guaranteed by the trainer entry check, so no group-level validation happens here; missing train meta
        fields still fail fast.

        Args:
            rollout_groups (list[list[RolloutState]]): Rollout groups selected for training.
            rollout_idx (int): Global rollout (train step) id written into ``batch_attr``.

        Returns:
            _PreparedBatch: Flat rollout items in training order with original
            reward/group mappings, reward clusters for statistics, and the batch-level
            ``TrainBatchAttr`` derived in the same single pass over the states.
        """
        cluster_rewards_list: list[float] = []
        distillation_reward_observations: list[tuple[RolloutState, float]] = []
        rollout_items: list[RolloutState] = []
        reward_groups: list[_GroupRewardInfo] = []
        use_3d_position_ids = False
        has_rollout_logprobs = False
        has_routed_experts = False
        for group in rollout_groups:
            # Phase 1: session reward clustering keeps one reward per agentic session so
            # rewards/* and advantages are not amplified by per-session segment counts.
            cluster_rewards, session_representatives, cluster_indices = self._cluster_group_rewards(group)
            cluster_rewards_list.extend(cluster_rewards)
            distillation_reward_observations.extend(zip(session_representatives, cluster_rewards))

            reward_groups.append(
                {
                    "rewards": cluster_rewards,
                    "representatives": session_representatives,
                    "cluster_indices": cluster_indices,
                }
            )
            for state in group:
                rollout_items.append(state)
                use_3d_position_ids = use_3d_position_ids or _state_uses_3d_position_ids(state)
                has_rollout_logprobs = has_rollout_logprobs or _state_has_rollout_logprobs(state)
                has_routed_experts = has_routed_experts or _state_has_routed_experts(state)
                # 只读 AgentLoop 规范化时写入的轻量 meta 字段（AgentLoop canonicalization），不触碰
                # input_ids/labels 等重载荷；meta 缺失说明样本未过 canonicalize，直接 fail fast。
                if (
                    state.extra_fields.get("supervised_tokens") is None
                    or state.extra_fields.get("train_prompt_length") is None
                    or state.num_tokens is None
                ):
                    raise ValueError(
                        "Rollout state is missing the train meta fields written by AgentLoop canonicalization: "
                        f"rollout_id={state.rollout_id}"
                    )

        return {
            "rollout_items": rollout_items,
            "reward_groups": reward_groups,
            "batch_attr": {
                "rollout_idx": rollout_idx,
                "use_3d_position_ids": use_3d_position_ids,
                "pack_loss_keys": ["rollout_logprobs"] if has_rollout_logprobs else [],
                "has_routed_experts": has_routed_experts,
            },
            "cluster_rewards": cluster_rewards_list,
            "distillation_reward_observations": distillation_reward_observations,
        }

    def _build_pack_plan(
        self,
        rollout_items: list[RolloutState],
        pack_max_length: int,
        worker_cfg: WorkerConfig,
        data_replicate_size: int,
    ) -> _PackPlan:
        """Build an index-only per-DP-rank plan without training targets.

        Args:
            rollout_items (list[RolloutState]): Flat rollout states in training order.
            pack_max_length (int): Maximum token budget of one pack.
            worker_cfg (WorkerConfig): Training worker config providing packing options.
            data_replicate_size (int): Number of workers holding one data replica.

        Returns:
            _PackPlan: Global sample indices and local pack indices per DP rank,
            plus packing metrics. Payload assembly happens separately, once the
            algorithm has produced its training targets.
        """
        plan_begin = time.perf_counter()
        lengths = []
        real_tokens = 0
        for state in rollout_items:
            assert state.num_tokens is not None, (
                f"Rollout state is missing num_tokens written by AgentLoop canonicalization: rollout_id={state.rollout_id}"
            )
            lengths.append(state.num_tokens)
            real_tokens += state.num_tokens
        data_packer = RLDataPacker(
            pack_max_length=pack_max_length,
            world_size=len(self.workers),
            data_replicate_size=data_replicate_size,
            optimizer_steps=worker_cfg.optimizer_steps,
            pack_strategy=worker_cfg.pack_strategy,
            pack_seed=worker_cfg.pack_seed,
            pack_extra_buffer_size=worker_cfg.pack_extra_buffer_size,
        )
        packed_plan, padding_tokens = data_packer.pack(lengths)

        dp_plans: dict[int, _DPRankPlan] = {}
        for dp_rank, dp_plan in enumerate(packed_plan):
            global_indices = sorted({index for step in dp_plan for pack in step for index in pack})
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(global_indices)}
            dp_plans[dp_rank] = {
                "plan": [[[global_to_local[index] for index in pack] for pack in step] for step in dp_plan],
                "global_indices": global_indices,
            }
        total_packs = sum(len(step) for dp_plan in packed_plan for step in dp_plan)
        total_tokens = real_tokens + padding_tokens
        plan_log = {
            "packing/real_tokens": float(real_tokens),
            "packing/padding_tokens": float(padding_tokens),
            "packing/efficiency": real_tokens / total_tokens if total_tokens > 0 else 0.0,
            "packing/num_packs": float(total_packs),
            "packing/num_optimizer_steps": float(len(packed_plan[0])),
            "packing/plan_time_s": time.perf_counter() - plan_begin,
        }
        return {"dp_plans": dp_plans, "plan_log": plan_log}

    def _estimate_advantages(
        self,
        prepared: _PreparedBatch,
        rollout_items: list[RolloutState],
        pack_plan: _PackPlan,
        value_batches: list[TokenBatchProto] | None,
        data_replicate_size: int,
    ) -> tuple[list[float], list[list[float]] | None, list[list[float]] | None, dict[str, float] | None]:
        """Estimate advantages with the configured mode and return the stats.

        Args:
            prepared (_PreparedBatch): Output of ``_prepare_rollout_items`` holding the
                pre-clustered reward groups consumed by the non-GAE path.
            rollout_items (list[RolloutState]): Flat states in training order.
            pack_plan (_PackPlan): Maps each DP rank's packed samples back to global indices.
            value_batches (list[TokenBatchProto] | None): Critic values gathered by
                ``collect_values``. ``None`` when values stay worker-local (segment GAE)
                or no critic is configured.
            data_replicate_size (int): Workers that share one DP rank.

        Returns:
            sample_advantages (list[float]): One float per rollout item, for ``advantages/*``
                logs. Empty for segment GAE, whose per-sample advantages live worker-side.
            token_advantages (list[list[float]] | None): One row per sample and one value
                per shifted token, passed to actor ``fit``. ``None`` unless session GAE.
            token_returns (list[list[float]] | None): Critic returns with the same layout,
                passed to critic ``fit``. ``None`` unless session GAE.
            advantage_stats (dict[str, float] | None): ``mean``/``min``/``max`` of the
                supervised-token advantages. ``None`` for the non-GAE and session-GAE paths,
                whose stats are derived downstream from the returned advantages.
        """
        if self._gae_reward_scope is None:
            # Group-based scalar advantages (GRPO-style, no critic): one advantage per
            # sample, scattered from each session cluster's estimate.
            sample_advantages: list[float] = []
            for group in prepared["reward_groups"]:
                sample_advantages.extend(
                    self._compute_group_advantages(
                        group["rewards"], group["representatives"], group["cluster_indices"]
                    )
                )
            return sample_advantages, None, None, None
        elif self._gae_reward_scope == "segment":
            # Segment GAE stays worker-local: values were cached by ``collect_values`` and
            # each worker computes per-segment targets in place. Only replica-0 workers
            # contribute to the global normalization statistics returned here.
            from xtuner.v1.rl.advantage.gae import DecoupledGAEEstimator, GAEEstimator

            estimator = self.advantage_estimator
            assert isinstance(estimator, (GAEEstimator, DecoupledGAEEstimator))
            gae_handles = [
                worker.critic_compute_segment_gae.remote(  # type: ignore[attr-defined]
                    estimator=estimator,
                    contribute_to_global_stats=worker_idx % data_replicate_size == 0,
                )
                for worker_idx, worker in enumerate(self.workers)
            ]
            gae_results = [result for result in ray.get(gae_handles, timeout=TRAIN_RAY_GET_TIMEOUT) if result]
            advantage_count = sum(result["count"] for result in gae_results)
            advantage_stats = {
                "mean": sum(result["sum"] for result in gae_results) / advantage_count if advantage_count else 0.0,
                "min": min((result["min"] for result in gae_results), default=0.0),
                "max": max((result["max"] for result in gae_results), default=0.0),
            }
            return [], None, None, advantage_stats
        elif self._gae_reward_scope == "session":
            # Session GAE groups tokens across samples of one session, so critic values must
            # be gathered on the controller. Scatter each rank's local values back to global
            # sample order before running the estimator.
            assert value_batches is not None
            from xtuner.v1.rl.advantage.gae import GAEEstimator

            estimator = self.advantage_estimator
            assert isinstance(estimator, GAEEstimator)
            dense_values: list[list[float] | None] = [None] * len(rollout_items)
            for dp_rank, dispatch in pack_plan["dp_plans"].items():
                local_values = [row.tolist() for row in value_batches[dp_rank * data_replicate_size].chunk()]
                sample_indices = dispatch["global_indices"]
                if len(local_values) != len(sample_indices):
                    raise ValueError(
                        f"Critic values for dp_rank={dp_rank} do not match the packed samples: "
                        f"{len(local_values)} vs {len(sample_indices)}"
                    )
                for local_index, global_index in enumerate(sample_indices):
                    dense_values[global_index] = local_values[local_index]
            if any(sample_values is None for sample_values in dense_values):
                raise RuntimeError("Critic values do not cover every training sample")
            token_advantages, token_returns = estimator.compute_session_gae(
                rollout_items,
                cast(list[list[float]], dense_values),
            )
            # Sample-level log values: the mean of each sample's supervised-token advantages.
            sample_means: list[float] = []
            for state, row in zip(rollout_items, token_advantages):
                labels = state.labels or []
                mask = [label != -100 for label in labels[1:]]
                if len(mask) != len(row):
                    raise ValueError(f"shifted label length {len(mask)} does not match advantage length {len(row)}")
                supervised = [advantage for advantage, keep in zip(row, mask) if keep]
                sample_means.append(sum(supervised) / len(supervised) if supervised else 0.0)
            return sample_means, token_advantages, token_returns, None
        else:
            raise ValueError(f"Unknown GAE reward scope: {self._gae_reward_scope}")

    def fit(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        rollout_idx: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[WorkerLogItem], dict[str, float]]:
        timers: dict[str, float] = {}
        has_critic = self._has_critic
        if self._gae_reward_scope is not None and not has_critic:
            raise RuntimeError("GAE requires a configured critic role")
        if has_critic and self._gae_reward_scope is None:
            raise RuntimeError("Critic training requires a GAE advantage estimator")

        prepared = self._prepare_rollout_items(rollout_groups, rollout_idx)
        rollout_items = prepared["rollout_items"]
        if not rollout_items:
            raise ValueError(f"Rollout {rollout_idx} has no valid training samples")

        worker_cfg = ray.get(self.workers[0].get_worker_cfg.remote())  # type: ignore[attr-defined]
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]

        pack_plan = self._build_pack_plan(rollout_items, pack_max_length, worker_cfg, data_replicate_size)

        # Pixel-value object refs must stay alive until every worker has consumed the
        # dispatch, so the finally clause releases them on both the success and the
        # failure path.
        handles: list[ray.ObjectRef] = []
        sample_advantages: list[float] = []
        token_advantages: list[list[float]] | None = None
        token_returns: list[list[float]] | None = None
        advantage_stats: dict[str, float] | None = None
        segment_gae = self._gae_reward_scope == "segment"
        try:
            # Nested in a list so Ray does not dereference it before the worker
            # explicitly fetches it. Replicas of one DP rank share the ref.
            item_refs_by_dp = {
                dp_rank: [ray.put([rollout_items[index] for index in dispatch["global_indices"]])]
                for dp_rank, dispatch in pack_plan["dp_plans"].items()
            }
            # 1. Each worker fetches its rank's rollout states and pack plan once.
            # Later critic and actor calls read that cache instead of passing again.
            load_handles = []
            for worker_idx, worker in enumerate(self.workers):
                dp_rank = worker_idx // data_replicate_size
                dp_dispatch = pack_plan["dp_plans"][dp_rank]
                load_handles.append(
                    worker.load_train_batch.remote(  # type: ignore[attr-defined]
                        rollout_item_refs=item_refs_by_dp[dp_rank],
                        pack_plan=dp_dispatch["plan"],
                        batch_attr=prepared["batch_attr"],
                    )
                )
            ray.get(load_handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            critic_log_infos: list[WorkerLogItem] | None = None
            value_batches: list[TokenBatchProto | None] | None = None
            # 2. Optional. Critic value forward. Segment GAE keeps values worker-local;
            # session GAE returns them to the controller for cross-sequence grouping.
            if has_critic:
                ray.get(
                    [worker.switch_role.remote("critic") for worker in self.workers],  # type: ignore[attr-defined]
                    timeout=TRAIN_RAY_GET_TIMEOUT,
                )
                value_handles = [
                    worker.critic_collect_values.remote(return_values=not segment_gae)  # type: ignore[attr-defined]
                    for worker in self.workers
                ]
                with timer("critic_collect_values", timers):
                    value_batches = ray.get(value_handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            # 2.5. Estimate advantages with the configured mode: group scalars without a
            # critic, worker-local segment GAE, or controller-side session GAE.
            sample_advantages, token_advantages, token_returns, advantage_stats = self._estimate_advantages(
                prepared,
                rollout_items,
                pack_plan,
                cast(list[TokenBatchProto], value_batches) if value_batches is not None else None,
                data_replicate_size,
            )
            # 3. Optional. Critic fit. Each worker trains the value head on its samples' token returns.
            if has_critic:
                critic_handles = []
                for worker_idx, worker in enumerate(self.workers):
                    if segment_gae:
                        critic_handles.append(
                            worker.critic_fit.remote()  # type: ignore[attr-defined]
                        )
                    else:
                        assert token_returns is not None
                        dp_rank = worker_idx // data_replicate_size
                        local_indices = pack_plan["dp_plans"][dp_rank]["global_indices"]
                        critic_handles.append(
                            worker.critic_fit.remote(  # type: ignore[attr-defined]
                                token_returns=TokenBatchProto.concat(
                                    [
                                        torch.tensor(token_returns[index], dtype=torch.float32)
                                        for index in local_indices
                                    ]
                                ),
                            )
                        )
                with timer("critic_fit", timers):
                    critic_log_infos = ray.get(critic_handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            # 4. Actor fit. Each worker trains the policy on its samples'
            # advantages; the advantages are passed as actor_fit arguments and
            # staged into the worker's sample cache on arrival (fit() then
            # reads them from the cache).
            ray.get(
                [worker.switch_role.remote("actor") for worker in self.workers],  # type: ignore[attr-defined]
                timeout=TRAIN_RAY_GET_TIMEOUT,
            )

            for worker_idx, worker in enumerate(self.workers):
                dp_rank = worker_idx // data_replicate_size
                indices = pack_plan["dp_plans"][dp_rank]["global_indices"]
                if segment_gae:
                    handles.append(worker.actor_fit.remote())  # type: ignore[attr-defined]
                    continue
                elif token_advantages is None:
                    advantages: list[float] | TokenBatchProto = [sample_advantages[index] for index in indices]
                else:
                    advantages = TokenBatchProto.concat(
                        [torch.tensor(token_advantages[index], dtype=torch.float32) for index in indices]
                    )
                handles.append(worker.actor_fit.remote(advantages=advantages))  # type: ignore[attr-defined]
            with timer("actor_fit", timers):
                log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            # No role restore here: a fit failure aborts the whole run (the
            # trainer has no retry path), so nothing consumes the restored
            # residency — process exit reclaims the GPU. Restore (and re-protect
            # it) only if a skip-step/retry policy is ever introduced.
            # Critic-stage failures skip actor fit, so release the worker-side cache here.
            try:
                ray.get(
                    [worker.release_train_batch.remote() for worker in self.workers],  # type: ignore[attr-defined]
                    timeout=TRAIN_RAY_GET_TIMEOUT,
                )
            except (RuntimeError, ray.exceptions.RayError) as exc:
                self.logger.warning(f"Failed to release cached training batch: {exc}")
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for state in rollout_items:
                pixel_values = state.mm_info.get("pixel_values") if state.mm_info is not None else None
                if pixel_values is not None:
                    if isinstance(pixel_values, ray.ObjectRef):
                        free_pixel_value_refs.append(pixel_values)  # 单 ref：直接收集
                    else:
                        free_pixel_value_refs.extend(r for r in pixel_values if isinstance(r, ray.ObjectRef))
            if free_pixel_value_refs:
                free_object_refs(free_pixel_value_refs)
        del rollout_items

        data_info = self._build_trainer_log_info(
            prepared,
            pack_plan,
            log_infos,
            None if advantage_stats is not None else (token_advantages or sample_advantages),
            advantage_stats=advantage_stats,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
        data_info.update(self._critic_log_scalars(critic_log_infos))
        data_info.update({f"time/{name}": value for name, value in timers.items()})
        return log_infos, data_info

    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def bind_rollout_weight_update(
        self,
        *,
        targets,
        rollout_config,
    ):
        ray.get(
            [
                worker.bind_rollout_weight_update.remote(
                    targets=targets,
                    rollout_config=rollout_config,
                )
                for worker in self.workers
            ]
        )

    def weight_update(self, **kwargs):
        """Update the weights from the training workers."""
        handles = [worker.weight_update.remote(**kwargs) for worker in self.workers]
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def has_registered_weight_checkpoint(self) -> bool:
        handles = [worker.has_registered_weight_checkpoint.remote() for worker in self.workers]
        return all(ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT))

    def suspend_train_nccl_process_groups(self):
        """Suspend train-side NCCL process groups after weight sync."""
        handles = [
            worker.suspend_train_nccl_process_groups.remote()  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        results = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        self.logger.info(f"Suspended train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def resume_train_nccl_process_groups(self):
        """Resume train-side NCCL process groups before training."""
        handles = [
            worker.resume_train_nccl_process_groups.remote()  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        results = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        self.logger.info(f"Resumed train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        handles = [worker.resume.remote(load_checkpoint_cfg) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return
