from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Literal, Sequence, TypedDict, cast

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.advantage.base import AdvantageEstimator, TokenAdvantageEstimator
from xtuner.v1.rl.advantage.gae import session_token_targets
from xtuner.v1.rl.advantage.stats import TokenStats
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.profile import timer

from .pack import PackPlan, RLDataPacker, TokenBatchProto
from .worker import TrainBatchAttr, TrainingWorker, WorkerConfig, WorkerLogItem


if TYPE_CHECKING:
    from xtuner.v1.rl.advantage.base import AdvantageEstimator

TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


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


class TrainingController:
    def __init__(
        self,
        workers: list[TrainingWorker],
        advantage_estimator: AdvantageEstimator | TokenAdvantageEstimator | None = None,
        distillation: DistillationTrainerAdapter | None = None,
        has_critic: bool = False,
    ) -> None:
        self.workers = workers
        self.advantage_estimator = advantage_estimator
        self.distillation = distillation if distillation is not None else DistillationTrainerAdapter(None)
        self.task_adv_weight = self.distillation.task_adv_weight
        self.logger = get_logger()
        # Token estimators and the critic are configured together; scalar
        # estimators drive group-based training without a critic.
        self._token_advantages = isinstance(advantage_estimator, TokenAdvantageEstimator)
        if self._token_advantages != has_critic:
            raise ValueError("Token estimator and critic must be configured together")
        self._has_critic = has_critic

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
            estimator = self.advantage_estimator
            # Group-scalar training is the only caller of this path.
            assert not isinstance(estimator, TokenAdvantageEstimator)
            cluster_advantages = estimator.compute(
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

    def _prepare_group_targets(self, prepared: _PreparedBatch, pack_plan: PackPlan) -> list[float]:
        """Estimate group-scalar advantages and install them per DP rank.

        Args:
            prepared (_PreparedBatch): Output of ``_prepare_rollout_items``.
            pack_plan (PackPlan): Controller-side packing plan.

        Returns:
            list[float]: One scalar advantage per rollout item in training order.
        """
        advantages: list[float] = []
        for group in prepared["reward_groups"]:
            advantages.extend(
                self._compute_group_advantages(group["rewards"], group["representatives"], group["cluster_indices"])
            )
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        handles = []
        for worker_index, worker in enumerate(self.workers):
            rank = pack_plan.ranks[worker_index // data_replicate_size]
            handles.append(
                worker.install_advantages.remote(  # type: ignore[attr-defined]
                    [advantages[index] for index in rank.global_indices]
                )
            )
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return advantages

    def _build_trainer_log_info(
        self,
        prepared: _PreparedBatch,
        pack_plan: PackPlan,
        log_infos: list[WorkerLogItem],
        advantages: Sequence[float | list[float]] | None,
        token_stats: TokenStats | None = None,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> dict[str, float]:
        """Assemble the complete ``data_info`` step log from one fit's phases.

        All log-related logic of ``fit`` lives here: data statistics of the prepared
        batch, plan-derived packing metrics and the maxima of the per-worker
        conversion/materialize timings.

        Args:
            prepared (_PreparedBatch): Output of ``_prepare_rollout_items``.
            pack_plan (PackPlan): Output of ``_build_pack_plan``.
            log_infos (list[WorkerLogItem]): Per-worker training logs of the batch.
            advantages: Final scalar/token advantages in global sample order. Ignored
                when ``token_stats`` carries PPO action-token statistics.
            token_stats (TokenStats | None): Normalized PPO advantage statistics; when set,
                ``advantages/*`` use the effective action-token口径.
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
            meta = state.train_meta
            assert meta is not None
            prompt_len_list.append(int(meta.prompt_length))
            response_len_list.append(int(meta.supervised_tokens))
            turns = state.extra_fields.get("agent_tool_turns")
            if isinstance(turns, int):
                tool_turns_list.append(turns)
            assert state.num_tokens is not None
            training_tokens += state.num_tokens
            if token_stats is None:
                # Advantage metrics count one entry per supervised position, matching the
                # per-position advantage layout built by the worker conversion.
                if advantage is None:
                    continue
                if isinstance(advantage, list):
                    labels = state.labels
                    assert labels is not None
                    advantages_list.extend(value for value, label in zip(advantage, labels[1:]) if label != -100)
                else:
                    advantages_list.extend([advantage] * meta.supervised_tokens)

        # rewards/* report the per-session reward distribution; batch_size/training_samples
        # count the rollout segments included in training.
        rewards_t = _stat_tensor(prepared["cluster_rewards"])
        advantage_log: dict[str, float]
        if token_stats is not None:
            advantage_log = {
                "advantages/mean": token_stats.mean,
                "advantages/min": token_stats.min if token_stats.min is not None else 0.0,
                "advantages/max": token_stats.max if token_stats.max is not None else 0.0,
                "advantages/sum": token_stats.sum,
                "advantages/count": float(token_stats.count),
            }
        else:
            advantages_t = _stat_tensor(advantages_list)
            advantage_log = {
                "advantages/mean": advantages_t.mean().item(),
                "advantages/min": advantages_t.min().item(),
                "advantages/max": advantages_t.max().item(),
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
            **advantage_log,
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
        info_dict.update(pack_plan.metrics)
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
                meta = state.train_meta
                if meta is None or state.num_tokens is None:
                    raise ValueError(
                        "Rollout state is missing the train meta written by RolloutState.set_training_fields: "
                        f"rollout_id={state.rollout_id}"
                    )
                use_3d_position_ids = use_3d_position_ids or meta.position_layout == "mrope_3d"
                has_rollout_logprobs = has_rollout_logprobs or meta.has_rollout_logprobs
                has_routed_experts = has_routed_experts or meta.has_routed_experts

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
    ) -> PackPlan:
        """Build an index-only per-DP-rank plan without training targets.

        Args:
            rollout_items (list[RolloutState]): Flat rollout states in training order.
            pack_max_length (int): Maximum token budget of one pack.
            worker_cfg (WorkerConfig): Training worker config providing packing options.
            data_replicate_size (int): Number of workers holding one data replica.

        Returns:
            PackPlan: Global sample indices and local pack indices per DP rank,
            plus packing metrics. Payload assembly happens separately, once the
            algorithm has produced its training targets.
        """
        lengths = []
        for state in rollout_items:
            assert state.num_tokens is not None, (
                f"Rollout state is missing num_tokens written by RolloutState.set_training_fields: "
                f"rollout_id={state.rollout_id}"
            )
            lengths.append(state.num_tokens)
        data_packer = RLDataPacker(
            pack_max_length=pack_max_length,
            world_size=len(self.workers),
            data_replicate_size=data_replicate_size,
            optimizer_steps=worker_cfg.optimizer_steps,
            pack_strategy=worker_cfg.pack_strategy,
            pack_seed=worker_cfg.pack_seed,
            pack_extra_buffer_size=worker_cfg.pack_extra_buffer_size,
        )
        return data_packer.plan(lengths)

    def _prepare_ppo_targets(
        self,
        prepared: _PreparedBatch,
        pack_plan: PackPlan,
        data_replicate_size: int,
        timers: dict[str, float],
    ) -> TokenStats:
        """Collect critic values and install complete PPO targets on every
        worker.

        Postcondition: every worker's batch owns frozen ``old_values``,
        ``advantages`` and ``returns``. Segment GAE computes targets per worker;
        session GAE gathers representative values, restores global sample order
        and installs one target pair per DP rank.

        Args:
            prepared (_PreparedBatch): Output of ``_prepare_rollout_items``.
            pack_plan (PackPlan): Controller-side packing plan.
            data_replicate_size (int): Number of workers holding one data replica.
            timers (dict[str, float]): Step timing accumulator.

        Returns:
            TokenStats: Normalized effective-action-token statistics.
        """
        estimator = cast(TokenAdvantageEstimator, self.advantage_estimator)
        session = estimator.reward_scope == "session"
        is_representative = [worker_index % data_replicate_size == 0 for worker_index in range(len(self.workers))]

        with timer("critic_collect_values", timers):
            value_results = ray.get(
                [
                    worker.critic_collect_values.remote(return_values=session and is_representative[worker_index])  # type: ignore[attr-defined]
                    for worker_index, worker in enumerate(self.workers)
                ],
                timeout=TRAIN_RAY_GET_TIMEOUT,
            )

        if not session:
            stats = ray.get(
                [
                    worker.critic_compute_segment_targets.remote(  # type: ignore[attr-defined]
                        estimator, contribute=is_representative[worker_index]
                    )
                    for worker_index, worker in enumerate(self.workers)
                ],
                timeout=TRAIN_RAY_GET_TIMEOUT,
            )
            return TokenStats.merge(cast(list[TokenStats], stats))

        rows_by_dp = [
            value_results[worker_index] for worker_index in range(len(self.workers)) if is_representative[worker_index]
        ]
        values = pack_plan.restore_rows(cast(list[TokenBatchProto], rows_by_dp))
        advantages, returns = session_token_targets(estimator, prepared["rollout_items"], values)
        targets_by_dp = [
            ray.put(
                (
                    TokenBatchProto.concat([advantages[index] for index in rank.global_indices]),
                    TokenBatchProto.concat([returns[index] for index in rank.global_indices]),
                )
            )
            for rank in pack_plan.ranks
        ]
        ray.get(
            [
                worker.install_ppo_targets.remote(targets_by_dp[worker_index // data_replicate_size])  # type: ignore[attr-defined]
                for worker_index, worker in enumerate(self.workers)
            ],
            timeout=TRAIN_RAY_GET_TIMEOUT,
        )
        return TokenStats.merge(
            [
                TokenStats.from_masked(advantage, torch.tensor(state.labels or [])[1:].ne(-100))
                for state, advantage in zip(prepared["rollout_items"], advantages)
            ]
        )

    @contextmanager
    def _loaded_batch(
        self,
        rollout_items: list[RolloutState],
        pack_plan: PackPlan,
        batch_attr: TrainBatchAttr,
        data_replicate_size: int,
    ) -> Iterator[None]:
        """Load one local batch per worker and release it on every exit path.

        Remote pixel/routed refs stay owned by the trainer and are only released
        after a successful ``fit``.

        Args:
            rollout_items (list[RolloutState]): Flat states in training order.
            pack_plan (PackPlan): Controller-side packing plan.
            batch_attr (TrainBatchAttr): Batch-level attributes shared by workers.
            data_replicate_size (int): Number of workers holding one data replica.

        Yields:
            None
        """
        refs_by_dp: list[list[ray.ObjectRef]] = []
        try:
            # Nested in a list so Ray does not dereference it before the worker
            # fetches it; replicas of one DP rank share the ref.
            refs_by_dp = [
                [ray.put([rollout_items[index] for index in rank.global_indices])] for rank in pack_plan.ranks
            ]
            load_handles = []
            for worker_idx, worker in enumerate(self.workers):
                dp_rank = worker_idx // data_replicate_size
                load_handles.append(
                    worker.load_train_batch.remote(  # type: ignore[attr-defined]
                        refs_by_dp[dp_rank],
                        pack_plan.ranks[dp_rank],
                        batch_attr,
                    )
                )
            ray.get(load_handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            yield
        finally:
            original_error = sys.exception()
            try:
                ray.get(
                    [worker.release_train_batch.remote() for worker in self.workers],  # type: ignore[attr-defined]
                    timeout=TRAIN_RAY_GET_TIMEOUT,
                )
            except (RuntimeError, ray.exceptions.RayError) as exc:
                self.logger.warning(f"Failed to release cached training batch: {exc}")
                if original_error is None:
                    raise
            finally:
                refs_by_dp.clear()

    def fit(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        rollout_idx: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[WorkerLogItem], dict[str, float]]:
        timers: dict[str, float] = {}
        prepared = self._prepare_rollout_items(rollout_groups, rollout_idx)
        rollout_items = prepared["rollout_items"]
        if not rollout_items:
            raise ValueError(f"Rollout {rollout_idx} has no valid training samples")

        worker_cfg = ray.get(self.workers[0].get_worker_cfg.remote())  # type: ignore[attr-defined]
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]

        pack_plan = self._build_pack_plan(rollout_items, pack_max_length, worker_cfg, data_replicate_size)

        sample_advantages: list[float] | None = None
        token_stats: TokenStats | None = None
        critic_log_infos: list[WorkerLogItem] | None = None
        with self._loaded_batch(rollout_items, pack_plan, prepared["batch_attr"], data_replicate_size):
            if self._has_critic:
                ray.get(
                    [worker.set_residency.remote("critic", "critic") for worker in self.workers],  # type: ignore[attr-defined]
                    timeout=TRAIN_RAY_GET_TIMEOUT,
                )
                token_stats = self._prepare_ppo_targets(prepared, pack_plan, data_replicate_size, timers)
                with timer("critic_fit", timers):
                    critic_log_infos = ray.get(
                        [worker.critic_fit.remote() for worker in self.workers],  # type: ignore[attr-defined]
                        timeout=TRAIN_RAY_GET_TIMEOUT,
                    )
            else:
                sample_advantages = self._prepare_group_targets(prepared, pack_plan)

            ray.get(
                [worker.set_residency.remote("actor", "actor") for worker in self.workers],  # type: ignore[attr-defined]
                timeout=TRAIN_RAY_GET_TIMEOUT,
            )
            with timer("actor_fit", timers):
                log_infos = ray.get(
                    [worker.actor_fit.remote() for worker in self.workers],  # type: ignore[attr-defined]
                    timeout=TRAIN_RAY_GET_TIMEOUT,
                )
        del rollout_items

        data_info = self._build_trainer_log_info(
            prepared,
            pack_plan,
            log_infos,
            sample_advantages,
            token_stats=token_stats,
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
