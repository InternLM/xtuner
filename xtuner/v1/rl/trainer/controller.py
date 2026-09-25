from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger

from .pack import DPRankPackIndices, RLDataPacker
from .worker import TrainBatchAttr, TrainingWorker, WorkerConfig, WorkerLogItem


if TYPE_CHECKING:
    from xtuner.v1.rl.advantage.base import AdvantageEstimator

TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class _DPRankDispatch(TypedDict):
    """Per-DP-rank payload of one training batch, before the Ray upload in
    ``fit``."""

    plan: DPRankPackIndices
    advantages: list[float]
    rollout_items: list[RolloutState]


class _PackPlan(TypedDict):
    """Controller-side packing result of one fit batch.

    dp_dispatches (dict[int, _DPRankDispatch]): One dispatch payload per DP rank.
    plan_log (dict[str, float]): Plan-derived ``packing/*`` metrics, including the
        wall time of ``_build_pack_plan`` itself.
    """

    dp_dispatches: dict[int, _DPRankDispatch]
    plan_log: dict[str, float]


class _PreparedBatch(TypedDict):
    """Outputs of ``_prepare_rollout_items`` consumed by the fit phases."""

    rollout_items: list[RolloutState]
    advantages: list[float]
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
    ) -> None:
        self.workers = workers
        self.advantage_estimator = advantage_estimator
        self.distillation = distillation if distillation is not None else DistillationTrainerAdapter(None)
        self.task_adv_weight = self.distillation.task_adv_weight
        self.logger = get_logger()

    def _cluster_group_rewards(
        self, group: list[RolloutState]
    ) -> tuple[list[float], list[RolloutState], list[int | None]]:
        # 按 session 聚类奖励：agentic session 可能拆成多个共享同一 reward 的可训练 segment，
        # 每个 session 只计一次，避免 rewards/* 与 advantage 被 segment 数放大。
        # session_id 只由 agentic loop / XTUNER_DETERMINISTIC 写入；普通 RL 回退到 rollout_id。
        rewards_by_session: dict[Any, float] = {}
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
                rewards_by_session[session_key] = reward
                session_representatives.append(state)
            cluster_indices.append(len(session_representatives) - 1)
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

    def _build_trainer_log_info(
        self,
        prepared: _PreparedBatch,
        pack_plan: _PackPlan,
        log_infos: list[WorkerLogItem],
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
        for state, advantage in zip(prepared["rollout_items"], prepared["advantages"]):
            # Advantage metrics count one entry per supervised position, matching the
            # per-position advantage layout built by the worker conversion.
            supervised_positions = int(state.extra_fields["supervised_tokens"])
            advantages_list.extend([advantage] * supervised_positions)
            prompt_len_list.append(int(state.extra_fields["train_prompt_length"]))
            response_len_list.append(supervised_positions)
            turns = state.extra_fields.get("agent_tool_turns")
            if isinstance(turns, int):
                tool_turns_list.append(turns)
            assert state.num_tokens is not None
            training_tokens += state.num_tokens

        # rewards/* report the per-session reward distribution; batch_size/training_samples
        # count the rollout segments included in training.
        rewards_t = _stat_tensor(prepared["cluster_rewards"])
        advantages_t = _stat_tensor(advantages_list)
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
            "advantages/mean": advantages_t.mean().item(),
            "advantages/min": advantages_t.min().item(),
            "advantages/max": advantages_t.max().item(),
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
        """Cluster rewards, estimate advantages and validate the train meta
        fields without building training tensors.

        Groups arrive pre-filtered: the replay buffer selects by ``Status.COMPLETED``
        and sample alignment is guaranteed by ``canonicalize_train_fields`` on the
        rollout side, so no group-level validation happens here; missing train meta
        fields still fail fast.

        Args:
            rollout_groups (list[list[RolloutState]]): Rollout groups selected for training.
            rollout_idx (int): Global rollout (train step) id written into ``batch_attr``.

        Returns:
            _PreparedBatch: Flat rollout items in training order with aligned
            advantages, reward clusters for the step statistics, and the batch-level
            ``TrainBatchAttr`` derived in the same single pass over the states.
        """
        cluster_rewards_list: list[float] = []
        distillation_reward_observations: list[tuple[RolloutState, float]] = []
        rollout_items: list[RolloutState] = []
        advantages: list[float] = []
        use_3d_position_ids = False
        has_rollout_logprobs = False
        has_routed_experts = False
        for group in rollout_groups:
            # Phase 1: session reward clustering keeps one reward per agentic session so
            # rewards/* and advantages are not amplified by per-session segment counts.
            cluster_rewards, session_representatives, cluster_indices = self._cluster_group_rewards(group)
            cluster_rewards_list.extend(cluster_rewards)
            distillation_reward_observations.extend(zip(session_representatives, cluster_rewards))

            # Phase 2: group-level advantage estimation scattered back to members.
            sample_advantages = self._compute_group_advantages(
                cluster_rewards, session_representatives, cluster_indices
            )
            for state, advantage in zip(group, sample_advantages):
                rollout_items.append(state)
                advantages.append(advantage)
                use_3d_position_ids = use_3d_position_ids or _state_uses_3d_position_ids(state)
                has_rollout_logprobs = has_rollout_logprobs or _state_has_rollout_logprobs(state)
                has_routed_experts = has_routed_experts or _state_has_routed_experts(state)
                # 只读 AgentLoop 规范化时写入的轻量 meta 字段（write_train_meta），不触碰
                # input_ids/labels 等重载荷；meta 缺失说明样本未过 canonicalize，直接 fail fast。
                if (
                    state.extra_fields.get("supervised_tokens") is None
                    or state.extra_fields.get("train_prompt_length") is None
                    or state.num_tokens is None
                ):
                    raise ValueError(
                        "Rollout state is missing the train meta fields written by write_train_meta: "
                        f"rollout_id={state.rollout_id}"
                    )

        return {
            "rollout_items": rollout_items,
            "advantages": advantages,
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
        advantages: list[float],
        pack_max_length: int,
        worker_cfg: WorkerConfig,
        data_replicate_size: int,
    ) -> _PackPlan:
        """Build the per-DP-rank pack plan and dispatch payloads without
        uploading.

        Args:
            rollout_items (list[RolloutState]): Flat rollout states in training order.
            advantages (list[float]): Scalar advantage per rollout state, aligned with
                ``rollout_items``.
            pack_max_length (int): Maximum token budget of one pack.
            worker_cfg (WorkerConfig): Training worker config providing packing options.
            data_replicate_size (int): Number of workers holding one data replica.

        Returns:
            _PackPlan: One dispatch payload per DP rank — local plan, local advantages
            and this rank's rollout states — plus the plan-derived ``packing/*``
            metrics including the wall time of this call. Uploading the payloads to
            the Ray object store happens in ``fit``.
        """
        plan_begin = time.perf_counter()
        lengths = []
        real_tokens = 0
        for state in rollout_items:
            assert state.num_tokens is not None, (
                f"Rollout state is missing num_tokens written by write_train_meta: rollout_id={state.rollout_id}"
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
        )
        packed_plan, padding_tokens = data_packer.pack(lengths)

        dp_dispatches: dict[int, _DPRankDispatch] = {}
        for dp_rank, dp_plan in enumerate(packed_plan):
            global_indices = sorted({index for step in dp_plan for pack in step for index in pack})
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(global_indices)}
            dp_dispatches[dp_rank] = {
                "plan": [[[global_to_local[index] for index in pack] for pack in step] for step in dp_plan],
                "advantages": [advantages[index] for index in global_indices],
                "rollout_items": [rollout_items[index] for index in global_indices],
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
        return {"dp_dispatches": dp_dispatches, "plan_log": plan_log}

    def fit(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        rollout_idx: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[WorkerLogItem], dict[str, float]]:
        prepared = self._prepare_rollout_items(rollout_groups, rollout_idx)
        rollout_items = prepared["rollout_items"]
        if not rollout_items:
            raise ValueError(f"Rollout {rollout_idx} has no valid training samples")

        worker_cfg = ray.get(self.workers[0].get_worker_cfg.remote())  # type: ignore[attr-defined]
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]

        pack_plan = self._build_pack_plan(
            rollout_items, prepared["advantages"], pack_max_length, worker_cfg, data_replicate_size
        )

        # Pixel-value object refs must stay alive until every worker has consumed the
        # dispatch, so the finally clause releases them on both the success and the
        # failure path.
        handles: list[ray.ObjectRef] = []
        try:
            for worker_idx, worker in enumerate(self.workers):
                dp_rank = worker_idx // data_replicate_size
                dp_dispatch = pack_plan["dp_dispatches"][dp_rank]
                # Nested in a list so Ray does not dereference it before TrainingWorker.fit
                # explicitly fetches it; one DP rank's replicas share the ref.
                item_refs = [ray.put(dp_dispatch["rollout_items"])]
                handles.append(
                    worker.fit.remote(  # type: ignore[attr-defined]
                        rollout_item_refs=item_refs,
                        advantages=dp_dispatch["advantages"],
                        pack_plan=dp_dispatch["plan"],
                        batch_attr=prepared["batch_attr"],
                    )
                )
            log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for state in rollout_items:
                pixel_values = state.mm_info.get("pixel_values") if state.mm_info is not None else None
                if pixel_values is not None:
                    free_pixel_value_refs.extend(pixel_values)
            if free_pixel_value_refs:
                free_object_refs(free_pixel_value_refs)
        del rollout_items

        data_info = self._build_trainer_log_info(
            prepared,
            pack_plan,
            log_infos,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
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
