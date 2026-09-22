from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, is_valid_for_training
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .pack import DPRankPackIndices, PackedDataIndices, RLDataPacker
from .worker import TrainingWorker, WorkerConfig, WorkerLogItem


if TYPE_CHECKING:
    from xtuner.v1.rl.advantage.base import AdvantageEstimator

TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class _SampleObservation(TypedDict):
    """Per-sample metrics observed while converting one rollout state."""

    advantage: float
    supervised_positions: int
    prompt_len: int
    tool_turns: int | None
    training_tokens: int


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

    def _summarize_data_info(
        self,
        observations: list[_SampleObservation],
        cluster_rewards_list: list[float],
        distillation_reward_observations: list[tuple[RolloutState, float]],
        training_samples: int,
        raw_rewards_sum: float,
        raw_rewards_count: int,
    ) -> dict[str, float]:
        """Reduce phase-1/phase-3 observations into the step ``data_info``
        metrics.

        Args:
            observations (list[_SampleObservation]): Per-sample metrics collected in phase 3.
            cluster_rewards_list (list[float]): Session-clustered rewards collected in phase 1.
            distillation_reward_observations (list[tuple[RolloutState, float]]): Session
                representative states paired with their cluster reward.
            training_samples (int): Number of states in the valid groups selected for training.
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
        for observation in observations:
            # Advantage metrics count one entry per supervised position, matching the
            # per-position advantage layout built in phase 3.
            advantages_list.extend([observation["advantage"]] * observation["supervised_positions"])
            prompt_len_list.append(observation["prompt_len"])
            response_len_list.append(observation["supervised_positions"])
            if observation["tool_turns"] is not None:
                tool_turns_list.append(observation["tool_turns"])
            training_tokens += observation["training_tokens"]

        # rewards/* report the per-session reward distribution; batch_size/training_samples
        # count the valid rollout segments included in training.
        rewards_t = _stat_tensor(cluster_rewards_list)
        advantages_t = _stat_tensor(advantages_list)
        prompt_len_t = _stat_tensor(prompt_len_list)
        response_len_t = _stat_tensor(response_len_list)

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
        info_dict.update(self.distillation.reward_scalars(distillation_reward_observations))
        if tool_turns_list:
            tool_turns_t = torch.tensor(tool_turns_list, dtype=torch.float32)
            info_dict["tool_turns/mean"] = tool_turns_t.mean().item()
            info_dict["tool_turns/min"] = float(tool_turns_t.min().item())
            info_dict["tool_turns/max"] = float(tool_turns_t.max().item())
        return info_dict

    def _prepare_rollout_items(
        self,
        rollout_groups: list[list[RolloutState]],
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[RolloutState], list[float], dict[str, float], bool]:
        """Validate groups, cluster rewards, estimate advantages and summarize
        step statistics without building training tensors.

        Args:
            rollout_groups (list[list[RolloutState]]): Rollout groups selected for training.
            raw_rewards_sum (float): Producer-side raw reward sum used by ``raw_rewards/mean``.
            raw_rewards_count (int): Producer-side raw reward count used by ``raw_rewards/mean``.

        Returns:
            tuple[list[RolloutState], list[float], dict[str, float], bool]: Flat rollout
            items in final training order, per-sample scalar advantages aligned with the
            items, the step statistics dictionary, and whether the batch uniformly uses
            3D MRoPE position ids.
        """
        use_3d_position_ids = any(_state_uses_3d_position_ids(state) for group in rollout_groups for state in group)

        cluster_rewards_list: list[float] = []
        distillation_reward_observations: list[tuple[RolloutState, float]] = []
        observations: list[_SampleObservation] = []
        training_samples = 0
        rollout_items: list[RolloutState] = []
        advantages: list[float] = []
        for group in rollout_groups:
            if not is_valid_for_training(group, self.logger):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue
            training_samples += len(group)

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
                observations.append(self._observe_rollout_state(state, advantage))

        if not XTUNER_DETERMINISTIC:
            paired_items = list(zip(rollout_items, advantages))
            random.shuffle(paired_items)
            rollout_items = [item for item, _ in paired_items]
            advantages = [advantage for _, advantage in paired_items]

        info_dict = self._summarize_data_info(
            observations,
            cluster_rewards_list,
            distillation_reward_observations,
            training_samples,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
        return rollout_items, advantages, info_dict, use_3d_position_ids

    def _observe_rollout_state(self, state: RolloutState, advantage: float) -> _SampleObservation:
        # 只读 AgentLoop 规范化时写入的轻量统计字段（write_train_meta），保证下一阶段
        # metadata 替换 RolloutState 时本函数无需修改。
        supervised_tokens = state.extra_fields.get("supervised_tokens")
        prompt_length = state.extra_fields.get("train_prompt_length")
        num_tokens = state.num_tokens
        if supervised_tokens is None or prompt_length is None or num_tokens is None:
            raise ValueError(
                "Rollout state is missing the train meta fields written by write_train_meta: "
                f"rollout_id={state.rollout_id}"
            )
        turns = state.extra_fields.get("agent_tool_turns")
        return {
            "advantage": advantage,
            "supervised_positions": int(supervised_tokens),
            "prompt_len": int(prompt_length),
            "tool_turns": turns if isinstance(turns, int) else None,
            "training_tokens": int(num_tokens),
        }

    def _build_pack_plan(
        self,
        rollout_items: list[RolloutState],
        pack_max_length: int,
        worker_cfg: WorkerConfig,
        data_replicate_size: int,
    ) -> tuple[PackedDataIndices, int]:
        lengths = []
        for state in rollout_items:
            assert state.num_tokens is not None, (
                f"Rollout state is missing num_tokens written by write_train_meta: rollout_id={state.rollout_id}"
            )
            lengths.append(state.num_tokens)
        data_packer = RLDataPacker(
            pack_max_length=pack_max_length,
            world_size=len(self.workers),
            data_replicate_size=data_replicate_size,
            optimizer_steps=worker_cfg.optimizer_steps,
            pack_strategy=worker_cfg.pack_strategy,
            pack_seed=worker_cfg.pack_seed,
        )
        return data_packer.pack(lengths)

    def _build_dp_rank_inputs(
        self,
        rollout_items: list[RolloutState],
        advantages: list[float],
        packed_indices: PackedDataIndices,
    ) -> tuple[dict[int, list[ray.ObjectRef]], dict[int, list[float]], dict[int, DPRankPackIndices]]:
        rollout_item_refs: dict[int, list[ray.ObjectRef]] = {}
        local_advantages: dict[int, list[float]] = {}
        local_pack_plans: dict[int, DPRankPackIndices] = {}
        for dp_rank, dp_plan in enumerate(packed_indices):
            global_indices = sorted({index for step in dp_plan for pack in step for index in pack})
            global_to_local = {global_index: local_index for local_index, global_index in enumerate(global_indices)}
            local_pack_plans[dp_rank] = [
                [[global_to_local[index] for index in pack] for pack in step] for step in dp_plan
            ]
            local_advantages[dp_rank] = [advantages[index] for index in global_indices]
            # Keep the ObjectRef nested in a list so Ray does not dereference it before
            # TrainingWorker.fit explicitly fetches it; one DP rank's replicas share it.
            rollout_item_refs[dp_rank] = [ray.put([rollout_items[index] for index in global_indices])]
        return rollout_item_refs, local_advantages, local_pack_plans

    def fit(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        rollout_idx: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[WorkerLogItem], dict[str, float]]:
        rollout_items, advantages, data_info, use_3d_position_ids = self._prepare_rollout_items(
            rollout_groups,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
        if not rollout_items:
            raise ValueError(f"Rollout {rollout_idx} has no valid training samples")

        worker_cfg = ray.get(self.workers[0].get_worker_cfg.remote())  # type: ignore[attr-defined]
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]

        plan_begin = time.perf_counter()
        packed_plan, padding_tokens = self._build_pack_plan(
            rollout_items, pack_max_length, worker_cfg, data_replicate_size
        )
        data_info["packing/plan_time_s"] = time.perf_counter() - plan_begin

        real_tokens = 0
        for state in rollout_items:
            assert state.num_tokens is not None
            real_tokens += state.num_tokens
        total_packs = sum(len(step) for dp_plan in packed_plan for step in dp_plan)
        total_tokens = real_tokens + padding_tokens
        data_info["packing/real_tokens"] = float(real_tokens)
        data_info["packing/padding_tokens"] = float(padding_tokens)
        data_info["packing/efficiency"] = real_tokens / total_tokens if total_tokens > 0 else 0.0
        data_info["packing/num_packs"] = float(total_packs)
        data_info["packing/num_optimizer_steps"] = float(len(packed_plan[0]))

        # 批级 presence 位只跨 controller 边界传递模板信息；token 级字段留在 per-DP refs 里。
        pack_loss_keys = (
            ["rollout_logprobs"] if any(_state_has_rollout_logprobs(state) for state in rollout_items) else []
        )
        has_routed_experts = any(_state_has_routed_experts(state) for state in rollout_items)

        rollout_item_refs, local_advantages, local_pack_plans = self._build_dp_rank_inputs(
            rollout_items, advantages, packed_plan
        )
        del packed_plan, advantages

        handles = []
        for worker_idx, worker in enumerate(self.workers):
            dp_rank = worker_idx // data_replicate_size
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    rollout_item_refs=rollout_item_refs[dp_rank],
                    advantages=local_advantages[dp_rank],
                    pack_plan=local_pack_plans[dp_rank],
                    use_3d_position_ids=use_3d_position_ids,
                    pack_loss_keys=pack_loss_keys,
                    has_routed_experts=has_routed_experts,
                    rollout_idx=rollout_idx,
                )
            )
        try:
            log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            # free pixel values ref（ownership 不变：controller 在所有 worker 消费完后释放）
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for state in rollout_items:
                pixel_values = state.mm_info.get("pixel_values") if state.mm_info is not None else None
                if pixel_values is not None:
                    free_pixel_value_refs.extend(pixel_values)
            if len(free_pixel_value_refs) > 0:
                free_object_refs(free_pixel_value_refs)
            del rollout_item_refs, local_advantages, local_pack_plans, rollout_items

        for key, target_key in (
            ("packing_conversion_time_s", "packing/conversion_time_s_max"),
            ("packing_materialize_time_s", "packing/materialize_time_s_max"),
        ):
            worker_values = [log_info[key] for log_info in log_infos if key in log_info]
            if worker_values:
                data_info[target_key] = max(worker_values)
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
