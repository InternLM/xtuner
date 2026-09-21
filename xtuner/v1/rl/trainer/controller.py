from __future__ import annotations

import math
import os
import random
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
import ray
import torch
from typing_extensions import NotRequired

from xtuner.v1.data_proto.rl_data import RolloutState, is_valid_for_training
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .worker import TrainingWorker, WorkerLogItem


if TYPE_CHECKING:
    from xtuner.v1.rl.advantage.base import AdvantageEstimator

TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: list[float]
    rollout_logprobs: torch.Tensor | None
    teacher_logprobs: NotRequired[torch.Tensor | None]
    target_token_ids: NotRequired[torch.Tensor | None]
    teacher_indices: NotRequired[torch.Tensor | None]


class PackedBatch(TypedDict):
    """Output of ``TrainingController._packing``: samples concatenated into one
    sequence."""

    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantages: torch.Tensor
    rollout_logprobs: torch.Tensor | None
    teacher_logprobs: torch.Tensor | None
    target_token_ids: torch.Tensor | None
    teacher_indices: torch.Tensor | None


class _SampleObservation(TypedDict):
    """Per-sample metrics observed while converting one rollout state."""

    advantage: float
    supervised_positions: int
    prompt_len: int
    tool_turns: int | None
    training_tokens: int


def get_train_seq_ctx(
    input_ids: torch.LongTensor,
    position_ids: np.ndarray | None = None,
    multimodal_train_info: dict | None = None,
) -> SequenceContext:
    """Build a CPU ``SequenceContext`` for one training sample.

    Args:
        input_ids (torch.LongTensor): Model input tokens with shape ``(1, seq_len)``.
        position_ids (np.ndarray | None): Optional position ids. A 3D array triggers
            the VLM MRoPE layout; a prompt-only position segment is extended to cover
            the whole input.
        multimodal_train_info (dict | None): Optional multimodal payload with
            ``pixel_values``, ``image_grid_thw`` and ``num_img_tokens``.

    Returns:
        SequenceContext: The CPU sequence context of the sample.
    """
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    position_ids = _to_cpu_tensor(position_ids, dtype=torch.long)
    if position_ids is not None and len(position_ids.shape) == 3:
        # Match get_rope_index_3: response text continues from a single global
        # max(T, H, W), not per-axis maxima. Per-axis max diverges when the
        # prompt ends on image tokens (T≈0 while H/W are large). The extension
        # length is derived from the position deficit so samples whose positions
        # already cover the whole input (full-sequence agentic samples) need no
        # extra shape information.
        num_extend = input_ids.size(-1) - position_ids.size(-1)
        if num_extend > 0:
            max_value = position_ids.amax()
            response_position_ids = (
                (torch.arange(1, num_extend + 1, device=position_ids.device, dtype=position_ids.dtype) + max_value)
                .view(1, 1, -1)
                .expand(3, 1, -1)
            )
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        seq_ctx.position_ids = position_ids  # type: ignore[assignment]
        assert position_ids.size(-1) == input_ids.size(-1)

    if multimodal_train_info:
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = _to_cpu_tensor(multimodal_train_info.get("image_grid_thw"), dtype=torch.long)
        num_img_tokens = multimodal_train_info.get("num_img_tokens")
        if num_img_tokens is not None:
            seq_ctx.num_img_tokens = [num_img_tokens]
    return seq_ctx


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


def _verify_packed_alignment(packed_batch: PackedBatch) -> None:
    # Position-wise fields must stay aligned with input_ids; a mismatch would silently shift
    # the element-wise policy loss or crash on shape mismatch in the loss function.
    assert packed_batch["seq_ctx"].input_ids is not None
    seq_len = packed_batch["seq_ctx"].input_ids.shape[1]
    shifted_labels = packed_batch["shifted_labels"]
    advantages = packed_batch["advantages"]
    assert shifted_labels.shape[1] == seq_len, f"{shifted_labels.shape[1]} vs {seq_len}"
    assert advantages.shape[1] == seq_len, f"{advantages.shape[1]} vs {seq_len}"
    for field_name, optional_field in (
        ("rollout_logprobs", packed_batch["rollout_logprobs"]),
        ("teacher_logprobs", packed_batch["teacher_logprobs"]),
        ("target_token_ids", packed_batch["target_token_ids"]),
        ("teacher_indices", packed_batch["teacher_indices"]),
    ):
        if optional_field is not None:
            assert optional_field.shape[1] == seq_len, f"{field_name}: {optional_field.shape[1]} vs {seq_len}"


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

    # TODO(hha): 这个逻辑不够通用，应该复用 sft 函数，从而支持 expand soft pack
    def _get_pack_infos(self, dataset, num_tokens, target, random=None):
        inds = list(range(len(dataset)))
        if random is not None:
            random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        longest = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= target:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        "indices": item_buffer,
                        "longest": int(longest),
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                "indices": item_buffer,
                "longest": int(longest),
            }

            pack_infos.append(info)

        return pack_infos

    # TODO(hha): 这个逻辑不够通用，和模型绑定了
    def _packing(self, data_batches, pack_max_length, language_cfg):
        pack_infos = self._get_pack_infos(
            data_batches,
            [data["seq_ctx"].input_ids.numel() for data in data_batches],
            pack_max_length,
        )
        packed_data_batches = []

        is_qwen3_vl = any(data["seq_ctx"].position_ids.ndim == 3 for data in data_batches)

        has_rollout_routed_experts = False
        if data_batches[0]["seq_ctx"].rollout_routed_experts is not None:
            assert language_cfg is not None
            has_rollout_routed_experts = True
            n_routed_experts = language_cfg.n_routed_experts

        for pack_info in pack_infos:
            indices = pack_info["indices"]
            total_len = sum([data_batches[i]["seq_ctx"].input_ids.shape[1] for i in indices])
            pad_len = pack_max_length - total_len
            seq_ctx_list = [data_batches[i]["seq_ctx"] for i in indices]
            label_list = [data_batches[i]["shifted_labels"] for i in indices]
            advantage_list = [data_batches[i]["advantage"] for i in indices]

            rollout_logprobs_list = None
            if "rollout_logprobs" in data_batches[0] and data_batches[0]["rollout_logprobs"] is not None:
                rollout_logprobs_list = [data_batches[i]["rollout_logprobs"] for i in indices]

            teacher_logprobs_list = None
            if "teacher_logprobs" in data_batches[0] and data_batches[0]["teacher_logprobs"] is not None:
                teacher_logprobs_list = [data_batches[i]["teacher_logprobs"] for i in indices]

            target_token_ids_list = None
            if "target_token_ids" in data_batches[0] and data_batches[0]["target_token_ids"] is not None:
                target_token_ids_list = [data_batches[i]["target_token_ids"] for i in indices]

            teacher_indices_list = None
            if "teacher_indices" in data_batches[0] and data_batches[0]["teacher_indices"] is not None:
                teacher_indices_list = [data_batches[i]["teacher_indices"] for i in indices]

            if pad_len > 0:
                # Reduce the attn calculation time by using multiple short sequence packs
                pad_tokens = tuple(
                    torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                    for _ in range(pad_len // 1024)
                )
                if pad_len % 1024 > 0:
                    pad_tokens = pad_tokens + (
                        torch.zeros(1, pad_len % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"),
                    )
                pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
                pad_seq_ctx.num_padding = pad_len
                pad_labels = torch.full(
                    (1, pad_len),
                    -100,
                    dtype=data_batches[0]["shifted_labels"].dtype,
                    device=data_batches[0]["shifted_labels"].device,
                )
                pad_advantages = [-100] * pad_len
                if is_qwen3_vl:
                    _position_ids_list = []
                    for pad_token in pad_tokens:
                        _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                        _position_ids_list.append(_position_ids)
                    pad_seq_ctx.position_ids = torch.cat(_position_ids_list, dim=-1)

                if has_rollout_routed_experts:
                    pad_rand_index = torch.randint(low=0, high=n_routed_experts, size=(pad_len, 1, 1))
                    pad_seq_ctx.rollout_routed_experts = pad_rand_index

                seq_ctx_list.append(pad_seq_ctx)
                label_list.append(pad_labels)
                advantage_list.append(pad_advantages)
                if rollout_logprobs_list is not None:
                    pad_rollout_logprobs = torch.zeros(
                        1,
                        pad_len,
                        dtype=data_batches[0]["rollout_logprobs"].dtype,
                        device=data_batches[0]["shifted_labels"].device,
                    )
                    rollout_logprobs_list.append(pad_rollout_logprobs)
                if teacher_logprobs_list is not None:
                    teacher_logprobs_shape = data_batches[0]["teacher_logprobs"].shape[2:]
                    pad_teacher_logprobs = torch.zeros(
                        (1, pad_len, *teacher_logprobs_shape),
                        dtype=data_batches[0]["teacher_logprobs"].dtype,
                        device=data_batches[0]["shifted_labels"].device,
                    )
                    teacher_logprobs_list.append(pad_teacher_logprobs)
                if target_token_ids_list is not None:
                    target_token_ids_shape = data_batches[0]["target_token_ids"].shape[2:]
                    pad_target_token_ids = torch.zeros(
                        (1, pad_len, *target_token_ids_shape),
                        dtype=data_batches[0]["target_token_ids"].dtype,
                        device=data_batches[0]["shifted_labels"].device,
                    )
                    target_token_ids_list.append(pad_target_token_ids)
                if teacher_indices_list is not None:
                    pad_teacher_indices = torch.full(
                        (1, pad_len),
                        -1,
                        dtype=data_batches[0]["teacher_indices"].dtype,
                        device=data_batches[0]["teacher_indices"].device,
                    )
                    teacher_indices_list.append(pad_teacher_indices)

            seq_ctx = SequenceContext.cat(seq_ctx_list)
            shifted_labels = torch.cat(label_list, dim=1)  # (1, max_len)
            advantage_flat = [item for sublist in advantage_list for item in sublist]
            advantages = torch.tensor(advantage_flat, dtype=torch.float32).unsqueeze(0)

            rollout_logprobs = None
            if rollout_logprobs_list is not None:
                rollout_logprobs = torch.cat(rollout_logprobs_list, dim=1)  # (1, max_len)

            teacher_logprobs = None
            if teacher_logprobs_list is not None:
                teacher_logprobs = torch.cat(teacher_logprobs_list, dim=1)  # (1, max_len)

            target_token_ids = None
            if target_token_ids_list is not None:
                target_token_ids = torch.cat(target_token_ids_list, dim=1)

            teacher_indices = None
            if teacher_indices_list is not None:
                teacher_indices = torch.cat(teacher_indices_list, dim=1)  # (1, max_len)

            packed_batch = {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
                "advantages": advantages,
                "rollout_logprobs": rollout_logprobs,
                "teacher_logprobs": teacher_logprobs,
                "target_token_ids": target_token_ids,
                "teacher_indices": teacher_indices,
            }
            _verify_packed_alignment(packed_batch)
            packed_data_batches.append(packed_batch)
        return packed_data_batches

    def _grouped_by_max_length(self, packed_data_batches):
        # sort 过后可能第一个 batch 会有很多 pad tokens，因为最后一个 pack 可能只有少量真实数据。
        # 比如组成了 16 个 pack，第 16 个 pack 可能只有几条真实数据，剩下的都是 pad tokens。
        # 排序后这条 pack 会被放在最前面，导致 rank0 的第一个 step 消耗的有效 token 数往往少于其他 rank，是正常现象。
        return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

    def _rollout_groups_to_colate_items(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[ColateItem], dict[str, float]]:
        """Convert rollout groups into packable training items and step
        statistics.

        Samples arrive canonicalized from the agent loop (``input_ids``/``labels``/
        ``logprobs`` share one full-sequence length) with token staleness already baked
        into ``labels``. The conversion runs in phases: (1) cluster group rewards by
        session, (2) estimate group-level advantages, (3) shift and tensorize each
        rollout state into a ``ColateItem``, and (4) summarize the ``data_info``
        metrics.

        Args:
            rollout_groups (list[list[RolloutState]]): Rollout groups selected for training.
            pack_max_length (int): Maximum token length per sample accepted for packing.
            raw_rewards_sum (float): Producer-side raw reward sum used by ``raw_rewards/mean``.
            raw_rewards_count (int): Producer-side raw reward count used by ``raw_rewards/mean``.

        Returns:
            tuple[list[ColateItem], dict[str, float]]: Packable training items and the step
            statistics dictionary.
        """
        has_3d_position = any(
            state.position_ids is not None and state.position_ids.ndim == 3
            for group in rollout_groups
            for state in group
        )

        cluster_rewards_list: list[float] = []
        distillation_reward_observations: list[tuple[RolloutState, float]] = []
        observations: list[_SampleObservation] = []
        training_samples = 0

        data_batches: list[ColateItem] = []
        for group in rollout_groups:
            if not is_valid_for_training(group, self.logger):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue
            training_samples += len(group)

            # Phase 1: session reward clustering. Collect rewards independently from
            # task-advantage computation: pure OPD may omit rewards entirely; when rewards
            # are present they remain useful observability signals.
            cluster_rewards, session_representatives, cluster_indices = self._cluster_group_rewards(group)
            cluster_rewards_list.extend(cluster_rewards)
            distillation_reward_observations.extend(zip(session_representatives, cluster_rewards))

            # Phase 2: group-level advantage estimation.
            sample_advantages = self._compute_group_advantages(
                cluster_rewards, session_representatives, cluster_indices
            )

            # Phase 3: per-sample shift, tensorization and seq_ctx construction.
            for i, state in enumerate(group):
                data_batch, observation = self._convert_rollout_state(state, sample_advantages[i], has_3d_position)
                assert observation["training_tokens"] <= pack_max_length, (
                    f"{observation['training_tokens']} vs {pack_max_length}"
                )
                data_batches.append(data_batch)
                observations.append(observation)

        if not XTUNER_DETERMINISTIC:
            random.shuffle(data_batches)

        # Phase 4: step statistics summarization.
        info_dict = self._summarize_data_info(
            observations,
            cluster_rewards_list,
            distillation_reward_observations,
            training_samples,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
        return data_batches, info_dict

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

    def _convert_rollout_state(
        self,
        state: RolloutState,
        advantage_val: float,
        has_3d_position: bool,
    ) -> tuple[ColateItem, _SampleObservation]:
        """Shift, tensorize and build the sequence context of one rollout
        state.

        Args:
            state (RolloutState): Canonicalized rollout state with full-sequence ``input_ids``/``labels``.
            advantage_val (float): Group-level advantage broadcast to every supervised position.
            has_3d_position (bool): Whether any sample in the batch carries 3D MRoPE position ids.

        Returns:
            tuple[ColateItem, _SampleObservation]: The packable training item and the per-sample
            metrics observed during conversion.
        """
        input_ids = state.input_ids
        labels = state.labels
        assert input_ids is not None and labels is not None and len(input_ids) == len(labels), (
            f"Rollout state is not canonicalized for training: {state}"
        )
        input_len = len(input_ids)
        shifted_labels = labels[1:]

        input_ids_t = cast(torch.LongTensor, torch.tensor(input_ids[:-1], dtype=torch.int64).unsqueeze(0))
        shifted_labels_t = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

        rollout_logprobs: torch.Tensor | None = None
        if state.logprobs is not None:
            assert len(state.logprobs) == input_len, f"{len(state.logprobs)} vs {input_len}, data: {state}"
            rollout_logprobs = torch.tensor(state.logprobs[1:], dtype=torch.float32).unsqueeze(0)
            assert rollout_logprobs.size() == shifted_labels_t.size(), (
                f"{rollout_logprobs.size()} vs {shifted_labels_t.size()}"
            )

        # Keep the advantage layout aligned with input_ids (response excludes EOS).
        # Prompt positions predict prompt tokens whose shifted labels are -100, so their
        # advantage is 0; the last entry of response_ids (EOS) is only a label, never an input.
        actual_advantages = [0.0 if label == -100 else advantage_val for label in shifted_labels]

        # prompt_len 统计原始输入 prompt 长度（VLM 的 prompt 在 train_prompt_ids）；
        # response_len 统计 LLM 生成的 token 数（labels 非 -100 的监督位）。环境/工具
        # 插入的语义洞不计入两者；prompt_ids 缺失时 prompt_len 退回洞计数。
        prompt_ids = state.extra_fields.get("train_prompt_ids") or state.prompt_ids
        prompt_len = len(prompt_ids) if prompt_ids else sum(label == -100 for label in shifted_labels)
        response_len = sum(label != -100 for label in shifted_labels)

        position_ids = state.position_ids
        if has_3d_position and (position_ids is None or position_ids.ndim != 3):
            seq_len = input_ids_t.size(-1)
            text_position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, 1, -1)
            # Mixed batches use three-axis MRoPE position IDs. Repeat the normal text
            # positions on every axis so text samples have the same rank as the VLM
            # samples when their sequence contexts are packed together.
            position_ids = np.broadcast_to(text_position_ids, (3, 1, seq_len)).copy()
        multimodal_train_info = cast(dict | None, state.mm_info)
        seq_ctx = get_train_seq_ctx(input_ids_t, position_ids, multimodal_train_info)
        seq_ctx.rollout_routed_experts = state.routed_experts  # type: ignore[assignment] # n,layer*expert

        teacher_fields = self.distillation.rollout_teacher_targets(state, shifted_labels=shifted_labels)

        data_dict: ColateItem = {
            "seq_ctx": seq_ctx,
            "shifted_labels": shifted_labels_t,
            "advantage": actual_advantages,
            "rollout_logprobs": rollout_logprobs,
        }
        data_dict.update(cast(ColateItem, teacher_fields))

        # 有可能有重复，但是没有其他更好办法
        turns = state.extra_fields.get("agent_tool_turns")
        observation: _SampleObservation = {
            "advantage": advantage_val,
            "supervised_positions": response_len,
            "prompt_len": prompt_len,
            "tool_turns": turns if isinstance(turns, int) else None,
            "training_tokens": input_len - 1,
        }
        return data_dict, observation

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

    def fit(
        self,
        rollout_groups: list[list[RolloutState]],
        pack_max_length: int,
        rollout_idx: int,
        raw_rewards_sum: float = 0.0,
        raw_rewards_count: int = 0,
    ) -> tuple[list[WorkerLogItem], dict[str, float]]:
        data_batches, data_info = self._rollout_groups_to_colate_items(
            rollout_groups,
            pack_max_length,
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
        )
        has_rollout_routed_experts = False
        language_cfg = None
        if data_batches[0]["seq_ctx"].rollout_routed_experts is not None:
            model_cfg = ray.get(self.workers[0].get_model_cfg.remote())  # type: ignore[attr-defined]
            has_rollout_routed_experts = True
            language_cfg = model_cfg
            if isinstance(model_cfg, BaseComposeConfig):
                language_cfg = model_cfg.text_config

        packed_data_batches = self._packing(data_batches, pack_max_length, language_cfg)
        # packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        # TODO(hha): 这个逻辑不够通用，和模型绑定了
        is_qwen3_vl = False
        if len(packed_data_batches[0]["seq_ctx"].position_ids.shape) == 3:
            is_qwen3_vl = True

        # todo: support round up
        num_packed_data_batches = len(packed_data_batches)
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
        dp_size = len(self.workers) // data_replicate_size
        pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
        if pad_num > 0:
            # Reduce the attn calculation time by using multiple short sequence packs
            assert data_batches[0]["seq_ctx"].input_ids is not None
            pad_tokens = tuple(
                torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
                for _ in range(pack_max_length // 1024)
            )
            if pack_max_length % 1024 > 0:
                assert data_batches[0]["seq_ctx"].input_ids is not None
                pad_tokens = pad_tokens + (
                    torch.zeros(
                        1, pack_max_length % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"
                    ),
                )
            pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore
            pad_seq_ctx.num_padding = pack_max_length
            if is_qwen3_vl:
                _position_ids_list = []
                for pad_token in pad_tokens:
                    _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                    _position_ids_list.append(_position_ids)
                pad_seq_ctx.position_ids = torch.cat(_position_ids_list, dim=-1)  # type: ignore

            pad_shifted_labels = torch.full(
                (1, pack_max_length),
                -100,
                dtype=packed_data_batches[0]["shifted_labels"].dtype,
                device="cpu",
            )
            pad_advantages = torch.full(
                (1, pack_max_length),
                -100,
                dtype=packed_data_batches[0]["advantages"].dtype,
                device="cpu",
            )

            if has_rollout_routed_experts:
                pad_rand_index = torch.randint(
                    low=0,
                    high=1,
                    size=(1, 1, 1),  # add dummy data, true data will be initialized in train worker.fit
                )
                pad_seq_ctx.rollout_routed_experts = pad_rand_index

            pad_rollout_logprobs = None
            if "rollout_logprobs" in packed_data_batches[0] and packed_data_batches[0]["rollout_logprobs"] is not None:
                pad_rollout_logprobs = torch.zeros(
                    1, pack_max_length, dtype=packed_data_batches[0]["rollout_logprobs"].dtype, device="cpu"
                )
            pad_teacher_logprobs = None
            if "teacher_logprobs" in packed_data_batches[0] and packed_data_batches[0]["teacher_logprobs"] is not None:
                teacher_logprobs_shape = packed_data_batches[0]["teacher_logprobs"].shape[2:]
                pad_teacher_logprobs = torch.zeros(
                    (1, pack_max_length, *teacher_logprobs_shape),
                    dtype=packed_data_batches[0]["teacher_logprobs"].dtype,
                    device="cpu",
                )
            pad_target_token_ids = None
            if "target_token_ids" in packed_data_batches[0] and packed_data_batches[0]["target_token_ids"] is not None:
                target_token_ids_shape = packed_data_batches[0]["target_token_ids"].shape[2:]
                pad_target_token_ids = torch.zeros(
                    (1, pack_max_length, *target_token_ids_shape),
                    dtype=packed_data_batches[0]["target_token_ids"].dtype,
                    device="cpu",
                )
            pad_teacher_indices = None
            if "teacher_indices" in packed_data_batches[0] and packed_data_batches[0]["teacher_indices"] is not None:
                pad_teacher_indices = torch.full(
                    (1, pack_max_length),
                    -1,
                    dtype=packed_data_batches[0]["teacher_indices"].dtype,
                    device="cpu",
                )
            pad_data = {
                "seq_ctx": pad_seq_ctx,
                "shifted_labels": pad_shifted_labels,
                "advantages": pad_advantages,
                "rollout_logprobs": pad_rollout_logprobs,
                "teacher_logprobs": pad_teacher_logprobs,
                "target_token_ids": pad_target_token_ids,
                "teacher_indices": pad_teacher_indices,
            }
            pad_data_samples = [pad_data for _ in range(pad_num)]
            packed_data_batches = packed_data_batches + pad_data_samples

        handles = []
        data_batch_refs = {}
        for worker_idx, worker in enumerate(self.workers):
            dp_idx = worker_idx // data_replicate_size
            if dp_idx not in data_batch_refs:
                data_batch_refs[dp_idx] = ray.put(packed_data_batches[dp_idx::dp_size])
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=data_batch_refs[dp_idx],
                    rollout_idx=rollout_idx,
                )
            )
        try:
            log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            # free pixel values ref
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for data in packed_data_batches:
                if data["seq_ctx"].pixel_values is not None:
                    free_pixel_value_refs.extend(data["seq_ctx"].pixel_values)
            if len(free_pixel_value_refs) > 0:
                free_object_refs(free_pixel_value_refs)
            del data_batch_refs
            del packed_data_batches
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


def _to_cpu_tensor(value: np.ndarray | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
    return torch.as_tensor(value, dtype=dtype, device="cpu")
