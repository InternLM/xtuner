import math
import os
from typing import Any, Literal, TypedDict

import ray
import torch
from typing_extensions import NotRequired

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger

from .worker import TrainingWorker, WorkerLogItem


TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: float
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


class TrainingWorkerGroup:
    """Role-scoped wrapper around one set of training worker handles.

    The first migration stage only registers the actor group. ``attached`` is
    reserved for critic/reference/teacher groups in a later stage.
    """

    def __init__(
        self,
        name: str,
        workers: list[Any],
        *,
        attached: bool = False,
        can_sync_rollout: bool = False,
    ) -> None:
        """Create a role-scoped view over training worker handles.

        Args:
            name: Logical role name, for example ``"actor"``. The name is
                also used by :class:`TrainingController` for dispatch.
            workers: Ray actor handles that belong to this role. All handles
                are expected to represent the same data-parallel worker set.
            attached: Whether this role is hosted in the actor process. It is
                reserved for the future critic/reference/teacher roles and is
                not active in the actor-only migration stage.
            can_sync_rollout: Whether this group is allowed to push weights to
                the rollout workers. This must be true only for the actor.
        """
        if not name:
            raise ValueError("worker group name must not be empty")
        self.name = name
        self.workers = workers
        self.attached = attached
        self.can_sync_rollout = can_sync_rollout

    @staticmethod
    def _call(worker: Any, method: str, *args, **kwargs):
        call = getattr(worker, method)
        remote = getattr(call, "remote", None)
        return remote(*args, **kwargs) if remote is not None else call(*args, **kwargs)

    def fit(
        self,
        packed_data_batches: list[PackedBatch],
        data_replicate_size: int,
        rollout_idx: int,
    ) -> list[WorkerLogItem]:
        """Dispatch already-packed data using the existing DP assignment."""
        if not self.workers:
            return []
        if data_replicate_size <= 0 or len(self.workers) % data_replicate_size != 0:
            raise ValueError("worker count must be divisible by data_replicate_size")

        dp_size = len(self.workers) // data_replicate_size
        data_batch_refs: dict[int, ray.ObjectRef] = {}
        handles = []
        for worker_idx, worker in enumerate(self.workers):
            dp_idx = worker_idx // data_replicate_size
            if dp_idx not in data_batch_refs:
                data_batch_refs[dp_idx] = ray.put(packed_data_batches[dp_idx::dp_size])
            handles.append(
                self._call(
                    worker,
                    "fit",
                    data_batches=data_batch_refs[dp_idx],
                    rollout_idx=rollout_idx,
                )
            )
        try:
            return ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        finally:
            data_batch_refs.clear()

    def broadcast_host(self, method: str, *args, **kwargs):
        """Broadcast a worker lifecycle method and resolve its results."""
        handles = [self._call(worker, method, *args, **kwargs) for worker in self.workers]
        if handles and isinstance(handles[0], ray.ObjectRef):
            return ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return handles


class TrainingController:
    def __init__(self, workers: list[TrainingWorker] | None = None) -> None:
        self.workers = workers or []
        self.logger = get_logger()
        self._groups: dict[str, TrainingWorkerGroup] = {}
        self.register(TrainingWorkerGroup("actor", self.workers, can_sync_rollout=True))

    def register(self, group: TrainingWorkerGroup) -> None:
        if group.name in self._groups:
            raise ValueError(f"duplicate worker group: {group.name}")
        self._groups[group.name] = group

    def group(self, name: str) -> TrainingWorkerGroup:
        try:
            return self._groups[name]
        except KeyError as exc:
            raise KeyError(f"worker group is not registered: {name}") from exc

    def switch_role_modules(self, role: str) -> None:
        """Reserve the role-switching boundary for later attached roles."""
        if role != "actor":
            raise KeyError(f"role is not available in actor-only stage: {role}")
        # The current TrainingWorker has only actor modules, so selecting the
        # actor role is already the steady state and requires no remote call.

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

    def fit(self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int) -> list[WorkerLogItem]:
        """Run the actor-only training path for one batch."""
        return self.train_grpo_batch(data_batches, pack_max_length, rollout_idx)

    def train_grpo_batch(
        self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int
    ) -> list[WorkerLogItem]:
        """Keep the existing packing path and route its result to actor."""
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

        try:
            log_infos = self.group("actor").fit(
                packed_data_batches,
                data_replicate_size=data_replicate_size,
                rollout_idx=rollout_idx,
            )
        finally:
            # free pixel values ref
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for data in packed_data_batches:
                if data["seq_ctx"].pixel_values is not None:
                    free_pixel_value_refs.extend(data["seq_ctx"].pixel_values)
            if len(free_pixel_value_refs) > 0:
                free_object_refs(free_pixel_value_refs)
            del packed_data_batches
        return log_infos

    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            self.group("actor").broadcast_host("offload_model")
        elif target == "optimizer":
            self.group("actor").broadcast_host("offload_optimizer")
        elif target == "all":
            self.group("actor").broadcast_host("offload_model")
            self.group("actor").broadcast_host("offload_optimizer")
        return

    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            self.group("actor").broadcast_host("onload_model")
        elif target == "optimizer":
            self.group("actor").broadcast_host("onload_optimizer")
        elif target == "all":
            self.group("actor").broadcast_host("onload_model")
            self.group("actor").broadcast_host("onload_optimizer")
        return

    def bind_rollout_weight_update(
        self,
        *,
        targets,
        rollout_config,
    ):
        self.group("actor").broadcast_host(
            "bind_rollout_weight_update",
            targets=targets,
            rollout_config=rollout_config,
        )

    def weight_update(self, **kwargs):
        """Update rollout weights from the actor group only."""
        actor = self.group("actor")
        if not actor.can_sync_rollout:
            raise RuntimeError("actor group is not configured for rollout sync")
        actor.broadcast_host("weight_update", **kwargs)

    def has_registered_weight_checkpoint(self) -> bool:
        return all(self.group("actor").broadcast_host("has_registered_weight_checkpoint"))

    def suspend_train_nccl_process_groups(self):
        """Suspend train-side NCCL process groups after weight sync."""
        results = self.group("actor").broadcast_host("suspend_train_nccl_process_groups")
        self.logger.info(f"Suspended train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def resume_train_nccl_process_groups(self):
        """Resume train-side NCCL process groups before training."""
        results = self.group("actor").broadcast_host("resume_train_nccl_process_groups")
        self.logger.info(f"Resumed train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        self.group("actor").broadcast_host("save_hf", hf_dir, save_dtype)
        return

    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        self.group("actor").broadcast_host("resume", load_checkpoint_cfg)
        return

    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        self.group("actor").broadcast_host("save", dcp_dir, no_save_optimizer)
        return
