import math
import random
from typing import Literal, TypedDict, cast

import numpy as np
import ray
import torch
from ray.actor import ActorProxy

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.utils import get_seqlen_balanced_partitions
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger, ray_method

from .worker import TrainingWorker


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: float
    rollout_logprobs: torch.Tensor | None


class RawTrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers
        refs = [
            self.workers[0].get_model_cfg.remote(),
            self.workers[0].get_worker_cfg.remote(),
            self.workers[0].get_data_replicate_size.remote(),
        ]
        self.model_cfg, self.worker_cfg, self.data_replicate_size = ray.get(refs)

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

        is_qwen3_vl = False
        if len(data_batches[0]["seq_ctx"].position_ids.shape) == 3:
            is_qwen3_vl = True

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
                advantage_list.extend(
                    [-100] * math.ceil(pad_len / 1024)
                )  # can be any number, pad tokens are excluded from the calculation of the loss function.

                if rollout_logprobs_list is not None:
                    pad_rollout_logprobs = torch.zeros(
                        1,
                        pad_len,
                        dtype=data_batches[0]["rollout_logprobs"].dtype,
                        device=data_batches[0]["shifted_labels"].device,
                    )
                    rollout_logprobs_list.append(pad_rollout_logprobs)

            seq_ctx = SequenceContext.pack(seq_ctx_list)
            shifted_labels = torch.cat(label_list, dim=1)  # (1, max_len)
            advantages = torch.tensor(advantage_list).float().unsqueeze(0)  # (1, num_samples)
            cu_seq_lens_q = seq_ctx.cu_seq_lens_q
            num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            advantages = torch.repeat_interleave(advantages, num_tokens, dim=1)  # (1, max_len)

            rollout_logprobs = None
            if rollout_logprobs_list is not None:
                rollout_logprobs = torch.cat(rollout_logprobs_list, dim=1)  # (1, max_len)

            packed_data_batches.append(
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": shifted_labels,
                    "advantages": advantages,
                    "rollout_logprobs": rollout_logprobs,
                }
            )
        return packed_data_batches

    def _grouped_by_max_length(self, packed_data_batches):
        # sort 过后可能第一个 batch 会有很多 pad tokens，因为最后一个 pack 可能只有少量真实数据。
        # 比如组成了 16 个 pack，第 16 个 pack 可能只有几条真实数据，剩下的都是 pad tokens。
        # 排序后这条 pack 会被放在最前面，导致 rank0 的第一个 step 消耗的有效 token 数往往少于其他 rank，是正常现象。
        return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

    def _balance_split_batch(self, data_batches, partition_size):
        """Reorder the data on single controller such that each dp rank gets
        similar total tokens."""
        global_seqlen_lst = [data["seq_ctx"].input_ids.numel() for data in data_batches]
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=partition_size, equal_size=True
        )
        balanced_batches = []
        tokens_in_partition = []
        for partition in global_partition_lst:
            partition_batch = [data_batches[i] for i in partition]
            tokens_in_partition.append(sum(data["seq_ctx"].input_ids.numel() for data in partition_batch))
            balanced_batches.append(partition_batch)
        get_logger().info(f"Balanced split into {partition_size} partitions with tokens: {tokens_in_partition}")
        return balanced_batches

    def _create_padding_sample(
        self,
        pad_len: int,
        pack_max_length: int,
        is_qwen3_vl: bool = False,
        has_rollout_routed_experts: bool = False,
        has_rollout_logprobs: bool = True,
        n_routed_experts: int | None = None,
        split_size: int = 1024,
    ):
        # padding input_ids
        pad_tokens = tuple(
            torch.zeros(1, split_size, dtype=torch.long, device="cpu") for _ in range(pad_len // split_size)
        )
        if pad_len % split_size > 0:
            pad_tokens = pad_tokens + (torch.zeros(1, pad_len % split_size, dtype=torch.long, device="cpu"),)
        pad_tokens = cast(tuple[torch.LongTensor, ...], pad_tokens)
        pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
        pad_seq_ctx.num_padding = pad_len

        # padding mm positions_ids
        if is_qwen3_vl:
            _position_ids_list = []
            for pad_token in pad_tokens:
                _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                _position_ids_list.append(_position_ids)
            position_ids = torch.cat(_position_ids_list, dim=-1)
            position_ids = cast(torch.LongTensor, position_ids)
            pad_seq_ctx.position_ids = position_ids

        # padding rollout routed experts
        if has_rollout_routed_experts:
            assert n_routed_experts, "n_routed_experts must be provided when has_rollout_routed_experts is True"
            if pad_len == pack_max_length:
                pad_rand_index = torch.randint(
                    low=0, high=1, size=(1, 1, 1)
                )  # add dummy data, true data will be initialized in train worker.fit
            else:
                pad_rand_index = torch.randint(low=0, high=n_routed_experts, size=(pad_len, 1, 1))
            pad_seq_ctx.rollout_routed_experts = pad_rand_index

        pad_labels = torch.full((1, pad_len), -100, dtype=torch.long, device="cpu")
        pad_advantage_length = pack_max_length if pad_len == pack_max_length else math.ceil(pad_len / 1024)
        pad_advantage = torch.full(
            (1, pad_advantage_length),
            -100,
            dtype=torch.float32,
            device="cpu",
        )
        pad_rollout_logprobs = (
            torch.zeros(1, pad_len, dtype=torch.float32, device="cpu") if has_rollout_logprobs else None
        )

        return {
            "seq_ctx": pad_seq_ctx,
            "shifted_labels": pad_labels,
            "advantages": pad_advantage,
            "rollout_logprobs": pad_rollout_logprobs,
        }

    def _pack(self, mini_batch, pack_max_length):
        assert len(mini_batch) > 0, "mini_batch should not be empty"
        seqlen_list = []
        for data in mini_batch:
            assert data["seq_ctx"].input_ids.numel() <= pack_max_length, (
                f"Single sample seq len {data['seq_ctx'].input_ids.numel()} exceeds pack_max_length {pack_max_length}"
            )
            seqlen_list.append(data["seq_ctx"].input_ids.numel())
        total_length = sum(seqlen_list)

        if total_length <= pack_max_length:
            return [mini_batch]  # No packing needed

        num_packs = math.ceil(total_length / pack_max_length)
        partitions_indices = get_seqlen_balanced_partitions(
            seqlen_list=seqlen_list, k_partitions=num_packs, equal_size=False
        )

        packed_mini_batches = []
        for partition in partitions_indices:
            packed_batch = [mini_batch[i] for i in partition]
            packed_mini_batches.append(packed_batch)
        return packed_mini_batches

    def _get_data_batches_properties(self, data_batches: list[ColateItem]):
        """Extract properties from the first element of data_batches."""
        if not data_batches:
            return {
                "is_qwen3_vl": False,
                "has_rollout_routed_experts": False,
                "has_rollout_logprobs": False,
                "n_routed_experts": None,
            }

        first_item = data_batches[0]
        seq_ctx = first_item["seq_ctx"]

        is_qwen3_vl = seq_ctx.position_ids is not None and len(seq_ctx.position_ids.shape) == 3
        has_rollout_logprobs = "rollout_logprobs" in first_item and first_item["rollout_logprobs"] is not None
        has_rollout_routed_experts = seq_ctx.rollout_routed_experts is not None

        language_cfg = None
        if has_rollout_routed_experts:
            language_cfg = self.model_cfg
            if isinstance(self.model_cfg, BaseComposeConfig):
                language_cfg = self.model_cfg.text_config

        return {
            "is_qwen3_vl": is_qwen3_vl,
            "has_rollout_routed_experts": has_rollout_routed_experts,
            "has_rollout_logprobs": has_rollout_logprobs,
            "n_routed_experts": language_cfg.n_routed_experts if language_cfg is not None else None,
        }

    @ray_method
    def fit(
        self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int, enable_dp_balance: bool = True
    ):
        batch_props = self._get_data_batches_properties(data_batches)
        is_qwen3_vl = batch_props["is_qwen3_vl"]
        has_rollout_routed_experts = batch_props["has_rollout_routed_experts"]
        has_rollout_logprobs = batch_props["has_rollout_logprobs"]
        n_routed_experts = batch_props["n_routed_experts"]

        world_size = len(self.workers)
        dp_size = world_size // self.data_replicate_size
        assert world_size % self.data_replicate_size == 0, "world_size must be divisible by data_replicate_size"
        optimizer_steps = self.worker_cfg.optimizer_steps

        if enable_dp_balance:
            # 按照 dp_size 对数据进行重新分配，保证每个 dp rank 上的 token 数量大致相同
            batches_per_dp_group = self._balance_split_batch(data_batches, dp_size)
        else:
            batches_per_dp_group = np.array_split(data_batches, dp_size)
            tokens_in_partition = []
            for batch in batches_per_dp_group:
                tokens_in_partition.append(sum(data["seq_ctx"].input_ids.numel() for data in batch))
            get_logger().info(f"default split into {dp_size} partitions with tokens: {tokens_in_partition}")

        packed_data_batches: list[list[list[dict]]] = [[[] for _ in range(optimizer_steps)] for _ in range(dp_size)]
        max_packs_per_card = [0] * optimizer_steps

        for dp_rank, dp_worker_data_batches in enumerate(batches_per_dp_group):
            # 每个worker 内部按照optimizer_steps将token均分
            if enable_dp_balance:
                random.shuffle(dp_worker_data_batches)
            mini_batch_for_steps = self._balance_split_batch(dp_worker_data_batches, optimizer_steps)

            for step_idx, step_mini_batch in enumerate(mini_batch_for_steps):
                # pack
                pack_mini_batch = self._pack(step_mini_batch, pack_max_length)
                if len(pack_mini_batch) > max_packs_per_card[step_idx]:
                    max_packs_per_card[step_idx] = len(pack_mini_batch)

                for pack in pack_mini_batch:
                    seq_ctx_list = [item["seq_ctx"] for item in pack]
                    label_list = [item["shifted_labels"] for item in pack]
                    advantage_list = [torch.tensor([item["advantage"]]).float().unsqueeze(0) for item in pack]
                    rollout_logprobs_list = [
                        item["rollout_logprobs"] if has_rollout_logprobs else None for item in pack
                    ]
                    padding_len = pack_max_length - sum([item["seq_ctx"].input_ids.numel() for item in pack])
                    if padding_len > 0:
                        padding_sample = self._create_padding_sample(
                            padding_len,
                            pack_max_length,
                            is_qwen3_vl=is_qwen3_vl,
                            has_rollout_routed_experts=has_rollout_routed_experts,
                            has_rollout_logprobs=has_rollout_logprobs,
                            n_routed_experts=n_routed_experts,
                        )
                        seq_ctx_list.append(padding_sample["seq_ctx"])
                        label_list.append(padding_sample["shifted_labels"])
                        advantage_list.append(padding_sample["advantages"])
                        rollout_logprobs_list.append(padding_sample["rollout_logprobs"])

                    packed_seq_ctx = SequenceContext.pack(seq_ctx_list)
                    packed_shifted_labels = torch.cat(label_list, dim=1)
                    cu_seq_lens_q = packed_seq_ctx.cu_seq_lens_q
                    packed_num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
                    packed_advantages = torch.cat(advantage_list, dim=1)
                    packed_advantages = torch.repeat_interleave(packed_advantages, packed_num_tokens, dim=1)
                    if has_rollout_logprobs:
                        cast_rollout_logprobs_list = [cast(torch.Tensor, item) for item in rollout_logprobs_list]
                        packed_rollout_logprobs = torch.cat(cast_rollout_logprobs_list, dim=1)
                    else:
                        packed_rollout_logprobs = None
                    packed_data_batches[dp_rank][step_idx].append(
                        {
                            "seq_ctx": packed_seq_ctx,
                            "shifted_labels": packed_shifted_labels,
                            "advantages": packed_advantages,
                            "rollout_logprobs": packed_rollout_logprobs,
                        }
                    )

        get_logger().info(f"Gradient accumulation steps: {max_packs_per_card}")
        # padding for each worker to have same number of packs
        for dp_rank in range(dp_size):
            for step_idx in range(optimizer_steps):
                max_packs = max_packs_per_card[step_idx]
                num_current_packs = len(packed_data_batches[dp_rank][step_idx])
                num_padding_packs = max_packs - num_current_packs

                if num_padding_packs > 0:
                    padding_sample = self._create_padding_sample(
                        pack_max_length,
                        pack_max_length,
                        is_qwen3_vl=is_qwen3_vl,
                        has_rollout_routed_experts=has_rollout_routed_experts,
                        has_rollout_logprobs=has_rollout_logprobs,
                        n_routed_experts=n_routed_experts,
                    )
                    padding_samples = [padding_sample for _ in range(num_padding_packs)]
                    packed_data_batches[dp_rank][step_idx].extend(padding_samples)

        handles = []
        for worker_idx, worker in enumerate(self.workers):
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=packed_data_batches[worker_idx // self.data_replicate_size],
                    rollout_idx=rollout_idx,
                )
            )
        ray.get(handles)

    @ray_method
    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
        return

    @ray_method
    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
        elif target == "optimizer":
            ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
        elif target == "all":
            ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
            ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
        return

    @ray_method
    def update_rollout_info(self, info_dict):
        ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

    @ray_method
    def update_weights(self):
        """Update the weights of the training workers."""
        handles = [worker.update_weights.remote() for worker in self.workers]
        ray.get(handles)
        return

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles)
        return

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        handles = [worker.resume.remote(load_checkpoint_cfg) for worker in self.workers]  # type: ignore
        ray.get(handles)
        return

    @ray_method
    def save_dcp(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save_dcp.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles)
        return

    @ray_method
    def ready(self) -> bool:
        return True


TrainingController = ray.remote(RawTrainingController)
TrainingControllerProxy = ActorProxy[RawTrainingController]
