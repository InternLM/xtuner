import math
from pathlib import Path
from typing import cast

import numpy as np
import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.utils import get_seqlen_balanced_partitions
from xtuner.v1.utils import get_logger

from .worker import WorkerInputItem


class DataBatchPacker:
    def __init__(
        self,
        pack_max_length: int,
        world_size: int,
        data_replicate_size: int,
        optimizer_steps: int,
        pack_strategy: str = "greedy",
        model_cfg: TransformerConfig | None = None,
        worker_log_dir: str | None = None,
    ):
        self.pack_max_length = pack_max_length
        self.world_size = world_size
        self.data_replicate_size = data_replicate_size
        self.optimizer_steps = optimizer_steps
        if worker_log_dir is not None:
            self.worker_log_dir = Path(worker_log_dir) if isinstance(worker_log_dir, str) else worker_log_dir
            self.logger = get_logger(log_dir=self.worker_log_dir, tag="TrainingController")
        else:
            self.logger = get_logger()

        self.data_batch_properties = {
            "is_qwen3_vl": False,
            "has_rollout_routed_experts": False,
            "has_rollout_logprobs": False,
            "n_routed_experts": None,
        }
        self.strategy_map = {"greedy": self.greedy_pack, "balance": self.balance_pack, "native": self.native_pack}
        if pack_strategy not in self.strategy_map:
            raise ValueError(f"Unknown packing strategy: {pack_strategy}")
        self._pack_impl = self.strategy_map[pack_strategy]
        self.dp_size = self.world_size // self.data_replicate_size
        self.padding_tokens = 0
        self.model_cfg = model_cfg

    def pack(self, data_batches: list[WorkerInputItem]) -> tuple[list[list[list[WorkerInputItem]]], int]:
        self.padding_tokens = 0
        if not data_batches:
            return [], 0
        self._set_data_batch_properties(data_batches)
        return self._pack_impl(data_batches), self.padding_tokens

    def greedy_pack(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 策略核心：贪心打包
        # 1. 使用贪心算法将所有样本打包成一个一维的 pack 列表。
        #    此过程不考虑 DP 和优化步骤，目标是尽可能填满每个 pack。
        pack_infos = self._get_pack_infos(
            data_batches,
            [data["seq_ctx"].input_ids.numel() for data in data_batches],  # type: ignore[union-attr]
            self.pack_max_length,
        )
        total_data_batches: list[WorkerInputItem] = []

        # 2. 遍历打包信息，将每个 pack 内的样本拼接并填充到 pack_max_length。
        for pack_info in pack_infos:
            indices = pack_info["indices"]
            batch4pack = [data_batches[i] for i in indices]
            packed_item = self._pad_and_pack_batches(batch4pack, self.pack_max_length)
            total_data_batches.append(packed_item)

        # 3. 为了均匀分配，填充整个 batch，使其总 pack 数能被 dp_size 整除。
        dp_size = self.world_size // self.data_replicate_size
        num_packed_data_batches = len(total_data_batches)
        pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
        if pad_num > 0:
            pad_data_samples = [
                self._create_padding_item(self.pack_max_length, self.pack_max_length) for _ in range(pad_num)
            ]
            total_data_batches = total_data_batches + pad_data_samples

        # 4. 将填充后的 pack 列表按 dp_size 和 optimizer_steps 重新分配。
        each_dp_batches_num = len(total_data_batches) // dp_size
        if each_dp_batches_num < self.optimizer_steps:
            iters_per_step = 1  # each optimizer step has at least one batch
            actual_optimizer_steps = each_dp_batches_num
        else:
            iters_per_step = math.ceil(each_dp_batches_num // self.optimizer_steps)
            actual_optimizer_steps = math.ceil(each_dp_batches_num // iters_per_step)
        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(dp_size)
        ]
        for dp_rank in range(dp_size):
            for step in range(actual_optimizer_steps):
                start_idx = dp_rank * each_dp_batches_num + step * iters_per_step
                end_idx = min(start_idx + iters_per_step, each_dp_batches_num * (dp_rank + 1))
                packed_data_batches[dp_rank][step] = total_data_batches[start_idx:end_idx]
        return packed_data_batches

    def balance_pack(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 策略核心：层层 token 均衡
        # 目标是让每个 DP rank 在每个 optimizer_step 中处理的 token 数都尽可能接近。
        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(self.optimizer_steps)] for _ in range(self.dp_size)
        ]
        # 1. 按照 dp_size 对数据进行重新分配，保证每个 dp rank 上的 token 数量大致相同
        batches_per_dp_group: list[list[WorkerInputItem]] = self._balance_split_batch(data_batches, self.dp_size)
        max_packs_per_step = [0] * self.optimizer_steps

        optimizer_steps = self.optimizer_steps
        for dp_rank, dp_worker_data_batches in enumerate(batches_per_dp_group):
            # 2. 在每个 DP 组内部，根据 token 数将数据均衡地分给 optimizer_steps 个 mini-batch。
            mini_batch_for_steps: list[list[WorkerInputItem]]
            if len(dp_worker_data_batches) < self.optimizer_steps:
                optimizer_steps = len(dp_worker_data_batches)
                mini_batch_for_steps = [[dp_worker_data_batches[i]] for i in range(optimizer_steps)]
            else:
                mini_batch_for_steps = self._balance_split_batch(dp_worker_data_batches, self.optimizer_steps)
            for step_idx, step_mini_batch in enumerate(mini_batch_for_steps):
                # 3. 第三次均衡：在每个 mini-batch 内部，再次进行均衡打包，并记录每个 step 的最大 pack 数。
                self._pack_mini_batches_for_each_optimizer_step(
                    packed_data_batches, step_mini_batch, dp_rank, step_idx, self.pack_max_length
                )
                if len(packed_data_batches[dp_rank][step_idx]) > max_packs_per_step[step_idx]:
                    max_packs_per_step[step_idx] = len(packed_data_batches[dp_rank][step_idx])

        self.logger.info(f"Gradient accumulation for each optimizer steps: {max_packs_per_step}")

        # 4. 最终填充：根据记录的最大 pack 数，将所有 DP rank 在每个 step 的 pack 数量填充至一致。
        for step_idx in range(optimizer_steps):
            max_packs = max_packs_per_step[step_idx]
            packed_data_batches = self._pad_to_max_packs_across_workes(
                packed_data_batches, step_idx, max_packs, self.pack_max_length
            )
        for dp_rank in range(self.dp_size):
            packed_data_batches[dp_rank] = packed_data_batches[dp_rank][:optimizer_steps]
        return packed_data_batches

    def native_pack(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 策略核心：按样本数量朴素切分,保证样本顺序
        # 这种方法不考虑 token 长度，仅保证每个 DP rank 和 optimizer_step 分到的样本数量大致相等。
        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(self.optimizer_steps)] for _ in range(self.dp_size)
        ]
        if len(data_batches) < self.dp_size:
            pad_num = self.dp_size - len(data_batches)
            for _ in range(pad_num):
                data_batches.append(
                    self._create_padding_item(data_batches[0]["seq_ctx"].input_ids.shape[1], self.pack_max_length)  # type: ignore[union-attr]
                )
        batches_per_dp_group: list[list[WorkerInputItem]] = np.array_split(data_batches, self.dp_size)
        max_packs_per_step = [0] * self.optimizer_steps

        optimizer_steps = self.optimizer_steps
        for dp_rank, dp_worker_data_batches in enumerate(batches_per_dp_group):
            mini_batch_for_steps: list[list[WorkerInputItem]]
            if len(dp_worker_data_batches) < self.optimizer_steps:
                optimizer_steps = len(dp_worker_data_batches)
                mini_batch_for_steps = [[dp_worker_data_batches[i]] for i in range(optimizer_steps)]
            else:
                mini_batch_for_steps = np.array_split(dp_worker_data_batches, self.optimizer_steps)
            for step_idx, step_mini_batch in enumerate(mini_batch_for_steps):
                self._pack_mini_batches_for_each_optimizer_step(
                    packed_data_batches, step_mini_batch, dp_rank, step_idx, self.pack_max_length
                )
                if len(packed_data_batches[dp_rank][step_idx]) > max_packs_per_step[step_idx]:
                    max_packs_per_step[step_idx] = len(packed_data_batches[dp_rank][step_idx])

        self.logger.info(f"Gradient accumulation for each optimizer steps: {max_packs_per_step}")

        # padding for each worker to have same number of packs in each optimizer step
        for step_idx in range(optimizer_steps):
            max_packs = max_packs_per_step[step_idx]
            packed_data_batches = self._pad_to_max_packs_across_workes(
                packed_data_batches, step_idx, max_packs, self.pack_max_length
            )

        for dp_rank in range(self.dp_size):
            packed_data_batches[dp_rank] = packed_data_batches[dp_rank][:optimizer_steps]
        return packed_data_batches

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

    def _balance_split_batch(self, data_batches: list[WorkerInputItem], partition_size) -> list[list[WorkerInputItem]]:
        """Reorder the data on single controller such that each dp rank gets
        similar total tokens."""
        global_seqlen_lst = [data["seq_ctx"].input_ids.numel() for data in data_batches]  # type: ignore[union-attr]
        balanced_batches: list[list[WorkerInputItem]] = []
        if len(global_seqlen_lst) >= partition_size:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=partition_size, equal_size=True
            )
            tokens_in_partition = []
            for partition in global_partition_lst:
                partition_batch = [data_batches[i] for i in partition]
                tokens_in_partition.append(sum(data["seq_ctx"].input_ids.numel() for data in partition_batch))
                balanced_batches.append(partition_batch)
            self.logger.info(f"Balanced split into {partition_size} partitions with tokens: {tokens_in_partition}")
        else:
            balanced_batches = [data_batches]
            pad_num = partition_size - len(global_seqlen_lst)
            for i in range(pad_num):
                balanced_batches.append([self._create_padding_item(global_seqlen_lst[0], self.pack_max_length)])
        return balanced_batches

    def _set_data_batch_properties(self, data_batches: list[WorkerInputItem]):
        if not data_batches:
            return

        first_item = data_batches[0]
        seq_ctx = first_item["seq_ctx"]

        self.data_batch_properties["is_qwen3_vl"] = (
            seq_ctx.position_ids is not None and len(seq_ctx.position_ids.shape) == 3
        )
        self.data_batch_properties["has_rollout_logprobs"] = (
            "rollout_logprobs" in first_item and first_item["rollout_logprobs"] is not None
        )
        self.data_batch_properties["has_rollout_routed_experts"] = seq_ctx.rollout_routed_experts is not None

        language_cfg = None
        if self.data_batch_properties["has_rollout_routed_experts"]:
            language_cfg = self.model_cfg
            if isinstance(self.model_cfg, BaseComposeConfig):
                language_cfg = self.model_cfg.text_config

        self.data_batch_properties["n_routed_experts"] = (
            language_cfg.n_routed_experts if language_cfg is not None else None
        )
        self.logger.info(f"Data batch properties set: {self.data_batch_properties}")

    def _pad_and_pack_batches(self, batch4pack: list[WorkerInputItem], pack_max_length: int) -> WorkerInputItem:
        seq_ctx_list = [item["seq_ctx"] for item in batch4pack]
        label_list = [item["shifted_labels"] for item in batch4pack]
        advantage_list = []
        for item in batch4pack:
            advantages = item["advantages"].reshape(1, -1)
            advantage_list.append(advantages)
        rollout_logprobs_list = [
            item["rollout_logprobs"] if self.data_batch_properties["has_rollout_logprobs"] else None
            for item in batch4pack
        ]
        cur_length = 0
        for item in batch4pack:
            cur_length += item["seq_ctx"].input_ids.numel()  # type: ignore[union-attr]
        padding_len = pack_max_length - cur_length

        if padding_len > 0:
            padding_item = self._create_padding_item(padding_len, pack_max_length)
            seq_ctx_list.append(padding_item["seq_ctx"])
            label_list.append(padding_item["shifted_labels"])
            advantage_list.append(padding_item["advantages"])
            rollout_logprobs_list.append(padding_item["rollout_logprobs"])

        packed_seq_ctx = SequenceContext.cat(seq_ctx_list)
        packed_shifted_labels = torch.cat(label_list, dim=1)  # type: ignore[arg-type]
        packed_shifted_labels = cast(torch.LongTensor, packed_shifted_labels)
        cu_seq_lens_q = packed_seq_ctx.cu_seq_lens_q
        packed_num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
        packed_advantages = torch.cat(advantage_list, dim=1)
        packed_advantages = torch.repeat_interleave(packed_advantages, packed_num_tokens, dim=1)
        if self.data_batch_properties["has_rollout_logprobs"]:
            cast_rollout_logprobs_list = [cast(torch.Tensor, item) for item in rollout_logprobs_list]
            packed_rollout_logprobs = torch.cat(cast_rollout_logprobs_list, dim=1)
        else:
            packed_rollout_logprobs = None

        optimizer_step_packs: WorkerInputItem = {
            "seq_ctx": packed_seq_ctx,
            "shifted_labels": packed_shifted_labels,
            "advantages": packed_advantages,
            "rollout_logprobs": packed_rollout_logprobs,
        }
        return optimizer_step_packs

    def _pad_to_max_packs_across_workes(
        self,
        packed_data_batches: list[list[list[WorkerInputItem]]],
        step_idx: int,
        max_packs: int,
        pack_max_length: int,
    ):
        for dp_rank in range(len(packed_data_batches)):
            num_current_packs = len(packed_data_batches[dp_rank][step_idx])
            num_padding_packs = max_packs - num_current_packs

            if num_padding_packs > 0:
                padding_items = [
                    self._create_padding_item(pack_max_length, pack_max_length) for _ in range(num_padding_packs)
                ]
                packed_data_batches[dp_rank][step_idx].extend(padding_items)
        return packed_data_batches

    def _pack_mini_batches_for_each_optimizer_step(
        self,
        packed_data_batches: list[list[list[WorkerInputItem]]],
        step_mini_batches: list[WorkerInputItem],
        dp_rank: int,
        step_idx: int,
        pack_max_length: int,
    ):
        seqlen_list = []
        for data in step_mini_batches:
            assert data["seq_ctx"].input_ids.numel() <= pack_max_length, (  # type: ignore[union-attr]
                f"Single sample seq len {data['seq_ctx'].input_ids.numel()} exceeds pack_max_length {pack_max_length}"  # type: ignore[union-attr]
            )
            seqlen_list.append(data["seq_ctx"].input_ids.numel())  # type: ignore[union-attr]
        total_length = sum(seqlen_list)

        batch_list_for_pack: list[list[WorkerInputItem]] = []
        if total_length > pack_max_length:
            # balance mini batches across gradient accumulation steps
            num_packs = math.ceil(total_length / pack_max_length)
            partitions_indices = get_seqlen_balanced_partitions(
                seqlen_list=seqlen_list, k_partitions=num_packs, equal_size=False
            )
            for partition in partitions_indices:
                batch_list = [step_mini_batches[i] for i in partition]
                batch_list_for_pack.append(batch_list)
        else:
            batch_list_for_pack = [step_mini_batches]

        for batch4pack in batch_list_for_pack:
            # pad and pack batches into a single optimizer step pack
            step_pack = self._pad_and_pack_batches(batch4pack, pack_max_length)
            packed_data_batches[dp_rank][step_idx].append(step_pack)

    def _create_padding_item(
        self,
        pad_len: int,
        pack_max_length: int,
        split_size: int = 1024,
    ) -> WorkerInputItem:
        # padding input_ids
        self.padding_tokens += pad_len
        pad_tokens = tuple(
            torch.zeros(1, split_size, dtype=torch.long, device="cpu") for _ in range(pad_len // split_size)
        )
        if pad_len % split_size > 0:
            pad_tokens = pad_tokens + (torch.zeros(1, pad_len % split_size, dtype=torch.long, device="cpu"),)
        pad_tokens = cast(tuple[torch.LongTensor, ...], pad_tokens)
        pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
        pad_seq_ctx.num_padding = pad_len

        # padding mm positions_ids
        if self.data_batch_properties["is_qwen3_vl"]:
            _position_ids_list = []
            for pad_token in pad_tokens:
                _position_ids = torch.arange(pad_token.size(-1)).view(1, 1, -1).expand(3, 1, -1)
                _position_ids_list.append(_position_ids)
            position_ids = torch.cat(_position_ids_list, dim=-1)
            position_ids = cast(torch.LongTensor, position_ids)
            pad_seq_ctx.position_ids = position_ids

        # padding rollout routed experts
        if self.data_batch_properties["has_rollout_routed_experts"]:
            assert self.data_batch_properties["n_routed_experts"], (
                "n_routed_experts must be provided when has_rollout_routed_experts is True"
            )
            if pad_len == pack_max_length:
                pad_rand_index = torch.randint(
                    low=0, high=1, size=(1, 1, 1)
                )  # add dummy data, true data will be initialized in train worker.fit
            else:
                pad_rand_index = torch.randint(
                    low=0, high=self.data_batch_properties["n_routed_experts"], size=(pad_len, 1, 1)
                )
            pad_seq_ctx.rollout_routed_experts = pad_rand_index

        pad_labels = cast(torch.LongTensor, torch.full((1, pad_len), -100, dtype=torch.int64, device="cpu"))
        pad_advantage_length = pack_max_length if pad_len == pack_max_length else math.ceil(pad_len / split_size)
        pad_advantage = torch.full(
            (1, pad_advantage_length),
            -100,
            dtype=torch.float32,
            device="cpu",
        )
        pad_rollout_logprobs = (
            torch.zeros(1, pad_len, dtype=torch.float32, device="cpu")
            if self.data_batch_properties["has_rollout_logprobs"]
            else None
        )

        padding_item: WorkerInputItem = {
            "seq_ctx": pad_seq_ctx,
            "shifted_labels": pad_labels,
            "advantages": pad_advantage,
            "rollout_logprobs": pad_rollout_logprobs,
        }
        return padding_item
