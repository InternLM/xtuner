import math
import random
from pathlib import Path
from typing import cast

import numpy as np
import torch

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.packing import get_pack_infos_by_expand_soft_split, get_pack_infos_by_soft_split
from xtuner.v1.datasets.sampler import get_length_grouped_indices
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.base.worker import WorkerInputItem
from xtuner.v1.utils import get_logger


class RLDataPacker:
    def __init__(
        self,
        pack_max_length: int,
        world_size: int,
        data_replicate_size: int,
        optimizer_steps: int,
        pack_strategy: str = "greedy",
        model_cfg: TransformerConfig | None = None,
        worker_log_dir: str | None = None,
        seed: int | None = None,
    ):
        self.pack_max_length = pack_max_length
        self.world_size = world_size
        self.data_replicate_size = data_replicate_size
        self.optimizer_steps = optimizer_steps
        self.split_size = 1024
        self.seed = seed
        self.random = random.Random()
        self.torch_generator = torch.Generator()
        if seed is not None:
            self.random = random.Random(seed)
            self.torch_generator.manual_seed(seed)

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
        self.strategy_map = {
            "greedy": self.greedy_pack_and_split,
            "balance": self.balance_split_and_pack,
            "native": self.native_split_and_pack,
        }
        if pack_strategy not in self.strategy_map:
            raise ValueError(f"Unknown packing strategy: {pack_strategy}")
        self._impl = self.strategy_map[pack_strategy]
        self.dp_size = self.world_size // self.data_replicate_size
        self.padding_tokens = 0
        self.model_cfg = model_cfg

    def pack(self, data_batches: list[WorkerInputItem]) -> tuple[list[list[list[WorkerInputItem]]], int]:
        self.padding_tokens = 0
        if not data_batches:
            return [], 0
        self._set_data_batch_properties(data_batches)
        return self._impl(data_batches), self.padding_tokens

    def native_split_and_pack(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 1. 预处理，保证样本数量可以被 dp_size 整除
        if len(data_batches) % self.dp_size != 0:
            pad_num = self.dp_size - (len(data_batches) % self.dp_size)
            padding_item = self._create_padding_item(self.split_size, self.pack_max_length)
            data_batches.extend([padding_item] * pad_num)

        # 2. 按照 dp_size 切分样本
        batches_per_dp_group: list[list[WorkerInputItem]] = np.array_split(data_batches, self.dp_size)
        actual_optimizer_steps = min(len(batches_per_dp_group[0]), self.optimizer_steps)
        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(self.dp_size)
        ]
        max_packs_per_step = [0] * actual_optimizer_steps

        for dp_rank, dp_worker_data_batches in enumerate(batches_per_dp_group):
            # 3. 按照 actual_optimizer_steps 切分样本
            batches_for_optim_steps = np.array_split(dp_worker_data_batches, actual_optimizer_steps)
            for step_idx, step_mini_batches in enumerate(batches_for_optim_steps):
                # 4. 对每个 optimizer step 的样本进行打包
                each_step_pack_list = self._pack(step_mini_batches, "soft", self.pack_max_length)
                packed_data_batches[dp_rank][step_idx] = each_step_pack_list
                max_packs_per_step[step_idx] = max(
                    max_packs_per_step[step_idx], len(packed_data_batches[dp_rank][step_idx])
                )

        self.logger.info(f"Gradient accumulation for each optimizer steps: {max_packs_per_step}")

        # 5. padding for each worker to have same number of packs in each optimizer step
        for step_idx in range(actual_optimizer_steps):
            max_packs = max_packs_per_step[step_idx]
            for dp_rank in range(self.dp_size):
                num_current_packs = len(packed_data_batches[dp_rank][step_idx])
                num_padding_packs = max_packs - num_current_packs

                if num_padding_packs > 0:
                    padding_items = [
                        self._create_padding_item(self.pack_max_length, self.pack_max_length)
                        for _ in range(num_padding_packs)
                    ]
                    packed_data_batches[dp_rank][step_idx].extend(padding_items)
        return packed_data_batches

    def balance_split_and_pack(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 1. 保证每张卡获取的样本总长度大致相等
        max_lengths = self._get_seqlen_from_data_batches(data_batches)
        indices = get_length_grouped_indices(
            max_lengths=max_lengths,
            group_batch_size=len(data_batches),
            group_size=self.dp_size,
            torch_generator=self.torch_generator,
            random_generator=self.random,
        )

        partitioned_data: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(self.optimizer_steps)] for _ in range(self.dp_size)
        ]

        # 2. 根据indices将样本分配到每张卡的每个 optimizer step 上
        for i, idx in enumerate(indices):
            dp_rank = i % self.dp_size
            step_idx = (i // self.dp_size) % self.optimizer_steps
            partitioned_data[dp_rank][step_idx].append(data_batches[idx])

        actual_optimizer_steps = 0
        for dp_rank in range(self.dp_size):
            rank_max_step = 0
            for step_idx in range(self.optimizer_steps):
                if len(partitioned_data[dp_rank][step_idx]) > 0:
                    rank_max_step = step_idx + 1
            actual_optimizer_steps = max(actual_optimizer_steps, rank_max_step)

        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(self.dp_size)
        ]

        max_packs_per_step = [0] * actual_optimizer_steps

        for dp_rank in range(self.dp_size):
            for step_idx in range(actual_optimizer_steps):
                # 3. 对每个卡每个 optimizer step 的样本进行打包
                step_data = partitioned_data[dp_rank][step_idx]
                packed_step_data = self._pack(step_data, "soft", self.pack_max_length)
                packed_data_batches[dp_rank][step_idx] = packed_step_data
                max_packs_per_step[step_idx] = max(
                    max_packs_per_step[step_idx], len(packed_data_batches[dp_rank][step_idx])
                )

        # 4. padding for each worker to have same number of packs in each optimizer step
        for step_idx in range(actual_optimizer_steps):
            max_packs = max_packs_per_step[step_idx]
            for dp_rank in range(self.dp_size):
                num_current_packs = len(packed_data_batches[dp_rank][step_idx])
                num_padding_packs = max_packs - num_current_packs

                if num_padding_packs > 0:
                    padding_items = [
                        self._create_padding_item(self.pack_max_length, self.pack_max_length)
                        for _ in range(num_padding_packs)
                    ]
                    packed_data_batches[dp_rank][step_idx].extend(padding_items)
        return packed_data_batches

    def greedy_pack_and_split(self, data_batches: list[WorkerInputItem]) -> list[list[list[WorkerInputItem]]]:
        # 1. 使用贪心算法将所有样本打包成一个一维的 pack 列表。
        total_data_batches = self._pack(data_batches, "expand_soft", self.pack_max_length)
        # 2. 为了均匀分配，填充整个 batch，使其总 pack 数能被 dp_size 整除。
        dp_size = self.world_size // self.data_replicate_size
        num_packed_data_batches = len(total_data_batches)
        pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
        if pad_num > 0:
            pad_data_samples = [
                self._create_padding_item(self.pack_max_length, self.pack_max_length) for _ in range(pad_num)
            ]
            total_data_batches = total_data_batches + pad_data_samples

        # 3. 将填充后的 pack 列表按 dp_size 和 optimizer_steps 重新分配。
        each_dp_batches_num = len(total_data_batches) // dp_size
        if each_dp_batches_num < self.optimizer_steps:
            iters_per_step = 1  # each optimizer step has at least one batch
            actual_optimizer_steps = each_dp_batches_num
        else:
            iters_per_step = math.ceil(each_dp_batches_num / self.optimizer_steps)
            actual_optimizer_steps = math.ceil(each_dp_batches_num / iters_per_step)
        packed_data_batches: list[list[list[WorkerInputItem]]] = [
            [[] for _ in range(actual_optimizer_steps)] for _ in range(dp_size)
        ]
        for dp_rank in range(dp_size):
            for step in range(actual_optimizer_steps):
                start_idx = dp_rank * each_dp_batches_num + step * iters_per_step
                end_idx = min(start_idx + iters_per_step, each_dp_batches_num * (dp_rank + 1))
                packed_data_batches[dp_rank][step] = total_data_batches[start_idx:end_idx]
        return packed_data_batches

    def _get_seqlen_from_data_batches(self, data_batches: list[WorkerInputItem]) -> list[int]:
        seqlen_list = []
        for data in data_batches:
            assert data["seq_ctx"].input_ids.numel() <= self.pack_max_length, (  # type: ignore[union-attr]
                f"Single sample seq len {data['seq_ctx'].input_ids.numel()} exceeds pack_max_length {self.pack_max_length}"  # type: ignore[union-attr]
            )
            seqlen_list.append(data["seq_ctx"].input_ids.numel())  # type: ignore[union-attr]
        return seqlen_list

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

    def _pack(
        self, data_batches: list[WorkerInputItem], pack_method: str, pack_max_length: int
    ) -> list[WorkerInputItem]:
        seqlen_list = self._get_seqlen_from_data_batches(data_batches)
        seqlen_list_np = np.array(seqlen_list)
        total_length = sum(seqlen_list)
        seqlen_indices = list(range(len(seqlen_list)))
        fake_dataset_id = 0
        each_step_pack_list: list[WorkerInputItem] = []
        if total_length > pack_max_length:
            if pack_method == "expand_soft":
                self.random.shuffle(seqlen_indices)
                pack_infos = get_pack_infos_by_expand_soft_split(
                    seqlen_indices, fake_dataset_id, seqlen_list_np, pack_max_length, pack_workers=1
                )
            elif pack_method == "soft":
                pack_infos = get_pack_infos_by_soft_split(
                    seqlen_indices,
                    fake_dataset_id,
                    seqlen_list_np,
                    pack_max_length,
                )
            else:
                raise ValueError(f"Unknown pack method: {pack_method}")
            for pack_info in pack_infos:
                indices = pack_info["indices"]
                batch4pack = [data_batches[i] for i in indices]
                each_step_pack_list.append(self._single_pack(batch4pack, pack_max_length))
        else:
            each_step_pack_list.append(self._single_pack(data_batches, pack_max_length))
        return each_step_pack_list

    def _single_pack(self, data_batches: list[WorkerInputItem], pack_max_length: int) -> WorkerInputItem:
        seq_ctx_list = [item["seq_ctx"] for item in data_batches]
        label_list = [item["shifted_labels"] for item in data_batches]
        advantage_list = []
        for item in data_batches:
            advantages = item["advantages"].reshape(1, -1)
            advantage_list.append(advantages)

        rollout_logprobs_list = [
            item["rollout_logprobs"] if self.data_batch_properties["has_rollout_logprobs"] else None
            for item in data_batches
        ]
        seqlen_list = self._get_seqlen_from_data_batches(data_batches)
        cur_length = sum(seqlen_list)
        padding_len = pack_max_length - cur_length

        padding_item: WorkerInputItem | None = None
        if padding_len > 0:
            padding_item = self._create_padding_item(padding_len, pack_max_length)
            seq_ctx_list.append(padding_item["seq_ctx"])
            label_list.append(padding_item["shifted_labels"])
            advantage_list.append(padding_item["advantages"])
            rollout_logprobs_list.append(padding_item["rollout_logprobs"])

        packed_seq_ctx = SequenceContext.cat(seq_ctx_list)
        packed_shifted_labels = torch.cat(label_list, dim=1)  # type: ignore[arg-type]
        packed_shifted_labels = cast(torch.LongTensor, packed_shifted_labels)
        packed_advantages = torch.cat(advantage_list, dim=1)
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
        packed_input_ids = cast(torch.Tensor, packed_seq_ctx.input_ids)
        assert packed_input_ids.numel() == pack_max_length, (
            f"Packed seq ctx length {packed_input_ids.numel()} does not match pack_max_length {pack_max_length}"
            f"padding input_ids length: {padding_item['seq_ctx'].input_ids.shape if padding_item else 0}"  # type: ignore[union-attr]
        )
        assert packed_seq_ctx.num_padding == (packed_advantages == -100).sum().item(), (
            f"Packed seq ctx num_padding {packed_seq_ctx.num_padding} and packed advantages num_padding "
            f"{(packed_advantages != -100).sum().item()} mismatch after packing."
        )
        return optimizer_step_packs

    def _create_padding_item(
        self,
        pad_len: int,
        pack_max_length: int,
    ) -> WorkerInputItem:
        # padding input_ids
        self.padding_tokens += pad_len
        pad_tokens = tuple(
            torch.zeros(1, self.split_size, dtype=torch.long, device="cpu") for _ in range(pad_len // self.split_size)
        )
        if pad_len % self.split_size > 0:
            pad_tokens = pad_tokens + (torch.zeros(1, pad_len % self.split_size, dtype=torch.long, device="cpu"),)
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

        pad_advantage = torch.full((1, pad_len), -100, dtype=torch.float32, device="cpu")
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
