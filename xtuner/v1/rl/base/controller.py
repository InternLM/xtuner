import math
from typing import Literal, TypedDict

import ray
import torch
from ray.actor import ActorProxy

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import ray_method

from .worker import TrainingWorker, WorkerLogItem


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor
    advantage: float
    rollout_logprobs: torch.Tensor | None


class RawTrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers

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

            seq_ctx = SequenceContext.cat(seq_ctx_list)
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

    @ray_method
    def fit(self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int) -> list[WorkerLogItem]:
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
            pad_data = {
                "seq_ctx": pad_seq_ctx,
                "shifted_labels": pad_shifted_labels,
                "advantages": pad_advantages,
                "rollout_logprobs": pad_rollout_logprobs,
            }
            pad_data_samples = [pad_data for _ in range(pad_num)]
            packed_data_batches = packed_data_batches + pad_data_samples

        print(f"len(packed_data_batches): {len(packed_data_batches)}")

        handles = []
        for worker_idx, worker in enumerate(self.workers):
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=packed_data_batches[(worker_idx // data_replicate_size) :: dp_size],
                    rollout_idx=rollout_idx,
                )
            )
        log_infos = ray.get(handles)
        return log_infos

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
    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles)
        return

    @ray_method
    def ready(self) -> bool:
        return True


TrainingController = ray.remote(RawTrainingController)
TrainingControllerProxy = ActorProxy[RawTrainingController]
