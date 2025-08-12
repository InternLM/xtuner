from typing import TypedDict

import ray
import torch
from ray.actor import ActorClass

from xtuner.v1.data_proto.sequence_context import SequenceContext


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shift_labels: torch.Tensor
    advantage: float


@ray.remote
class TrainingController:
    def __init__(self, workers: list[ActorClass]) -> None:
        self.workers = workers

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

    def _packing(self, data_batches, pack_max_length):
        pack_infos = self._get_pack_infos(
            data_batches,
            [data["seq_ctx"].input_ids.numel() for data in data_batches],
            pack_max_length,
        )
        packed_data_batches = []
        for pack_info in pack_infos:
            indices = pack_info["indices"]
            total_len = sum([data_batches[i]["seq_ctx"].input_ids.shape[1] for i in indices])
            pad_len = pack_max_length - total_len
            seq_ctx_list = [data_batches[i]["seq_ctx"] for i in indices]
            label_list = [data_batches[i]["shift_labels"] for i in indices]
            advantage_list = [data_batches[i]["advantage"] for i in indices]
            if pad_len > 0:
                pad_token_ids = torch.full(
                    (1, pad_len),
                    0,
                    dtype=data_batches[0]["seq_ctx"].input_ids.dtype,
                    device=data_batches[0]["seq_ctx"].input_ids.device,
                )
                pad_seq_ctx = SequenceContext.from_input_ids((pad_token_ids,), device=pad_token_ids.device)
                pad_seq_ctx.num_padding = pad_len
                pad_labels = torch.full(
                    (1, pad_len),
                    -100,
                    dtype=data_batches[0]["shift_labels"].dtype,
                    device=data_batches[0]["shift_labels"].device,
                )
                seq_ctx_list.append(pad_seq_ctx)
                label_list.append(pad_labels)
                advantage_list.append(
                    -100
                )  # can be any number, pad tokens are excluded from the calculation of the loss function.

            seq_ctx = SequenceContext.pack(seq_ctx_list)
            shift_labels = torch.cat(label_list, dim=1)  # (1, max_len)
            advantage = torch.tensor(advantage_list).float().unsqueeze(0)  # (1, num_samples)

            packed_data_batches.append(
                {
                    "seq_ctx": seq_ctx,
                    "shift_labels": shift_labels,
                    "advantage": advantage,
                }
            )
        return packed_data_batches

    def _grouped_by_max_length(self, packed_data_batches):
        return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

    def fit(self, data_batches: list[ColateItem], pack_max_length: int):
        packed_data_batches = self._packing(data_batches, pack_max_length)
        packed_data_batches = self._grouped_by_max_length(packed_data_batches)

        # todo: support round up
        num_packed_data_batches = len(packed_data_batches)
        data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())
        dp_size = len(self.workers) // data_replicate_size
        packed_data_batches = packed_data_batches[: (num_packed_data_batches // dp_size * dp_size)]

        print(f"len(packed_data_batches): {len(packed_data_batches)}")

        handles = []
        for worker_idx, worker in enumerate(self.workers):
            handles.append(
                worker.fit.remote(data_batches=packed_data_batches[(worker_idx // data_replicate_size) :: dp_size])
            )
        ray.get(handles)

    def offload(self):
        ray.get([worker.offload.remote() for worker in self.workers])
        return

    def onload(self):
        ray.get([worker.onload.remote() for worker in self.workers])
        return

    def update_rollout_info(self, info_dict):
        ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

    def update_weights(self):
        """Update the weights of the training workers."""
        handles = [worker.update_weights.remote() for worker in self.workers]
        ray.get(handles)
        return
