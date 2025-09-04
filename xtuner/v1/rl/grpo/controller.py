# import math
# from typing import Literal

# import ray
# import torch
# from typing_extensions import TypedDict

# from xtuner.v1.data_proto.sequence_context import SequenceContext

# from .worker import GRPOTrainingWorker


# class ColateItem(TypedDict):
#     seq_ctx: SequenceContext
#     shifted_labels: torch.Tensor
#     advantage: float


# class TrainingController:
#     def __init__(self, workers: list[GRPOTrainingWorker]) -> None:
#         self.workers = workers

#     def _get_pack_infos(self, dataset, num_tokens, target, random=None):
#         inds = list(range(len(dataset)))
#         if random is not None:
#             random.shuffle(inds)

#         item_buffer = []
#         length_buffer = []
#         longest = 0

#         pack_infos = []
#         for shfl_i in inds:
#             if num_tokens[shfl_i] + sum(length_buffer) <= target:
#                 item_buffer.append(shfl_i)
#                 length_buffer.append(num_tokens[shfl_i])
#                 longest = max(longest, num_tokens[shfl_i])
#             else:
#                 if len(item_buffer) > 0:
#                     info = {
#                         "indices": item_buffer,
#                         "longest": int(longest),
#                     }
#                     pack_infos.append(info)

#                 item_buffer = [shfl_i]
#                 length_buffer = [num_tokens[shfl_i]]
#                 longest = num_tokens[shfl_i]

#         if len(item_buffer) > 0:
#             info = {
#                 "indices": item_buffer,
#                 "longest": int(longest),
#             }

#             pack_infos.append(info)

#         return pack_infos

#     def _packing(self, data_batches, pack_max_length):
#         pack_infos = self._get_pack_infos(
#             data_batches,
#             [data["seq_ctx"].input_ids.numel() for data in data_batches],
#             pack_max_length,
#         )
#         packed_data_batches = []
#         for pack_info in pack_infos:
#             indices = pack_info["indices"]
#             total_len = sum([data_batches[i]["seq_ctx"].input_ids.shape[1] for i in indices])
#             pad_len = pack_max_length - total_len
#             seq_ctx_list = [data_batches[i]["seq_ctx"] for i in indices]
#             label_list = [data_batches[i]["shifted_labels"] for i in indices]
#             advantage_list = [data_batches[i]["advantage"] for i in indices]
#             if pad_len > 0:
#                 # Reduce the attn calculation time by using multiple short sequence packs
#                 pad_tokens = tuple(
#                     torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
#                     for _ in range(pad_len // 1024)
#                 )
#                 if pad_len % 1024 > 0:
#                     pad_tokens = pad_tokens + (
#                         torch.zeros(1, pad_len % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"),
#                     )
#                 pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")
#                 pad_seq_ctx.num_padding = pad_len
#                 pad_labels = torch.full(
#                     (1, pad_len),
#                     -100,
#                     dtype=data_batches[0]["shifted_labels"].dtype,
#                     device=data_batches[0]["shifted_labels"].device,
#                 )
#                 seq_ctx_list.append(pad_seq_ctx)
#                 label_list.append(pad_labels)
#                 advantage_list.extend(
#                     [-100] * math.ceil(pad_len / 1024)
#                 )  # can be any number, pad tokens are excluded from the calculation of the loss function.

#             seq_ctx = SequenceContext.pack(seq_ctx_list)
#             shifted_labels = torch.cat(label_list, dim=1)  # (1, max_len)
#             advantages = torch.tensor(advantage_list).float().unsqueeze(0)  # (1, num_samples)
#             cu_seq_lens_q = seq_ctx.cu_seq_lens_q
#             num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
#             advantages = torch.repeat_interleave(advantages, num_tokens, dim=1)  # (1, max_len)

#             packed_data_batches.append(
#                 {
#                     "seq_ctx": seq_ctx,
#                     "shifted_labels": shifted_labels,
#                     "advantages": advantages,
#                 }
#             )
#         return packed_data_batches

#     def _grouped_by_max_length(self, packed_data_batches):
#         # sort 过后可能第一个 batch 会有很多 pad tokens，因为最后一个 pack 可能只有少量真实数据。
#         # 比如组成了 16 个 pack，第 16 个 pack 可能只有几条真实数据，剩下的都是 pad tokens。
#         # 排序后这条 pack 会被放在最前面，导致 rank0 的第一个 step 消耗的有效 token 数往往少于其他 rank，是正常现象。
#         return sorted(packed_data_batches, key=lambda x: x["seq_ctx"].max_length_q, reverse=True)

#     def fit(self, data_batches: list[ColateItem], pack_max_length: int, rollout_idx: int):
#         packed_data_batches = self._packing(data_batches, pack_max_length)
#         packed_data_batches = self._grouped_by_max_length(packed_data_batches)

#         # todo: support round up
#         num_packed_data_batches = len(packed_data_batches)
#         data_replicate_size = ray.get(self.workers[0].get_data_replicate_size.remote())  # type: ignore[attr-defined]
#         dp_size = len(self.workers) // data_replicate_size
#         pad_num = math.ceil(num_packed_data_batches / dp_size) * dp_size - num_packed_data_batches
#         if pad_num > 0:
#             # Reduce the attn calculation time by using multiple short sequence packs
#             pad_tokens = tuple(
#                 torch.zeros(1, 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu")
#                 for _ in range(pack_max_length // 1024)
#             )
#             if pack_max_length % 1024 > 0:
#                 pad_tokens = pad_tokens + (
#                     torch.zeros(
#                         1, pack_max_length % 1024, dtype=data_batches[0]["seq_ctx"].input_ids.dtype, device="cpu"
#                     ),
#                 )
#             pad_seq_ctx = SequenceContext.from_input_ids(pad_tokens, device="cpu")  # type: ignore
#             pad_seq_ctx.num_padding = pack_max_length
#             pad_shifted_labels = torch.full(
#                 (1, pack_max_length),
#                 -100,
#                 dtype=packed_data_batches[0]["shifted_labels"].dtype,
#                 device="cpu",
#             )
#             pad_advantages = torch.full(
#                 (1, pack_max_length),
#                 -100,
#                 dtype=packed_data_batches[0]["advantages"].dtype,
#                 device="cpu",
#             )
#             pad_data = {
#                 "seq_ctx": pad_seq_ctx,
#                 "shifted_labels": pad_shifted_labels,
#                 "advantages": pad_advantages,
#             }
#             pad_data_samples = [pad_data for _ in range(pad_num)]
#             packed_data_batches = packed_data_batches + pad_data_samples

#         print(f"len(packed_data_batches): {len(packed_data_batches)}")

#         handles = []
#         for worker_idx, worker in enumerate(self.workers):
#             handles.append(
#                 worker.fit.remote(  # type: ignore[attr-defined]
#                     data_batches=packed_data_batches[(worker_idx // data_replicate_size) :: dp_size],
#                     rollout_idx=rollout_idx,
#                 )
#             )
#         ray.get(handles)

#     def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
#         if target == "model":
#             ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
#         elif target == "optimizer":
#             ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
#         elif target == "all":
#             ray.get([worker.offload_model.remote() for worker in self.workers])  # type: ignore
#             ray.get([worker.offload_optimizer.remote() for worker in self.workers])  # type: ignore
#         return

#     def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
#         """Onload the model or optimizer of the training workers."""
#         if target == "model":
#             ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
#         elif target == "optimizer":
#             ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
#         elif target == "all":
#             ray.get([worker.onload_model.remote() for worker in self.workers])  # type: ignore
#             ray.get([worker.onload_optimizer.remote() for worker in self.workers])  # type: ignore
#         return

#     def update_rollout_info(self, info_dict):
#         ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

#     def update_weights(self):
#         """Update the weights of the training workers."""
#         handles = [worker.update_weights.remote() for worker in self.workers]
#         ray.get(handles)
#         return

#     def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
#         ray.get(self.workers[0].save_hf.remote(hf_dir, save_dtype))  # type: ignore
#         return


# @ray.remote
# class GRPOTrainingController(TrainingController):
#     pass
