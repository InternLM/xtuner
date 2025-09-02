# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from .utils import pad_to_multiple_of, split_for_sequence_parallel


@dataclass
class SequenceContext:
    """Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    # TODO(HHA): 在仅计算 loss 情况下或者多模态情况下，input_ids 其实应该都可以为 None，不够通用
    input_ids: torch.LongTensor  # shape (1, seq_len)
    cu_seq_lens_q: torch.IntTensor
    cu_seq_lens_k: torch.IntTensor
    max_length_q: int
    max_length_k: int
    num_padding: int = 0
    sequence_parallel_mesh: DeviceMesh | None = None
    block_table: torch.Tensor | None = None
    device: str | torch.device = "cpu"  # TODO: 这个地方有点乱，到处是 device
    position_ids: torch.LongTensor | None = None

    # Intern-S1
    image_flags: torch.LongTensor | None = None

    # vllm model
    pixel_values: torch.FloatTensor | None = None
    inputs_embeds: torch.FloatTensor | None = None
    num_img_tokens: list[int] | None = None

    def __post_init__(self):
        if self.position_ids is None:
            seq_lens_k = self.cu_seq_lens_k[1:] - self.cu_seq_lens_k[:-1]
            seq_lens_q = self.cu_seq_lens_q[1:] - self.cu_seq_lens_q[:-1]

            _position_ids = [torch.arange(k - q, k) for q, k in zip(seq_lens_q, seq_lens_k)]
            position_ids = torch.cat(_position_ids).unsqueeze(0).to(self.cu_seq_lens_k.device)

            if self.sequence_parallel_mesh is not None:
                position_ids = split_for_sequence_parallel(position_ids, dim=1, sp_mesh=self.sequence_parallel_mesh)

            self.position_ids = position_ids

    @classmethod
    def from_input_ids(
        cls,
        input_ids: tuple[torch.LongTensor, ...],
        block_table: torch.Tensor | None = None,
        sp_mesh: DeviceMesh | None = None,
        device: str = "cuda",
    ) -> Self:
        assert isinstance(input_ids, (list, tuple))
        for ids in input_ids:
            assert ids.shape[0] == 1, "input_ids must have batch size of 1"
        num_tokens = [x.numel() for x in input_ids]

        cu_seq_lens = cast(torch.IntTensor, torch.cumsum(torch.LongTensor([0] + num_tokens), dim=0).to(device).int())
        return cls(
            input_ids=cast(torch.LongTensor, torch.cat(input_ids, dim=1).to(device)),
            cu_seq_lens_k=cu_seq_lens,
            cu_seq_lens_q=cu_seq_lens,
            max_length_q=cast(int, (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()),
            max_length_k=cast(int, (cu_seq_lens[1:] - cu_seq_lens[:-1]).max().item()),
            block_table=block_table,
            sequence_parallel_mesh=sp_mesh,
            device=device,
        )

    def split(self, sequence_parallel_mesh: DeviceMesh | None = None) -> Self:
        if sequence_parallel_mesh is None:
            sequence_parallel_mesh = self.sequence_parallel_mesh
        self.sequence_parallel_mesh = sequence_parallel_mesh

        if sequence_parallel_mesh is None:
            return self

        multiple_of = sequence_parallel_mesh.size()
        if self.input_ids is not None:
            pad_input_ids = pad_to_multiple_of(self.input_ids, 0, multiple_of, 1)
            sp_input_ids = cast(
                torch.LongTensor, split_for_sequence_parallel(pad_input_ids, dim=1, sp_mesh=sequence_parallel_mesh)
            )
            new_padding = pad_input_ids.numel() - self.input_ids.numel()
            if new_padding > 0:
                if self.num_padding > 0:
                    new_cu_seq_lens = self.cu_seq_lens_q.clone()
                    new_cu_seq_lens[-1] += new_padding
                else:
                    new_cu_seq_lens = torch.ones(self.cu_seq_lens_q.numel() + 1, dtype=torch.int32, device=self.device)
                    new_cu_seq_lens[: self.cu_seq_lens_q.numel()] = self.cu_seq_lens_q.clone()
                    new_cu_seq_lens[-1] = self.cu_seq_lens_q[-1] + new_padding
            else:
                new_cu_seq_lens = self.cu_seq_lens_q.clone()
            new_cu_seq_lens = cast(torch.IntTensor, new_cu_seq_lens)
            new_max_length = cast(int, max(self.seq_lens_q.max().item(), new_padding))
            num_non_padding = self.input_ids.shape[1] - self.num_padding
            start = sp_input_ids.shape[1] * sequence_parallel_mesh.get_local_rank()
            end = start + sp_input_ids.shape[1]
            sp_num_padding = max(0, min(sp_input_ids.shape[1], end - num_non_padding))

            sp_seq_ctx = self.__class__(
                input_ids=sp_input_ids,
                cu_seq_lens_q=new_cu_seq_lens,
                cu_seq_lens_k=new_cu_seq_lens,
                max_length_q=new_max_length,
                max_length_k=new_max_length,
                num_padding=sp_num_padding,
                block_table=self.block_table,
                device=sp_input_ids.device,
                sequence_parallel_mesh=sequence_parallel_mesh,
                # TODO: 没有 copy 方法比较难受,容易漏掉变量
                image_flags=self.image_flags,
                pixel_values=self.pixel_values,
            )
            return sp_seq_ctx
        else:
            return self

    @classmethod
    def pack(cls, sequence_context_list: list["SequenceContext"]):
        packed_input_ids: list[torch.Tensor] = []
        cu_seq_lens_q: list[torch.IntTensor] = []
        cu_seq_lens_k: list[torch.IntTensor] = []
        max_length_q = 0
        max_length_k = 0
        num_padding = 0
        device = []
        inputs_embeds = []
        for seq_ctx in sequence_context_list:
            assert seq_ctx.sequence_parallel_mesh is None
            # todo: support vlm model
            assert seq_ctx.pixel_values is None
            packed_input_ids.append(seq_ctx.input_ids)
            cu_seq_lens_q.append(
                seq_ctx.cu_seq_lens_q  # type: ignore
                if len(cu_seq_lens_q) == 0
                else (seq_ctx.cu_seq_lens_q + cu_seq_lens_q[-1][-1])[1:]
            )
            cu_seq_lens_k.append(
                seq_ctx.cu_seq_lens_k  # type: ignore
                if len(cu_seq_lens_k) == 0
                else (seq_ctx.cu_seq_lens_k + cu_seq_lens_k[-1][-1])[1:]
            )
            max_length_q = max(max_length_q, seq_ctx.max_length_q)
            max_length_k = max(max_length_k, seq_ctx.max_length_k)
            num_padding += seq_ctx.num_padding
            device.append(torch.device(seq_ctx.device))
            if seq_ctx.inputs_embeds is not None:
                inputs_embeds.append(seq_ctx.inputs_embeds)
        assert len(set(device)) == 1, f"All sequence contexts must be on the same device. Got {set(device)}"

        return cls(
            input_ids=torch.cat(packed_input_ids, dim=1),  # type: ignore
            cu_seq_lens_q=torch.cat(cu_seq_lens_q, dim=0),  # type: ignore
            cu_seq_lens_k=torch.cat(cu_seq_lens_k, dim=0),  # type: ignore
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            num_padding=num_padding,
            device=device[0],
            inputs_embeds=torch.cat(inputs_embeds, dim=1) if inputs_embeds else None,  # type: ignore
        )

    @property
    def mask(self) -> torch.BoolTensor:
        mask: torch.BoolTensor
        if self.input_ids is not None:
            mask = cast(torch.BoolTensor, torch.ones_like(self.input_ids, dtype=torch.bool))
        else:
            mask = cast(torch.BoolTensor, torch.ones_like(self.inputs_embeds[..., 0], dtype=torch.bool))
        if self.num_padding > 0:
            mask[..., -self.num_padding :] = False
        return mask

    @property
    def seq_lens_q(self) -> torch.LongTensor:
        return self.cu_seq_lens_q[1:] - self.cu_seq_lens_q[:-1]  # type: ignore

    @property
    def seq_lens_k(self) -> torch.LongTensor:
        return self.cu_seq_lens_k[1:] - self.cu_seq_lens_k[:-1]  # type: ignore

    # TODO: 暂时没有用到，可能要删掉
    def chunk(self, num_chunks: int) -> list[Self]:
        n = self.seq_lens_q.numel()
        assert n // num_chunks
        n_per_chunk = n // num_chunks

        q_lens_chunks = torch.chunk(self.seq_lens_q, chunks=num_chunks, dim=0)
        k_lens_chunks = torch.chunk(self.seq_lens_k, chunks=num_chunks, dim=0)

        lens_per_chunk = [chunk.sum() for chunk in q_lens_chunks]
        input_ids_chunks = torch.split(self.input_ids, lens_per_chunk, dim=1)  # type: ignore

        attn_meta_list: list[Self] = []
        for i in range(num_chunks):
            if self.block_table:
                block_table = self.block_table[i * n_per_chunk : (i + 1) * n_per_chunk]
            else:
                block_table = None
            # fmt: off
            _meta = self.__class__(
                input_ids=input_ids_chunks[i],  # type: ignore
                cu_seq_lens_q=self.cu_seq_lens_q[i * n_per_chunk : (i + 1) * n_per_chunk + 1] - self.cu_seq_lens_q[i * n_per_chunk],  # type: ignore
                cu_seq_lens_k=self.cu_seq_lens_k[i * n_per_chunk : (i + 1) * n_per_chunk + 1] - self.cu_seq_lens_k[i * n_per_chunk],  # type: ignore
                max_length_q=q_lens_chunks[i].max(), # type: ignore
                max_length_k=k_lens_chunks[i].max(), # type: ignore
                block_table=block_table,
                device=self.device,
                sequence_parallel_mesh=self.sequence_parallel_mesh,
            )
            # fmt: on
            attn_meta_list.append(_meta)
        return attn_meta_list

    def set_sp_mesh(self, sp_mesh: DeviceMesh) -> Self:
        """Set the sequence parallel mesh."""
        self.sequence_parallel_mesh = sp_mesh
        return self

    def to(self, device: torch.device | str):
        """Move all tensors in the context to the specified device.

        Args:
            device: The target device to move tensors to.

        Returns:
            Self: The context with tensors moved to the target device.
        """
        self.input_ids = self.input_ids.to(device)  # type: ignore
        self.cu_seq_lens_q = self.cu_seq_lens_q.to(device)  # type: ignore
        self.cu_seq_lens_k = self.cu_seq_lens_k.to(device)  # type: ignore

        if self.position_ids is not None and hasattr(self.position_ids, "to"):
            self.position_ids = self.position_ids.to(device)  # type: ignore

        if self.block_table is not None and hasattr(self.block_table, "to"):
            self.block_table = self.block_table.to(device)  # type: ignore

        if self.image_flags is not None and hasattr(self.image_flags, "to"):
            self.image_flags = self.image_flags.to(device)  # type: ignore

        if self.pixel_values is not None and hasattr(self.pixel_values, "to"):
            self.pixel_values = self.pixel_values.to(device)  # type: ignore

        if self.inputs_embeds is not None and hasattr(self.inputs_embeds, "to"):
            self.inputs_embeds = self.inputs_embeds.to(device)  # type: ignore

        self.device = device

        return self
