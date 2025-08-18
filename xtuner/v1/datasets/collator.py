from typing import TypedDict

import torch

from xtuner.utils import IGNORE_INDEX
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import get_logger
from xtuner.v1.utils.pad import pad_to_max_length

from .data_item import DataItem, InternS1DataItem


logger = get_logger()


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    labels: torch.Tensor


def fake_collator(instances: list[DataItem], **kwargs):
    return instances


def sft_llm_collator(
    instances: list[list[DataItem]], pack_max_length: int, padding_token_idx: int
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_lenghth
        if (total_num_tokens := sum(i["num_tokens"] for i in instance)) > pack_max_length:
            logger.warning(
                f"Found packed sample with {total_num_tokens} tokens, which is larger than the `pack_max_length`"
                f"{pack_max_length}, which is unexpected for packed dataset. dropping samples from the end."
            )

            for drop_from in range(len(instance) - 1, -1, -1):
                if total_num_tokens - instance[drop_from]["num_tokens"] <= pack_max_length:
                    if drop_from != 0:
                        instance = instance[:drop_from]
                    else:
                        data_item = instance[0]
                        data_item["input_ids"] = data_item["input_ids"][:pack_max_length]
                        data_item["labels"] = data_item["labels"][:pack_max_length]
                        data_item["num_tokens"] = len(data_item["input_ids"])
                        instance = [data_item]
                    break
                else:
                    total_num_tokens -= instance[drop_from]["num_tokens"]

        input_ids = torch.cat([torch.tensor(i["input_ids"]).view(1, -1) for i in instance], dim=-1)
        labels = torch.cat([torch.tensor(i["labels"]).view(1, -1) for i in instance], dim=-1)
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        num_tokens = [i["num_tokens"] for i in instance]
        if num_tokens[-1] == 1:
            num_tokens = num_tokens[:-1]  # remove the last sample if it is a single token
        else:
            num_tokens[-1] -= 1  # remove the last token if it is not a single token

        assert input_ids.shape == labels.shape, f"input_ids shape {input_ids.shape} != labels shape {labels.shape}"

        pad_len = pack_max_length - input_ids.shape[-1]

        if pad_len > 0:
            input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
            labels = pad_to_max_length(labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
            num_tokens = [0] + num_tokens + [pad_len]

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + [i["num_tokens"] for i in instance]

        cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

        seq_ctx = SequenceContext(
            input_ids=input_ids,  # type: ignore
            cu_seq_lens_q=cu_seq_lens,  # type: ignore
            cu_seq_lens_k=cu_seq_lens,  # type: ignore
            max_length_q=max(num_tokens),
            max_length_k=max(num_tokens),
            num_padding=pad_len,
        )
        ret.append(
            {
                "seq_ctx": seq_ctx,
                "labels": labels,
            }
        )

    return ret


def sft_vllm_collator(
    instances: list[list[InternS1DataItem]], pack_max_length: int, padding_token_idx: int
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_lenghth
        if (total_num_tokens := sum(i["num_tokens"] for i in instance)) > pack_max_length:
            logger.warning(
                f"Found packed sample with {total_num_tokens} tokens, which is larger than the `pack_max_length`"
                f"{pack_max_length}, which is unexpected for packed dataset. dropping samples from the end."
            )

            for drop_from in range(len(instance) - 1, -1, -1):
                if total_num_tokens - instance[drop_from]["num_tokens"] <= pack_max_length:
                    instance = instance[:drop_from]
                    break
                else:
                    total_num_tokens -= instance[drop_from]["num_tokens"]

        input_ids = torch.cat([torch.tensor(i["input_ids"]).view(1, -1) for i in instance], dim=-1)
        labels = torch.cat([torch.tensor(i["labels"]).view(1, -1) for i in instance], dim=-1)
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        num_tokens = [i["num_tokens"] for i in instance]
        if num_tokens[-1] == 1:
            num_tokens = num_tokens[:-1]  # remove the last sample if it is a single token
        else:
            num_tokens[-1] -= 1  # remove the last token if it is not a single token

        assert input_ids.shape == labels.shape, f"input_ids shape {input_ids.shape} != labels shape {labels.shape}"

        pad_len = pack_max_length - input_ids.shape[-1]

        if pad_len > 0:
            input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
            labels = pad_to_max_length(labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
            num_tokens = [0] + num_tokens + [pad_len]

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + [i["num_tokens"] for i in instance]

        cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

        num_img_tokens = []
        for data in instance:
            num_img_tokens.extend(data["num_img_tokens"])

        pixel_values = torch.cat([i["pixel_values"] for i in instance], dim=0)

        image_flags: torch.LongTensor | None = None
        if "image_flags" in instance[0]:
            image_flags = torch.cat([i["image_flags"] for i in instance], dim=0)  # type: ignore

        seq_ctx = SequenceContext(
            input_ids=input_ids,  # type: ignore
            cu_seq_lens_q=cu_seq_lens,  # type: ignore
            cu_seq_lens_k=cu_seq_lens,  # type: ignore
            max_length_q=max(num_tokens),
            max_length_k=max(num_tokens),
            num_padding=pad_len,
            pixel_values=pixel_values,  # type: ignore
            image_flags=image_flags,
            num_img_tokens=num_img_tokens,
        )
        ret.append(
            {
                "seq_ctx": seq_ctx,
                "labels": labels,
            }
        )

    return ret
