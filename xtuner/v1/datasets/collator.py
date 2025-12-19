import torch
from typing_extensions import TypedDict

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import IGNORE_INDEX, get_logger
from xtuner.v1.utils.pad import pad_to_max_length

from .data_item import DataItem, InternS1DataItem, QwenVL3DataItem


logger = get_logger()


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor


def fake_collator(instances: list[DataItem], **kwargs):
    return instances


def sft_llm_collator(
    instances: list[list[DataItem]], pack_max_length: int, padding_token_idx: int, pack_to_max_length: bool = True
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_length
        if isinstance(instance, dict):
            instance = [instance]

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
        shifted_labels = labels[:, 1:]
        num_tokens = [i["num_tokens"] for i in instance]
        if num_tokens[-1] == 1:
            num_tokens = num_tokens[:-1]  # remove the last sample if it is a single token
        else:
            num_tokens[-1] -= 1  # remove the last token if it is not a single token

        assert input_ids.shape == shifted_labels.shape, (
            f"input_ids shape {input_ids.shape} != shifted_labels shape {shifted_labels.shape}"
        )
        if pack_to_max_length:
            pad_len = pack_max_length - input_ids.shape[-1]
        else:
            pad_len = 0

        if pad_len > 0:
            input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
            shifted_labels = pad_to_max_length(shifted_labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
            num_tokens = [0] + num_tokens + [pad_len]

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + num_tokens

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
                "shifted_labels": shifted_labels,
            }
        )

    return ret


def intern_s1_vl_sft_collator(
    instances: list[list[InternS1DataItem]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_length
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
        shifted_labels = labels[:, 1:]
        num_tokens = [i["num_tokens"] for i in instance]
        if num_tokens[-1] == 1:
            num_tokens = num_tokens[:-1]  # remove the last sample if it is a single token
        else:
            num_tokens[-1] -= 1  # remove the last token if it is not a single token

        assert input_ids.shape == shifted_labels.shape, (
            f"input_ids shape {input_ids.shape} != shifted_labels shape {shifted_labels.shape}"
        )

        if pack_to_max_length:
            pad_len = pack_max_length - input_ids.shape[-1]
        else:
            pad_len = 0

        if pad_len > 0:
            input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
            shifted_labels = pad_to_max_length(shifted_labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
            num_tokens = [0] + num_tokens + [pad_len]

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + num_tokens

        cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

        num_img_tokens: list[int] = []
        for data in instance:
            num_img_tokens.extend(data.get("num_img_tokens", [0]))

        pixel_values: list | torch.Tensor | None
        pixel_values = [i["pixel_values"] for i in instance if "pixel_values" in i]
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        seq_ctx = SequenceContext(
            input_ids=input_ids,  # type: ignore
            cu_seq_lens_q=cu_seq_lens,  # type: ignore
            cu_seq_lens_k=cu_seq_lens,  # type: ignore
            max_length_q=max(num_tokens),
            max_length_k=max(num_tokens),
            num_padding=pad_len,
            pixel_values=pixel_values,  # type: ignore
            num_img_tokens=num_img_tokens,
        )
        ret.append(
            {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
            }
        )

    return ret


def qwen3_vl_sft_collator(
    instances: list[list[QwenVL3DataItem]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_length
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

        all_position_ids_none = all(
            "position_ids" not in _instance or _instance["position_ids"] is None for _instance in instance
        )
        position_ids_list = []
        if not all_position_ids_none:
            for _instance in instance:
                if "position_ids" in _instance and _instance["position_ids"] is not None:
                    position_ids_list.append(_instance["position_ids"])
                else:
                    position_ids_ = (
                        torch.arange(len(_instance["input_ids"]))
                        .view(1, 1, -1)
                        .expand(3, len(_instance["input_ids"]), -1)
                    )
                    position_ids_list.append(position_ids_)

        assert len(position_ids_list) == len(instance) or len(position_ids_list) == 0, (
            f"position_ids_list length {len(position_ids_list)} != instance length {len(instance)} or "
            f"position_ids_list is not empty"
        )

        input_ids = input_ids[:, :-1]
        shifted_labels = labels[:, 1:]

        position_ids: torch.Tensor | None = None
        if len(position_ids_list) > 0:
            position_ids = torch.cat(position_ids_list, dim=-1)
            position_ids = position_ids[:, :, :-1]

        num_tokens = [i["num_tokens"] for i in instance]
        if num_tokens[-1] == 1:
            num_tokens = num_tokens[:-1]  # remove the last sample if it is a single token
        else:
            num_tokens[-1] -= 1  # remove the last token if it is not a single token

        assert input_ids.shape == shifted_labels.shape, (
            f"input_ids shape {input_ids.shape} != shifted_labels shape {shifted_labels.shape}"
        )

        if pack_to_max_length:
            pad_len = pack_max_length - input_ids.shape[-1]
        else:
            pad_len = 0

        if pad_len > 0:
            input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
            shifted_labels = pad_to_max_length(shifted_labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
            if position_ids is not None:
                position_ids = pad_to_max_length(position_ids, 0, max_length=pack_max_length, dim=-1)
            num_tokens = [0] + num_tokens + [pad_len]

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + num_tokens

        cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

        num_img_tokens: list[int] = []
        for data in instance:
            num_img_tokens.extend(data.get("num_img_tokens", [0]))

        pixel_values: list | torch.Tensor | None
        pixel_values = [i["pixel_values"] for i in instance if "pixel_values" in i]
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None

        image_grid_thw: list | torch.Tensor | None
        image_grid_thw = [i["image_grid_thw"] for i in instance if "image_grid_thw" in i]
        if image_grid_thw:
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        else:
            image_grid_thw = None

        seq_ctx = SequenceContext(
            input_ids=input_ids,  # type: ignore
            cu_seq_lens_q=cu_seq_lens,  # type: ignore
            cu_seq_lens_k=cu_seq_lens,  # type: ignore
            max_length_q=max(num_tokens),
            max_length_k=max(num_tokens),
            num_padding=pad_len,
            position_ids=position_ids,  # type: ignore
            pixel_values=pixel_values,  # type: ignore
            image_grid_thw=image_grid_thw,
            num_img_tokens=num_img_tokens,
        )
        ret.append(
            {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
            }
        )

    return ret
