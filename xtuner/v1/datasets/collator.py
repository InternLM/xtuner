from typing import Sequence

import torch
from typing_extensions import TypedDict

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import IGNORE_INDEX, get_logger
from xtuner.v1.utils.pad import pad_to_max_length

from .data_item import DataItem, Glm53VLDataItem, InternS1DataItem, QwenVL3DataItem


logger = get_logger()


class ColateItem(TypedDict):
    seq_ctx: SequenceContext
    shifted_labels: torch.Tensor


def split_to_chunks(n: int, chunk_size: int) -> list[int]:
    full, remainder = divmod(n, chunk_size)
    result = [chunk_size] * full
    if remainder > 0:
        result.append(remainder)
    return result


def fake_collator(instances: list[DataItem], **kwargs):
    return instances


def build_text_ctx_labels(
    instance: Sequence[DataItem] | DataItem,
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool,
    pad_chunk_size: int,
) -> tuple[SequenceContext, torch.Tensor, Sequence[DataItem]]:
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
        pad_chunk_tokens = split_to_chunks(pad_len, pad_chunk_size)
        num_tokens = [0] + num_tokens + pad_chunk_tokens

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
    return seq_ctx, shifted_labels, instance


def sft_llm_collator(
    instances: list[list[DataItem]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
    pad_chunk_size: int = 256,
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_length
        seq_ctx, shifted_labels, _ = build_text_ctx_labels(
            instance,
            pack_max_length,
            padding_token_idx,
            pack_to_max_length,
            pad_chunk_size,
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
    pad_chunk_size: int = 256,
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
            pad_chunk_tokens = split_to_chunks(pad_len, pad_chunk_size)
            num_tokens = [0] + num_tokens + pad_chunk_tokens

        elif pad_len < 0:
            raise ValueError(
                f"Internal Error! Packed sample length {input_ids.shape[-1]} is larger than"
                f"packed_max_lenghth {pack_max_length}. Please report the bug to xtuner"
            )
        else:
            num_tokens = [0] + num_tokens

        cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()

        num_img_tokens: list[list] = []
        for data in instance:
            num_img_tokens.append(data.get("num_img_tokens", [0]))

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
    pad_chunk_size: int = 256,
) -> list[ColateItem]:
    ret: list[ColateItem] = []
    for instance in instances:
        # If the token number of the packed sample is larger than the packed_max_length
        seq_ctx, shifted_labels, instance = build_text_ctx_labels(  # type: ignore
            instance,
            pack_max_length,
            padding_token_idx,
            pack_to_max_length,
            pad_chunk_size,
        )

        all_position_ids_none = all(
            "position_ids" not in _instance or _instance["position_ids"] is None for _instance in instance
        )

        position_ids_list = []
        if not all_position_ids_none:
            for _instance in instance:
                if "position_ids" in _instance and _instance["position_ids"] is not None:
                    position_ids_ = _instance["position_ids"][..., :pack_max_length]
                    position_ids_list.append(position_ids_)
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

        position_ids: torch.Tensor | None = None
        if len(position_ids_list) > 0:
            position_ids = torch.cat(position_ids_list, dim=-1)
            position_ids = position_ids[:, :, :-1]
            if pack_to_max_length and pack_max_length - position_ids.shape[-1] > 0:
                position_ids = pad_to_max_length(position_ids, 0, max_length=pack_max_length, dim=-1)

        num_img_tokens: list[list[int]] = []
        for data in instance:
            num_img_tokens.append(data.get("num_img_tokens", [0]))

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

        if position_ids is not None:
            seq_ctx.position_ids = position_ids  # type: ignore

        seq_ctx.pixel_values = pixel_values  # type: ignore
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.num_img_tokens = num_img_tokens

        if position_ids is not None:
            assert seq_ctx.input_ids is not None, "input_ids should not be None"
            assert position_ids.shape[-1] == seq_ctx.input_ids.shape[-1], (
                f"position_ids length {position_ids.shape[-1]} != input_ids length {seq_ctx.input_ids.shape[-1]}"
            )
        ret.append(
            {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
            }
        )

    return ret


def glm53_vl_sft_collator(
    instances: list[list[Glm53VLDataItem]],
    pack_max_length: int,
    padding_token_idx: int,
    image_token_id: int,
    merge_unit: int,
    pack_to_max_length: bool = True,
    pad_chunk_size: int = 256,
    media_dtype: torch.dtype = torch.bfloat16,
) -> list[ColateItem]:
    """Collate GLM-5.3 packs into text contexts and concatenated media.

    Args:
        instances (list[list[Glm53VLDataItem]]): Tokenized items, one inner list per pack.
        pack_max_length (int): Maximum packed token length.
        padding_token_idx (int): Padding token id for ``input_ids``.
        image_token_id (int): Image placeholder token id, re-checked against the patch
            count after pack-level truncation.
        merge_unit (int): Vision tower spatial-merge unit (``spatial_merge_size ** 2``).
        pack_to_max_length (bool): Pad every pack up to ``pack_max_length``.
        pad_chunk_size (int): Padding granularity when ``pack_to_max_length`` is off.
        media_dtype (torch.dtype): Transport dtype applied element-wise to image/video
            patches before the pack-level concat (worker side, pre-IPC). Keep it equal to
            the vision tower's parameter dtype (bf16 by default); a lower-precision value
            under an fp32 tower rounds the patches at pack time.

    Returns:
        list[ColateItem]: One ``{"seq_ctx", "shifted_labels"}`` dict per pack.
    """
    ret: list[ColateItem] = []
    for instance in instances:
        # 占位符计数与媒体特征不符的样本（纯文本含字面 <|image|>、cache/runtime 计数漂移等）
        # 会在模型 splice 时计数错位崩溃；打包前丢弃并告警，不让单条异常样本中断训练——
        # 线上 256 卡 Step 8 挂掉的直接原因就是此处原来的逐样本 assert。pack 末尾用 padding
        # 补齐，丢样本不改变序列长度，DP 各 rank 的 collective 序列保持同步。
        kept_instance: list[Glm53VLDataItem] = []
        for data in instance:
            placeholder_count = data["input_ids"].count(image_token_id)
            expected_count = sum(data.get("num_img_tokens", [])) // merge_unit
            if placeholder_count != expected_count:
                logger.warning(
                    f"GLM-5.3-Flash visual placeholder count {placeholder_count} != expected {expected_count}; "
                    f"discard sample (num_tokens={data.get('num_tokens')}) from this pack instead of "
                    "halting training."
                )
                continue
            kept_instance.append(data)
        if not kept_instance:
            # 单样本 pack 是正常情况（长样本独占一个 pack、长度分组尾部等），整包被滤只说明
            # 这个 pack 恰好没有好样本，推不出系统性故障；此时 raise 会让单条脏样本中断训练
            # （E02 回归）。退回无 loss 占位项，照常走 build_text_ctx_labels：pack_to_max_length
            # 补 padding 保持序列长度，labels 全 IGNORE 对 loss 贡献为 0，与 runtime 通道
            # _discard_damaged_sample 的占位语义一致。系统性数据/代码问题由数据集级统计与
            # 配置校验暴露，不在此处报错。
            no_loss_placeholder: Glm53VLDataItem = {
                "input_ids": [padding_token_idx] * 2,
                "labels": [IGNORE_INDEX] * 2,
                "num_tokens": 2,
                "num_imgs": [0],
                "num_img_tokens": [0],
                "mm_token_type_ids": torch.zeros((1, 2), dtype=torch.long),
            }
            kept_instance = [no_loss_placeholder]
        seq_ctx, shifted_labels, instance = build_text_ctx_labels(  # type: ignore
            kept_instance,
            pack_max_length,
            padding_token_idx,
            pack_to_max_length,
            pad_chunk_size,
        )

        num_img_tokens: list[list[int]] = [data.get("num_img_tokens", [0]) for data in instance]

        # Cast image/video patches to the transport dtype (default bf16) on the worker side,
        # before cat and the cross-process IPC: the vision tower's patch_embed casts patches to
        # its parameter dtype anyway, so fp32 media would only double the bytes crossing shm,
        # main-process residency, and H2D. No-op when the dtypes already match.
        pixel_values: list | torch.Tensor | None
        pixel_values = [i["pixel_values"].to(media_dtype) for i in instance if "pixel_values" in i]
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

        image_grid_thw: list | torch.Tensor | None
        image_grid_thw = [i["image_grid_thw"] for i in instance if "image_grid_thw" in i]
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None

        pixel_values_videos: list | torch.Tensor | None
        pixel_values_videos = [
            i["pixel_values_videos"].to(media_dtype) for i in instance if "pixel_values_videos" in i
        ]
        pixel_values_videos = torch.cat(pixel_values_videos, dim=0) if pixel_values_videos else None

        video_grid_thw: list | torch.Tensor | None
        video_grid_thw = [i["video_grid_thw"] for i in instance if "video_grid_thw" in i]
        video_grid_thw = torch.cat(video_grid_thw, dim=0) if video_grid_thw else None

        # mm_token_type_ids must track input_ids through the exact same [:, :-1] drop + pad
        # (design doc §8.3). Slice each item to its own (possibly pack-truncated) input_ids
        # length first, so the rare single-oversized-item edge case in `build_text_ctx_labels`
        # (which truncates that item's input_ids/labels but knows nothing about this
        # GLM53-specific field) can't leave a stale, longer mm_token_type_ids behind.
        mm_token_type_ids = torch.cat([i["mm_token_type_ids"][:, : len(i["input_ids"])] for i in instance], dim=-1)
        mm_token_type_ids = mm_token_type_ids[:, :-1]
        if seq_ctx.num_padding > 0:
            mm_token_type_ids = pad_to_max_length(mm_token_type_ids, 0, max_length=pack_max_length, dim=-1)

        assert seq_ctx.input_ids is not None, "input_ids should not be None"
        assert mm_token_type_ids.shape[-1] == seq_ctx.input_ids.shape[-1], (
            f"mm_token_type_ids length {mm_token_type_ids.shape[-1]} != input_ids length {seq_ctx.input_ids.shape[-1]}"
        )

        # Re-assert the placeholder<->patch-count invariant per sample after truncation: the
        # pre-filter above has already dropped samples that were corrupt on arrival, so a
        # failure here means build_text_ctx_labels's own pack-level truncation desynced media
        # from text — a packing bug, not dirty data, and it must stay loud instead of silently
        # training on half a visual span (design doc §8.4).
        for data in instance:
            placeholder_count = data["input_ids"].count(image_token_id)
            expected_count = sum(data.get("num_img_tokens", [])) // merge_unit
            assert placeholder_count == expected_count, (
                f"GLM-5.3-Flash visual placeholder count {placeholder_count} != expected {expected_count} "
                "after pack truncation; pack_max_length is too small to hold this sample's visual span."
            )

        seq_ctx.pixel_values = pixel_values  # type: ignore
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.pixel_values_videos = pixel_values_videos  # type: ignore
        seq_ctx.video_grid_thw = video_grid_thw
        seq_ctx.mm_token_type_ids = mm_token_type_ids
        seq_ctx.num_img_tokens = num_img_tokens

        ret.append(
            {
                "seq_ctx": seq_ctx,
                "shifted_labels": shifted_labels,
            }
        )

    return ret
