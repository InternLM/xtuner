# Copyright (c) OpenMMLab. All rights reserved.
"""
Data collator for DPO (Direct Preference Optimization) training.

This module provides collators for handling preference data (chosen/rejected pairs)
in the xtuner v1 framework. It follows the existing xtuner v1 collator design
pattern, outputting SequenceContext and shifted_labels.
"""

import torch
from typing import Any
from typing_extensions import TypedDict

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import IGNORE_INDEX, get_logger
from xtuner.v1.utils.pad import pad_to_max_length


logger = get_logger()


class DPOColateItem(TypedDict):
    """Output of DPO collator, containing both chosen and rejected sequences."""
    # Chosen sequence
    chosen_seq_ctx: SequenceContext
    chosen_shifted_labels: torch.Tensor
    # Rejected sequence
    rejected_seq_ctx: SequenceContext
    rejected_shifted_labels: torch.Tensor
    # Optional: precomputed reference log probabilities
    ref_chosen_logps: torch.Tensor | None
    ref_rejected_logps: torch.Tensor | None


def qwen3_vl_dpo_collator(
    instances: list[list[dict]],
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> list[DPOColateItem]:
    """
    Collate DPO preference data for Qwen3-VL models.
    
    This function handles the additional visual features (pixel_values,
    image_grid_thw, position_ids) required by Qwen3-VL models.
    
    Args:
        instances: List of batched data items with visual features.
        pack_max_length: Maximum sequence length for packing.
        padding_token_idx: Token ID to use for padding.
        pack_to_max_length: Whether to pad sequences to max_length.
    
    Returns:
        List of DPOColateItem with visual features included in SequenceContext.
    
    Example:
        >>> from functools import partial
        >>> collator = partial(qwen3_vl_dpo_collator, pack_max_length=4096, padding_token_idx=0)
        >>> # Use with dataloader
    """
    ret: list[DPOColateItem] = []
    
    for instance in instances:
        if isinstance(instance, dict):
            instance = [instance]
        
        # Process chosen with visual features
        chosen_result = _process_qwen3_vl_sequence(
            instance, "chosen", pack_max_length, padding_token_idx, pack_to_max_length
        )
        
        # Process rejected with visual features
        rejected_result = _process_qwen3_vl_sequence(
            instance, "rejected", pack_max_length, padding_token_idx, pack_to_max_length
        )
        
        # Handle precomputed ref logps
        ref_chosen_logps = None
        ref_rejected_logps = None
        if "ref_chosen_logps" in instance[0]:
            ref_chosen_logps = torch.tensor([instance[0]["ref_chosen_logps"]])
        if "ref_rejected_logps" in instance[0]:
            ref_rejected_logps = torch.tensor([instance[0]["ref_rejected_logps"]])
        
        ret.append({
            "chosen_seq_ctx": chosen_result["seq_ctx"],
            "chosen_shifted_labels": chosen_result["shifted_labels"],
            "rejected_seq_ctx": rejected_result["seq_ctx"],
            "rejected_shifted_labels": rejected_result["shifted_labels"],
            "ref_chosen_logps": ref_chosen_logps,
            "ref_rejected_logps": ref_rejected_logps,
        })
    
    return ret


def _process_qwen3_vl_sequence(
    instance: list[dict],
    seq_type: str,  # "chosen" or "rejected"
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool = True,
) -> dict[str, Any]:
    """
    Process a single Qwen3-VL sequence for DPO training.
    
    This handles the visual features specific to Qwen3-VL models.
    
    Args:
        instance: List of data items.
        seq_type: "chosen" or "rejected".
        pack_max_length: Maximum sequence length.
        padding_token_idx: Padding token ID.
        pack_to_max_length: Whether to pad to max length.
    
    Returns:
        Dictionary with seq_ctx (including visual features) and shifted_labels.
    """
    # Extract data for the specified sequence type
    data = []
    for item in instance:
        input_ids_key = f"{seq_type}_input_ids"
        labels_key = f"{seq_type}_labels"
        
        input_ids = item.get(input_ids_key, item.get("input_ids", []))
        labels = item.get(labels_key, item.get("labels", []))
        
        data_item = {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": len(input_ids),
        }
        
        # Copy visual features (shared between chosen and rejected)
        if "pixel_values" in item:
            data_item["pixel_values"] = item["pixel_values"]
        if "image_grid_thw" in item:
            data_item["image_grid_thw"] = item["image_grid_thw"]
        if "num_img_tokens" in item:
            data_item["num_img_tokens"] = item["num_img_tokens"]
        if "position_ids" in item:
            data_item["position_ids"] = item.get(f"{seq_type}_position_ids", item["position_ids"])
        
        data.append(data_item)
    
    # Calculate total tokens and truncate if necessary
    total_num_tokens = sum(d.get("num_tokens", len(d.get("input_ids", []))) for d in data)
    
    if total_num_tokens > pack_max_length:
        logger.warning(
            f"Found sample with {total_num_tokens} tokens > pack_max_length {pack_max_length}. Truncating."
        )
        data[0]["input_ids"] = data[0]["input_ids"][:pack_max_length]
        data[0]["labels"] = data[0]["labels"][:pack_max_length]
        data[0]["num_tokens"] = pack_max_length
    
    # Concatenate input_ids and labels
    input_ids = torch.cat([torch.tensor(d["input_ids"]).view(1, -1) for d in data], dim=-1)
    labels = torch.cat([torch.tensor(d["labels"]).view(1, -1) for d in data], dim=-1)
    
    # Handle position_ids for Qwen3-VL (3D: temporal, height, width)
    all_position_ids_none = all(
        "position_ids" not in d or d["position_ids"] is None for d in data
    )
    position_ids_list = []
    if not all_position_ids_none:
        for d in data:
            if "position_ids" in d and d["position_ids"] is not None:
                pos_ids = d["position_ids"]
                if not isinstance(pos_ids, torch.Tensor):
                    pos_ids = torch.tensor(pos_ids)
                position_ids_list.append(pos_ids)
            else:
                # Create default position_ids (3, seq_len, seq_len) for Qwen3-VL
                seq_len = len(d["input_ids"])
                pos_ids = torch.arange(seq_len).view(1, 1, -1).expand(3, seq_len, -1)
                position_ids_list.append(pos_ids)
    
    # Shift for causal LM
    input_ids = input_ids[:, :-1]
    shifted_labels = labels[:, 1:]
    
    position_ids: torch.Tensor | None = None
    if len(position_ids_list) > 0:
        position_ids = torch.cat(position_ids_list, dim=-1)
        position_ids = position_ids[:, :, :-1]
    
    # Calculate num_tokens
    num_tokens = [d.get("num_tokens", len(d.get("input_ids", []))) for d in data]
    if num_tokens[-1] == 1:
        num_tokens = num_tokens[:-1]
    else:
        num_tokens[-1] -= 1
    
    # Padding
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
        raise ValueError(f"Sample length {input_ids.shape[-1]} > pack_max_length {pack_max_length}")
    else:
        num_tokens = [0] + num_tokens
    
    cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()
    
    # Collect visual features
    num_img_tokens: list[int] = []
    for d in data:
        num_img_tokens.extend(d.get("num_img_tokens", [0]))
    
    pixel_values: torch.Tensor | None = None
    pv_list = [d["pixel_values"] for d in data if "pixel_values" in d]
    if pv_list:
        if all(isinstance(pv, torch.Tensor) for pv in pv_list):
            pixel_values = torch.cat(pv_list, dim=0)
    
    image_grid_thw: torch.Tensor | None = None
    thw_list = [d["image_grid_thw"] for d in data if "image_grid_thw" in d]
    if thw_list:
        if all(isinstance(thw, torch.Tensor) for thw in thw_list):
            image_grid_thw = torch.cat(thw_list, dim=0)
    
    # Create SequenceContext with visual features
    seq_ctx = SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max(num_tokens),
        max_length_k=max(num_tokens),
        num_padding=pad_len,
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        num_img_tokens=num_img_tokens if num_img_tokens else None,
    )
    
    return {
        "seq_ctx": seq_ctx,
        "shifted_labels": shifted_labels,
    }
