# Copyright (c) OpenMMLab. All rights reserved.
"""
Preference Dataset for DPO (Direct Preference Optimization) training.

This module provides dataset classes for loading and processing preference data
(chosen/rejected pairs) in the xtuner v1 framework.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset

from xtuner.v1.utils import get_logger

from .data_item import DataItem, QwenVL3DataItem
from .jsonl import JsonlDataset
from .utils import CachableTokenizeFunction


logger = get_logger()
T = TypeVar("T")


class PreferenceDataItem(DataItem):
    """Data item for preference data containing chosen and rejected responses."""
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]
    # For VLM
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    position_ids: torch.Tensor | None
    num_img_tokens: list[int] | None


class PreferenceTokenizeFunction(CachableTokenizeFunction[PreferenceDataItem]):
    """
    Tokenize function for preference data.
    
    This function tokenizes the prompt, chosen, and rejected responses
    separately, then combines them for DPO training.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        max_prompt_length: int | None = None,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        add_eos: bool = True,
    ):
        """
        Initialize the preference tokenize function.
        
        Args:
            tokenizer: The tokenizer to use.
            max_length: Maximum total length (prompt + response).
            max_prompt_length: Maximum prompt length. If None, no limit.
            prompt_key: Key for prompt in the data dict.
            chosen_key: Key for chosen response in the data dict.
            rejected_key: Key for rejected response in the data dict.
            add_eos: Whether to add EOS token to responses.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.add_eos = add_eos
        self._state = "runtime"
    
    def set_state(self, state: str):
        """Set the state of the tokenize function (cache or runtime)."""
        self._state = state
    
    def hash(self) -> str:
        """Return a hash for caching purposes."""
        import hashlib
        config_str = f"{self.tokenizer.name_or_path}_{self.max_length}_{self.max_prompt_length}_{self.add_eos}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def __call__(self, data: dict[str, Any], **kwargs) -> PreferenceDataItem | dict:
        """
        Tokenize the preference data.
        
        Args:
            data: Dictionary containing prompt, chosen, and rejected.
        
        Returns:
            PreferenceDataItem with tokenized sequences.
        """
        prompt = data.get(self.prompt_key, "")
        chosen = data.get(self.chosen_key, "")
        rejected = data.get(self.rejected_key, "")
        
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        
        # Truncate prompt if needed
        if self.max_prompt_length is not None and len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[-self.max_prompt_length:]
        
        # Tokenize chosen and rejected
        chosen_ids = self.tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_ids = self.tokenizer(rejected, add_special_tokens=False)["input_ids"]
        
        # Add EOS token
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            chosen_ids = chosen_ids + [self.tokenizer.eos_token_id]
            rejected_ids = rejected_ids + [self.tokenizer.eos_token_id]
        
        # Calculate max completion length
        max_completion_length = self.max_length - len(prompt_ids)
        
        # Truncate completions if needed
        if len(chosen_ids) > max_completion_length:
            chosen_ids = chosen_ids[:max_completion_length]
        if len(rejected_ids) > max_completion_length:
            rejected_ids = rejected_ids[:max_completion_length]
        
        # Build full sequences
        # For DPO: prompt tokens are masked in labels (-100)
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        
        # Labels: mask prompt tokens
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids
        
        # Calculate num_tokens for packing
        num_tokens = max(len(chosen_input_ids), len(rejected_input_ids))
        
        if self._state == "cache":
            return {"num_tokens": num_tokens}
        
        return {
            "input_ids": chosen_input_ids,  # For compatibility with SFT collator
            "labels": chosen_labels,
            "num_tokens": num_tokens,
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
        }


class PreferenceJsonlDataset(JsonlDataset[PreferenceDataItem]):
    """
    JSONL dataset for preference data.
    
    Expected JSONL format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    
    For VLM:
    {"prompt": "...", "chosen": "...", "rejected": "...", "images": ["path/to/img.jpg"]}
    """
    
    def __init__(
        self,
        anno_path: str | Path,
        tokenize_fn: PreferenceTokenizeFunction | CachableTokenizeFunction | None = None,
        sample_ratio: float = 1.0,
        name: str = "preference",
        cache_dir: str | Path | None = None,
        max_length: int | None = None,
        cache_tag: str | None = None,
        **kwargs,
    ):
        """
        Initialize the preference JSONL dataset.
        
        Args:
            anno_path: Path to the JSONL file.
            tokenize_fn: Tokenize function for processing data.
            sample_ratio: Ratio of samples to use.
            name: Dataset name for logging.
            cache_dir: Directory for caching.
            max_length: Maximum sequence length.
            cache_tag: Tag for caching.
        """
        super().__init__(
            anno_path=anno_path,
            tokenize_fn=tokenize_fn,
            sample_ratio=sample_ratio,
            name=name,
            cache_dir=cache_dir,
            max_length=max_length,
            cache_tag=cache_tag,
            **kwargs,
        )


class VLMPreferenceJsonlDataset(PreferenceJsonlDataset):
    """
    JSONL dataset for VLM preference data with image support.
    
    Expected JSONL format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "...",
        "images": ["path/to/img1.jpg", "path/to/img2.jpg"]
    }
    """
    
    def __init__(
        self,
        *args,
        media_root: str | None = "",
        **kwargs,
    ):
        """
        Initialize the VLM preference dataset.
        
        Args:
            media_root: Root directory for media files.
        """
        from functools import wraps
        
        if media_root is None:
            media_root = ""
        self.media_root = media_root
        
        # IMPORTANT: Wrap tokenize_fn BEFORE calling super().__init__()
        # because count_tokens() is called during super().__init__() and needs media_root
        tokenize_fn = kwargs.get('tokenize_fn')
        if tokenize_fn is not None:
            original_tokenize_fn = tokenize_fn
            _media_root = media_root  # Capture in closure
            
            @wraps(original_tokenize_fn)
            def tokenize_fn_with_media_root(data, **fn_kwargs):
                # Always inject media_root if not provided
                if 'media_root' not in fn_kwargs:
                    fn_kwargs['media_root'] = _media_root
                return original_tokenize_fn(data, **fn_kwargs)
            
            # Preserve the original methods for CachableTokenizeFunction
            tokenize_fn_with_media_root.set_state = original_tokenize_fn.set_state
            tokenize_fn_with_media_root.hash = original_tokenize_fn.hash
            kwargs['tokenize_fn'] = tokenize_fn_with_media_root
        
        super().__init__(*args, **kwargs)
        
        # Fake data for error handling
        self.fake_data = {
            "id": -1,
            "prompt": "你好",
            "chosen": "你好呀！很高兴为你服务～",
            "rejected": "抱歉，我不太明白。",
        }
    
    def __getitem__(self, item) -> PreferenceDataItem | dict:
        """Get an item from the dataset with error handling."""
        try:            
            with open(self.path) as f:
                f.seek(self.offsets[item])
                line = f.readline()
            
            raw_data = json.loads(line)
            
            if self.tokenize_fn:
                tokenized_data = self.tokenize_fn(raw_data, media_root=self.media_root)
                return tokenized_data
            else:
                return raw_data
        except Exception as e:
            logger.warning(f"[{os.path.basename(self.path)}]: {e}. Dumping a fake data.")
            data = self.tokenize_fn(self.fake_data)
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            # Mask all labels for fake data
            if "chosen_labels" in data:
                data["chosen_labels"] = [-100] * len(data["chosen_input_ids"])
            if "rejected_labels" in data:
                data["rejected_labels"] = [-100] * len(data["rejected_input_ids"])
            return data


class InMemoryPreferenceDataset(Dataset):
    """
    In-memory preference dataset for smaller datasets.
    
    This dataset loads all data into memory for faster access during training.
    """
    
    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenize_fn: Callable | None = None,
    ):
        """
        Initialize the in-memory dataset.
        
        Args:
            data: List of preference data dictionaries.
            tokenize_fn: Optional tokenize function.
        """
        self.data = data
        self.tokenize_fn = tokenize_fn
        
        # Pre-tokenize if function provided
        if tokenize_fn is not None:
            self.tokenized_data = [tokenize_fn(d) for d in data]
            self.num_tokens = [d.get("num_tokens", 0) for d in self.tokenized_data]
        else:
            self.tokenized_data = None
            self.num_tokens = None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        if self.tokenized_data is not None:
            return self.tokenized_data[idx]
        return self.data[idx]
    
    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        tokenize_fn: Callable | None = None,
    ) -> "InMemoryPreferenceDataset":
        """
        Create dataset from a JSONL file.
        
        Args:
            path: Path to the JSONL file.
            tokenize_fn: Optional tokenize function.
        
        Returns:
            InMemoryPreferenceDataset instance.
        """
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        return cls(data, tokenize_fn)
    
    @classmethod
    def from_hf_dataset(
        cls,
        dataset,
        tokenize_fn: Callable | None = None,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
    ) -> "InMemoryPreferenceDataset":
        """
        Create dataset from a HuggingFace dataset.
        
        Args:
            dataset: HuggingFace dataset object.
            tokenize_fn: Optional tokenize function.
            prompt_key: Key for prompt in the dataset.
            chosen_key: Key for chosen response.
            rejected_key: Key for rejected response.
        
        Returns:
            InMemoryPreferenceDataset instance.
        """
        data = []
        for item in dataset:
            data.append({
                "prompt": item.get(prompt_key, ""),
                "chosen": item.get(chosen_key, ""),
                "rejected": item.get(rejected_key, ""),
            })
        return cls(data, tokenize_fn)

# ============================================================================
# DPO Tokenize Function - Inherits from Qwen3VLTokenizeFunction
# ============================================================================

from .mllm_tokenize_fn import OSSLoaderConfig
from .mllm_tokenize_fn.qwen3_vl_tokenize_fn import (
    Qwen3VLTokenizeFunction,
    Qwen3VLTokenizeFnConfig,
)


class Qwen3VLDPOTokenizeFnConfig(Qwen3VLTokenizeFnConfig):
    """
    Configuration for Qwen3-VL DPO tokenize function.
    
    Inherits from Qwen3VLTokenizeFnConfig to reuse all VLM processing config.
    """
    
    # DPO specific keys
    prompt_key: str = "question"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    images_key: str = "image"
    
    def build(
        self, tokenizer, tokenizer_hash: str | None = None, anno_name: str = "", **kwargs
    ) -> "Qwen3VLDPOTokenizeFunction":
        return Qwen3VLDPOTokenizeFunction(
            tokenizer=tokenizer,
            processor_path=self.processor_path,
            anno_name=anno_name,
            prompt_key=self.prompt_key,
            chosen_key=self.chosen_key,
            rejected_key=self.rejected_key,
            images_key=self.images_key,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            oss_loader_cfg=self.oss_loader_cfg,
            video_min_total_pixels=self.video_min_total_pixels,
            video_max_total_pixels=self.video_max_total_pixels,
            video_min_frames=self.video_min_frames,
            video_max_frames=self.video_max_frames,
            rand_video_max_frames=self.rand_video_max_frames,
            fps=self.fps,
            enable_3d_rope=self.enable_3d_rope,
            add_vision_id=self.add_vision_id,
            max_length=self.max_length,
            system_message=self.system_message,
            tokenizer_hash=tokenizer_hash,
            hash=self.hash,
            debug=self.debug,
            oss_time_log_thr=self.oss_time_log_thr,
            add_eos_token=self.add_eos_token,
            add_bos_token=self.add_bos_token,
        )


class Qwen3VLDPOTokenizeFunction(Qwen3VLTokenizeFunction):
    """
    DPO tokenize function for Qwen3-VL.
    
    Inherits from Qwen3VLTokenizeFunction to reuse all image/video processing logic.
    Handles DPO format: {question, chosen, rejected, image} -> two sequences.
    
    Expected data format (MMPR style):
    {
        "question": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}],
        "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}],
        "rejected": [{"role": "assistant", "content": [{"type": "text", "text": "..."}]}],
        "image": "path/to/image.jpg"
    }
    """
    
    def __init__(
        self,
        tokenizer,
        processor_path: str,
        anno_name: str,
        prompt_key: str = "question",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        images_key: str = "image",
        **kwargs,
    ):
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.images_key = images_key
        super().__init__(tokenizer, processor_path, anno_name, **kwargs)
    
    def _extract_text_from_messages(self, messages) -> str:
        """Extract text content from messages format."""
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list) and len(messages) > 0:
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                return item.get('text', '')
        return str(messages)
    
    def _convert_mmpr_to_xtuner_format(self, messages: list, images: list, image_wh_list: list | None = None) -> list:
        """
        Convert MMPR format messages to xtuner ChatMessages format.
        
        MMPR format: {"type": "image"}, {"type": "text", "text": "..."}
        xtuner format: {"type": "image_url", "image_url": {...}}, {"type": "text", "text": "<IMG_CONTEXT>..."}
        
        The key difference is that xtuner expects <IMG_CONTEXT> placeholder in text content,
        while MMPR uses separate {"type": "image"} items.
        
        Args:
            messages: List of message dicts
            images: List of image paths
            image_wh_list: List of [width, height] for each image
        """
        import copy
        converted = []
        image_idx = 0
        
        for msg in messages:
            new_msg = copy.deepcopy(msg)
            content = new_msg.get('content', [])
            
            if isinstance(content, list):
                new_content = []
                
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image':
                            # Convert MMPR image format to xtuner format
                            # MMPR may have only one {"type": "image"} but multiple images in the list
                            # Add ALL remaining images when we see the first image placeholder
                            while image_idx < len(images):
                                img_path = images[image_idx]
                                image_url_dict = {"url": img_path}
                                # Add image_wh if available
                                if image_wh_list is not None and image_idx < len(image_wh_list):
                                    image_url_dict["image_wh"] = image_wh_list[image_idx]
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": image_url_dict
                                })
                                image_idx += 1
                        elif item.get('type') == 'text':
                            # Data already preprocessed with <IMG_CONTEXT>, just pass through
                            new_content.append({
                                "type": "text",
                                "text": item.get('text', '')
                            })
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                
                new_msg['content'] = new_content
            converted.append(new_msg)
        
        return converted
    
    def _build_dpo_messages(self, data_item: dict, response_messages: list) -> dict:
        """Build messages format for DPO by combining prompt and response."""
        prompt_messages = data_item.get(self.prompt_key, [])
        images = data_item.get(self.images_key, [])
        image_wh_list = data_item.get("image_wh", None)
        
        # Normalize images to list
        if isinstance(images, str):
            images = [images] if images else []
        elif images is None:
            images = []
        
        # Ensure prompt_messages is a list
        if not isinstance(prompt_messages, list):
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": str(prompt_messages)}]}]
        
        # Convert MMPR format to xtuner format (with image_wh)
        prompt_messages = self._convert_mmpr_to_xtuner_format(prompt_messages, images, image_wh_list)
        response_messages = self._convert_mmpr_to_xtuner_format(response_messages, [], None)
        
        # Combine prompt and response
        combined_messages = prompt_messages + response_messages
        
        return {"messages": combined_messages}
    
    def __call__(self, item: dict, media_root: str = "", **kwargs):
        """
        Process DPO data item.
        
        Returns dict with:
            - chosen_input_ids, chosen_labels
            - rejected_input_ids, rejected_labels
            - pixel_values, image_grid_thw (shared between chosen and rejected)
            - num_tokens
        """
        from .data_item import CacheItem
        
        # Get chosen and rejected responses
        chosen_raw = item.get(self.chosen_key, [])
        rejected_raw = item.get(self.rejected_key, [])
        
        # Build combined messages for chosen and rejected
        chosen_item = self._build_dpo_messages(item, chosen_raw)
        rejected_item = self._build_dpo_messages(item, rejected_raw)

        # Use parent's __call__ to process (handles image extraction and processing)
        try:
            # Process chosen sequence (this also processes images)
            chosen_result = super().__call__(chosen_item, media_root=media_root, **kwargs)
            if self.state == "cache":
                # For cache mode, also check rejected sequence length
                rejected_result = super().__call__(rejected_item, media_root=media_root, **kwargs)
                num_tokens = max(
                    chosen_result.get("num_tokens", 0) if isinstance(chosen_result, dict) else chosen_result.num_tokens,
                    rejected_result.get("num_tokens", 0) if isinstance(rejected_result, dict) else rejected_result.num_tokens,
                )
                return {"num_tokens": num_tokens}
            
            # Process rejected sequence
            rejected_result = super().__call__(rejected_item, media_root=media_root, **kwargs)

            # Normalize outputs: parent may return dict or DataItem-like object
            def _as_dict(x: Any) -> dict:
                if isinstance(x, dict):
                    return x
                # DataItem / pydantic model / dataclass-like
                if hasattr(x, "model_dump"):
                    return x.model_dump()
                if hasattr(x, "__dict__"):
                    return dict(x.__dict__)
                return {"value": x}

            chosen_d = _as_dict(chosen_result)
            rejected_d = _as_dict(rejected_result)
            
            # Build DPO result
            result = {
                "input_ids": chosen_d["input_ids"],
                "labels": chosen_d["labels"],
                "num_tokens": max(len(chosen_d["input_ids"]), len(rejected_d["input_ids"])),
                "chosen_input_ids": chosen_d["input_ids"],
                "chosen_labels": chosen_d["labels"],
                "rejected_input_ids": rejected_d["input_ids"],
                "rejected_labels": rejected_d["labels"],
            }
            
            # Add visual features (shared between chosen and rejected)
            if chosen_d.get("pixel_values", None) is not None:
                result["pixel_values"] = chosen_d["pixel_values"]
            if chosen_d.get("image_grid_thw", None) is not None:
                result["image_grid_thw"] = chosen_d["image_grid_thw"]
            if chosen_d.get("position_ids", None) is not None:
                result["chosen_position_ids"] = chosen_d["position_ids"]
            if rejected_d.get("position_ids", None) is not None:
                result["rejected_position_ids"] = rejected_d["position_ids"]
            if chosen_d.get("num_img_tokens", None) is not None:
                result["num_img_tokens"] = chosen_d["num_img_tokens"]
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to process DPO item: {e}")
            if self.state == "cache":
                return {"num_tokens": 0}
            raise
