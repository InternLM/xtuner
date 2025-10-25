# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator


class HybridChatTemplate(BaseModel):
    """Define a Pydantic data model for a hybrid chat with attributes for
    system, user and assistant chat as well as function and interpreter calls
    and results."""
    chat_template: str | None = None

    # only compute loss on the last assistant response ignoring the multiple rounds of assistant
    only_last_assistant_loss: bool = False  # gpt_oss is True
    loss_assistant_format_mapping: Dict[str, str] = {}  # gpt_oss is {'<|end|>': '<|return|>'}

    # Multimodal Chat
    # Predefined token and index for images
    image_context_token: str | None = None
    video_context_token: str | None = None
    image_start_token: str = ""
    image_end_token: str = ""
    image_token_index: int = -100
