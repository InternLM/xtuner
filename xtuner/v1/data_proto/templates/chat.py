# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from pydantic import BaseModel, field_validator


class ChatTemplate(BaseModel):
    chat_template: str | None = None

    # only compute loss on the last assistant response ignoring the multiple rounds of assistant
    only_last_assistant_loss: bool = False  # gpt_oss is True
    loss_assistant_format_mapping: Dict[str, str] | None = None  # gpt_oss is {'<|end|>': '<|return|>'}
