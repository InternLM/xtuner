# Copyright (c) OpenMMLab. All rights reserved.
from datetime import datetime

from .chat import ChatTemplate
from .hybrid import HybridChatTemplate


current_date = datetime.now().strftime("%Y-%m-%d")

CHAT_TEMPLATE_MAP = {
    "intern-s1": HybridChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        assistant="{assistant}<|im_end|>",
        stop_words=["<|im_end|>"],
        image_start_token="<img>",
        image_end_token="</img>",
        image_context_token="<IMG_CONTEXT>",
        video_context_token="<IMG_CONTEXT>",
    ),
    "qwen3-vl": HybridChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        default_system="You are a helpful assistant.",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        stop_words=["<|im_end|>", "<|endoftext|>"],
        assistant="{assistant}<|im_end|>",
        image_start_token="<|vision_start|>",
        image_end_token="<|vision_end|>",
        image_context_token="<|image_pad|>",
        video_context_token="<|video_pad|>",
    ),
    "llama3": HybridChatTemplate(
        system="<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        user=(
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        assistant="{assistant}<|eot_id|>",
        sep="",
        stop_words=["<|eot_id|>"],
    ),
    "qwen3": HybridChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        developer="<|im_start|>system\n{developer}<|im_end|>\n",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        assistant="{assistant}<|im_end|>",
        stop_words=["<|im_end|>", "<|endoftext|>"],
        sep="\n",
    ),
    "gpt-oss": HybridChatTemplate(
        system="<|start|>system<|message|>{system}<|end|>",
        developer="<|start|>developer<|message|># Instructions\n\n{developer}\n\n<|end|>",
        default_system=f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {current_date}\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.",
        user="<|start|>user<|message|>{user}<|end|><|start|>assistant",
        assistant="<|channel|>final<|message|>{assistant}<|end|>",
        thinking="<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant",
        stop_words=["<|return|>"],
        sep="",
        # only compute loss on the last assistant response ignoring the multiple rounds of assistant
        only_last_assistant_loss=True,
        # if assistant calculates loss, use "<|channel|>final<|message|>{assistant}<|end|>"
        # else "<|channel|>final<|message|>{assistant}<|return|>"
        loss_assistant_format_mapping={"<|end|>": "<|return|>"},
    ),
    "deepseek-v3": HybridChatTemplate(
        system="<｜begin▁of▁sentence｜>{system}",
        developer="<｜begin▁of▁sentence｜>{developer}",
        user="<｜User｜>{user}<｜Assistant｜></think>",
        assistant="{assistant}<｜end▁of▁sentence｜>",
        stop_words=["<｜end▁of▁sentence｜>"],
        sep="",
    ),
}

__all__ = ["ChatTemplate", "HybridChatTemplate"]
