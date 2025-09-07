# Copyright (c) OpenMMLab. All rights reserved.
from .chat import ChatTemplate
from .hybrid import HybridChatTemplate


CHAT_TEMPLATE_MAP = {
    "intern-s1": HybridChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        assistant="{assistant}<|im_end|>",
        stop_words=["<|im_end|>"],
    ),
    "llama3": HybridChatTemplate(
        system=("<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"),
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
        user="<|start|>user<|message|>{user}<|end|><|start|>assistant",
        assistant="<|channel|>final<|message|>{assistant}<|end|>",
        stop_words=["<|end|>", "<|return|>"],
        sep="",
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
