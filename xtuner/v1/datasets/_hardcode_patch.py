# type: ignore
"""Temporary monkey patch for skipping empty <think></think> tags.

This file should be completely removed when the feature is no longer needed.
To remove:
1. Delete this file
2. Remove `from . import _hardcode_patch` in __init__.py
3. Remove XTUNER_SKIP_EMPTY_THINK environment variable usage

DO NOT refactor this into "proper" abstractions - it's meant to be disposable.


There are things in this world we do not out of desire, but out of necessity—
this code stands as witness to such fate.

Where my eyes gaze, only memories remain;
where my heart strays, only yesterday's pain;
where my sight stays, only regret's refrain.
"""

import functools
import os
from abc import ABC

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import get_logger

from .ftdp import FTDPTokenizeFnConfig
from .mllm_tokenize_fn import Qwen3VLTokenizeFunction
from .pt_tokenize_fn import PretrainTokenizeFunction
from .rl_tokenize_fn.rl_tokenize_fn import InternS1VLTokenizeFunction
from .sft_tokenize_fn import OpenaiTokenizeFunction


logger = get_logger()


# Need to inherit ABC otherwise the assignment of `__bases__` will failed
# https://stackoverflow.com/questions/3308792/python-object-layout
class _SkipEmptyThink(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        # For `__mro__` compatibility
        super().__init__(tokenizer, *args, **kwargs)
        self._tokenizer = tokenizer
        if "<think>" in tokenizer.added_tokens_encoder:
            self._skip_seq = [
                tokenizer.added_tokens_encoder["<think>"],
                tokenizer.added_tokens_encoder["</think>"],
            ]
        else:
            self._skip_seq = []

        if not self._skip_seq:
            logger.warning("`SkipEmptyThink` is disabled because <think> or </think> token is not found in tokenizer.")
        else:
            logger.info("`SkipEmptyThink` is enabled to skip empty <think></think> sequences in labels.")

    def process_labels(self, input_ids: list[int], labels: list[int]):
        """Remove all complete occurrences of skip_seq from labels.

        Args:
            input_ids (list[int]): Input token ID sequence.
            labels (list[int]): Input label sequence.

        Returns:
            list[int]: Labels with skip_seq occurrences removed.
        """
        assert len(input_ids) == len(labels), "Hardcode Error! input_ids and labels must have the same length."
        if not self._skip_seq:
            return labels

        new_labels = []
        buffer = []
        label_buffer = []

        for label_id, token_id in zip(labels, input_ids):
            buffer.append(token_id)
            label_buffer.append(label_id)

            # Check if buffer matches skip_seq
            if buffer == self._skip_seq:
                # Complete match found, replace with -100
                new_labels.extend([-100] * len(self._skip_seq))
                buffer = []
                label_buffer = []
            # Check if buffer is still a valid prefix
            elif buffer == self._skip_seq[: len(buffer)]:
                # Continue accumulating
                pass
            else:
                # No match, flush buffer and restart with current token
                new_labels.extend(label_buffer[:-1])  # Add all except current

                # Check if current token starts a new potential match
                if token_id == self._skip_seq[0]:
                    buffer = [token_id]
                    label_buffer = [label_id]
                else:
                    new_labels.append(label_id)
                    buffer = []
                    label_buffer = []

        # Handle remaining buffer (partial match at end)
        if buffer:
            new_labels.extend(label_buffer)

        return new_labels


def _label_processor_wrapper(fn):
    @functools.wraps(fn)
    def _wrapped(self, *args, **kwargs):
        ret = fn(self, *args, **kwargs)
        ret["labels"] = self.process_labels(ret["input_ids"], ret["labels"])
        return ret

    return _wrapped


if os.environ.get("XTUNER_SKIP_EMPTY_THINK", "0") == "1":
    for tokenize_fn_class in [Qwen3VLTokenizeFunction, InternS1VLTokenizeFunction]:
        if _SkipEmptyThink not in tokenize_fn_class.__mro__:
            tokenize_fn_class.__bases__ = (_SkipEmptyThink,) + tokenize_fn_class.__bases__

        tokenize_fn_class.multi_modal_get_item = _label_processor_wrapper(tokenize_fn_class.multi_modal_get_item)
        tokenize_fn_class.video_get_item = _label_processor_wrapper(tokenize_fn_class.video_get_item)
        tokenize_fn_class.pure_text_get_item = _label_processor_wrapper(tokenize_fn_class.pure_text_get_item)

    for text_tokenize_fn_class in [OpenaiTokenizeFunction, FTDPTokenizeFnConfig, PretrainTokenizeFunction]:
        if _SkipEmptyThink not in text_tokenize_fn_class.__mro__:
            text_tokenize_fn_class.__bases__ = (_SkipEmptyThink,) + text_tokenize_fn_class.__bases__

        text_tokenize_fn_class.__call__ = _label_processor_wrapper(text_tokenize_fn_class.__call__)
