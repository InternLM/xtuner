import copy
import re

from transformers import (PreTrainedTokenizerFast, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer


def unwarpper_model(model):
    if 'PeftModel' in model.__class__.__name__:
        model = model.base_model.model
    return model


def get_chat_utils(model):
    """Get utils by model type."""
    if model.__class__.__name__ == 'InferenceEngine':
        model = model.module
    model = model.llm
    is_internlm = 'InternLM' in unwarpper_model(model).__class__.__name__
    stop_criteria = StoppingCriteriaList()
    if is_internlm:
        stop_criteria.append(InternLMStoppingCriteria())
        return InternLMStreamer, stop_criteria
    else:
        return DecodeOutputStreamer, stop_criteria


class DecodeOutputStreamer(BaseStreamer):
    """Default streamer for HuggingFace models."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            self.decode = self._decode_with_raw_id
            self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')
        else:
            self.decode = self._decode_fallback

    def _decode_with_raw_id(self, value):
        """Convert token ids to tokens and decode."""

        tok = self.tokenizer._convert_id_to_token(value)
        if tok.startswith('▁'):  # sentencepiece
            space = ' '
            tok = tok[1:]
        else:
            space = ''
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>':
            tok = '\n'
        return space + tok

    def _decode_fallback(self, value):
        """Fallback decoder for non-fast tokenizer."""

        tok = self.tokenizer.decode(
            value,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        return tok + ' '

    def put(self, value):
        """Callback function to decode token and output to stdout."""

        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            tok = self.decode(value[0])
            print(tok, end='', flush=True)

        self.gen_len += 1

    def end(self):
        """Callback function to finish generation."""

        print('\n')


class InternLMStreamer(DecodeOutputStreamer):
    """Streamer for InternLM."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        BaseStreamer().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def decode(self, value):
        """Decode generated tokens for InternLM."""

        tok = self.tokenizer.decode(value)
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '<eoa>' or tok == '\r':
            tok = '\n'

        return tok


class InternLMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for HF version of InternLM."""

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        return input_ids[0, -1] in [2, 103028]


class StopWordStoppingCriteria(StoppingCriteria):
    """Stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word


def update_stop_criteria(base,
                         tokenizer,
                         command_stop_word=None,
                         answer_stop_word=None):
    command = copy.deepcopy(base)
    answer = copy.deepcopy(base)
    if command_stop_word is not None:
        command.append(StopWordStoppingCriteria(tokenizer, command_stop_word))
    if answer_stop_word is not None:
        answer.append(StopWordStoppingCriteria(tokenizer, answer_stop_word))
    return command, answer
