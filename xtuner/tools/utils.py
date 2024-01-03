# Copyright (c) OpenMMLab. All rights reserved.
import re

import torch
from transformers import PreTrainedTokenizerFast, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from xtuner.utils import StopWordStoppingCriteria


def get_base_model(model):
    if hasattr(model, 'llm'):
        model = model.llm
    if 'PeftModel' in model.__class__.__name__:
        model = model.base_model.model
    return model


def get_streamer(model):
    if model.__class__.__name__ == 'InferenceEngine':
        model = model.module
    base_model = get_base_model(model)
    base_model_name = base_model.__class__.__name__.lower()
    is_internlm = 'internlm' in base_model_name
    is_qwen = 'qwen' in base_model_name
    is_baichuan = 'baichuan' in base_model_name
    is_chatglm = 'chatglm' in base_model_name
    no_space = is_internlm or is_qwen or is_baichuan or is_chatglm
    if no_space:
        return NoSpaceStreamer
    else:
        return DecodeOutputStreamer


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


class NoSpaceStreamer(DecodeOutputStreamer):

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        BaseStreamer().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def decode(self, value):
        tok = self.tokenizer.decode(value)
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '\r':
            tok = '\n'

        return tok


def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


def auto_dtype_of_deepspeed_config(ds_config):
    if ds_config.get('fp16') and not ds_config.get('bf16'):
        if ds_config.get('fp16').get('enabled') == 'auto':
            ds_config['fp16']['enabled'] = torch.cuda.is_available()
    elif not ds_config.get('fp16') and ds_config.get('bf16'):
        if ds_config.get('bf16').get('enabled') == 'auto':
            ds_config['bf16']['enabled'] = torch.cuda.is_bf16_supported()
    elif ds_config.get('fp16') and ds_config.get('bf16'):
        if ds_config.get('fp16').get('enabled') == 'auto':
            ds_config['fp16']['enabled'] = torch.cuda.is_available()
        if ds_config.get('bf16').get('enabled') == 'auto':
            ds_config['bf16']['enabled'] = torch.cuda.is_bf16_supported()
        if (ds_config['fp16']['enabled'] is True
                and ds_config['bf16']['enabled'] is True):
            ds_config['fp16']['enabled'] = False
            ds_config['bf16']['enabled'] = True
    return ds_config


def is_cn_string(s):
    if re.search('[\u4e00-\u9fff]', s):
        return True
    return False
