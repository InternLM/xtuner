from typing import List, Union

from transformers import AutoTokenizer

from xtuner.registry import BUILDER


def build_tokenizer(tokenizer: Union[str, dict]):

    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    elif isinstance(tokenizer, dict):
        return BUILDER.build(tokenizer)
    else:
        raise TypeError


def get_bos_token_ids(tokenizer) -> List[int]:

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        bos_token_ids = []
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_ids = [64790, 64792]
    else:
        bos_token_ids = tokenizer.bos_token_id

    if isinstance(bos_token_ids, int):
        bos_token_ids = [bos_token_ids]

    return bos_token_ids


def get_eos_token_ids(tokenizer) -> List[int]:
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        eos_token_ids = tokenizer.eos_token_id
        assert eos_token_ids is not None, \
            'Please set eos_token for Qwen tokenizer!'
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        eos_token_ids = tokenizer.eos_token_id
    else:
        eos_token_ids = tokenizer.eos_token_id

    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    return eos_token_ids
