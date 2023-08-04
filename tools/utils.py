import copy
import re

from transformers import (PreTrainedTokenizerFast, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

PROMPT_TEMPLATE = {
    'title': {
        'INSTRUCTION_START':
        ('If you are an expert in writing papers, please generate '
         "a good paper title for this paper based on other authors' "
         'descriptions of their abstracts.\n\n'
         '### Descriptions:\n{input}\n\n### Title: '),
        'INSTRUCTION':
        '### Descriptions:\n{input}\n\n### Title: '
    },
    'plugins': {
        'INSTRUCTION_START':
        ('You are an AI assistant whose name is {bot_name}.\n'
         'Capabilities and tools that {bot_name} can possess.\n'
         '- Inner thoughts: enabled.\n'
         '- Web search: enabled. API: Search(query)\n'
         '- Calculator: enabled. API: Calculate(expression)\n'
         '- Equation solver: enabled. API: Solve(equation)\n'
         '- Text-to-image: disabled.\n'
         '- Image edition: disabled.\n'
         '- Text-to-speech: disabled.\n'
         '<|Human|>: {input}'),
        'INSTRUCTION':
        '<|Human|>: {input}'
    },
    'llama-2-chat': {
        'INSTRUCTION_START':
        ('[INST] <<SYS>>\n You are a helpful, respectful and honest '
         'assistant. Always answer as helpfully as possible, while being '
         'safe. Your answers should not include any harmful, unethical, '
         'racist, sexist, toxic, dangerous, or illegal content. Please ensure '
         'that your responses are socially unbiased and positive in nature. '
         '\n<</SYS>>\n\n{input} [/INST]'),
        'INSTRUCTION':
        '[INST] {input} [/INST]'
    },
    'alpaca': {
        'INSTRUCTION_START':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{input}\n\n'
         '### Response: '),
        'INSTRUCTION':
        '### Instruction:\n{input}\n\n### Response: '
    },
    'openassistant': {
        'INSTRUCTION': '### Human: {input}\n### Assistant: '
    },
    'medical': {
        'INSTRUCTION_START':
        ('请从一名专业医生的角度，对下述医学问题给出安全、可靠的回答。\n\n问：{input}\n\n答：'),
        'INSTRUCTION': '问：{input}\n\n答：'
    }
}


def unwarpper_model(model):
    if 'PeftModel' in model.__class__.__name__:
        model = model.base_model.model
    return model


def get_chat_utils(model):
    """Get utils by model type."""
    if model.__class__.__name__ == 'InferenceEngine':
        model = model.module
    model = model.llm
    is_peft = 'PeftModel' in model.__class__.__name__
    is_internlm = 'InternLM' in unwarpper_model(model).__class__.__name__
    stop_criteria = StoppingCriteriaList()
    if is_internlm:
        stop_criteria.append(InternLMStoppingCriteria())
        if is_peft:
            return BaseDecorator, InternLMStreamer, stop_criteria
        else:
            return InternLMDecorator, InternLMStreamer, stop_criteria
    else:
        return BaseDecorator, DecodeOutputStreamer, stop_criteria


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


class BaseDecorator:
    """Base decorator for decorating prompt and extracting generated output."""

    @classmethod
    def decorate(cls, prompt):
        """Abstract method for adding Add special tokens to prompt."""
        return prompt

    @classmethod
    def extract(cls, gen_out):
        """Abstract methods for extract generated output from model output."""
        return gen_out


class InternLMDecorator(BaseDecorator):
    """Decorator for InternLM."""

    regex = re.compile(r'<\|Bot\|>:(.*)')

    @classmethod
    def decorate(cls, prompt):
        """Decorate prompt for InternLM."""
        return f'<|User|>:{prompt}<eoh>'

    @classmethod
    def extract(cls, gen_out):
        """Extract generated tokens for InternLM."""
        return cls.regex.search(gen_out).group(1)


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
