# Copyright (c) OpenMMLab. All rights reserved.
import copy
from dataclasses import dataclass
from typing import List, Tuple

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


@dataclass
class ChatTemplateConfig:
    system: str
    instruction: str
    suffix: str
    suffix_as_eos: bool
    sep: str
    stop_words: Tuple[str] or List[str]

    @staticmethod
    def is_valid_text(text):
        return text != '' and text is not None

    def build_prompt(self, system_text, instruction_text, **kwargs):
        text = ''
        if self.is_valid_text(system_text):
            text += self.system.format(system=system_text, **kwargs)
        text += self.instruction.format(input=instruction_text, **kwargs)
        return text

    def template_map_fn_v1(self, example):
        conversation_original = example['conversation']
        conversation = []
        for i, single_turn in enumerate(conversation_original):
            system_data = single_turn.get('system', '')
            input_data = single_turn.get('input', '')
            if input_data is None:
                input_data = ''
            output_data = single_turn.get('output', '')

            input_text = ''
            if self.is_valid_text(system_data):
                input_text += self.system.format(system=system_data)
            input_text += self.instruction.format(
                input=input_data, round=i + 1)

            output_text = ''
            output_text += output_data
            if self.is_valid_text(output_data):
                output_text += self.suffix

            # suffix_as_eos is False ==> need_eos_token is True
            need_eos_token = not self.suffix_as_eos
            sep = self.sep if i < len(conversation_original) - 1 else ''
            conversation.append({
                'input': input_text,
                'output': output_text,
                'need_eos_token': need_eos_token,
                'sep': sep
            })
        return {'conversation': conversation}

    def template_map_fn_v2(self, example):
        messages_original = example
        messages = []
        n_turn = 1
        for data_idx, data in enumerate(messages_original):
            role = data['role']
            content_original = data['content']
            if role == 'system':
                content = self.system.format(system=content_original)
            elif role == 'user':
                content = self.instruction.format(
                    input=content_original, round=n_turn)
                n_turn += 1
            elif role == 'assistant':
                content = content_original
                if self.suffix != '':
                    content += self.suffix
            else:
                raise NotImplementedError
            messages.append({'role': data['role'], 'content': content})
        return messages

    def encode_map_fn(self,
                      example,
                      tokenizer,
                      max_length,
                      input_ids_with_output=True,
                      with_image_token=False):
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            bos_token_id = []
            eos_token_id = tokenizer.eos_token_id
            assert eos_token_id is not None, \
                'Please set eos_token for Qwen tokenizer!'
        elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
            bos_token_id = [64790, 64792]
            eos_token_id = tokenizer.eos_token_id
        else:
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
        if isinstance(bos_token_id, int):
            bos_token_id = [bos_token_id]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        if len(example['messages']) > 2:
            assert input_ids_with_output

        input_ids, labels = [], []
        next_needs_bos_token = True
        for data_idx, data in enumerate(example['messages']):
            if next_needs_bos_token:
                input_ids += bos_token_id
                labels += [IGNORE_INDEX] * len(bos_token_id)
            role = data['role']
            content = data['content']
            if role == 'system':
                token_ids = tokenizer.encode(content, add_special_tokens=False)
                label_ids = [IGNORE_INDEX] * len(token_ids)
            elif role == 'user':
                if DEFAULT_IMAGE_TOKEN in content and with_image_token:
                    chunk_token_ids = [
                        tokenizer.encode(chunk, add_special_tokens=False)
                        for chunk in content.split(DEFAULT_IMAGE_TOKEN)
                    ]
                    assert len(chunk_token_ids) == 2
                    token_ids = []
                    for chunk_idx, cur_token_ids in enumerate(chunk_token_ids):
                        token_ids.extend(cur_token_ids)
                        if chunk_idx != len(chunk_token_ids) - 1:
                            token_ids.append(IMAGE_TOKEN_INDEX)
                else:
                    token_ids = tokenizer.encode(
                        content, add_special_tokens=False)
                label_ids = [IGNORE_INDEX] * len(token_ids)
            elif role == 'assistant':
                token_ids = tokenizer.encode(content, add_special_tokens=False)
                label_ids = copy.deepcopy(token_ids)
            else:
                raise NotImplementedError
            input_ids += token_ids
            labels += label_ids
            next_needs_bos_token = False
            if role == 'assistant':
                # Add EOS_TOKEN (with loss)
                if not self.suffix_as_eos:
                    next_needs_bos_token = True
                    input_ids += eos_token_id
                    labels += copy.deepcopy(eos_token_id)
                # Add sep (without loss)
                if self.sep != '' and data_idx < len(example['messages']) - 1:
                    token_ids = tokenizer.encode(
                        self.sep, add_special_tokens=False)
                    input_ids += token_ids
                    labels += [IGNORE_INDEX] * len(token_ids)

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        return {'input_ids': input_ids, 'labels': labels}
