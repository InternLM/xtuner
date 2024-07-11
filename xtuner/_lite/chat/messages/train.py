import copy
import re
from typing import Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from transformers.tokenization_utils import PreTrainedTokenizer

from xtuner.utils import IGNORE_INDEX

from .chat import (ChatMsg, ImageContentItem, TextContentItem)
from ..templates import ChatTemplate, HybridChatTemplate


class TrainingMsg(ChatMsg):
    loss: Optional[bool] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.loss is None:
            if self.role == 'system':
                self.loss = False
            elif self.role == 'user':
                self.loss = False
            elif self.role == 'assistant':
                self.loss = True
            else:
                raise NotImplementedError

    def _encode_mm_content(self, text: str, mm_token_maps: Dict[str, int],
                           tokenizer: PreTrainedTokenizer):

        mm_tokens = mm_token_maps.keys()

        pattern = r'(' + '|'.join(mm_tokens) + r')'
        chunks = re.split(pattern, text)

        assert len(chunks) > 1

        token_ids = []
        for c in chunks:
            if c in mm_tokens:
                token_ids.append(mm_token_maps[c])
            else:
                token_ids.extend(tokenizer.encode(c, add_special_tokens=False))

        return token_ids

    def _with_multi_modal_content(self):
        flag = False

        if isinstance(self.content, list):
            for item in self.content:
                # TODO (pppppM) support video and audio
                if isinstance(item, ImageContentItem):
                    flag = True
                    break
        return flag

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: HybridChatTemplate,
    ):

        decorated = self.apply_chat_template(chat_template)

        if self._with_multi_modal_content():
            token_maps = chat_template.mm_token_maps
            token_ids = self._encode_mm_content(decorated, token_maps,
                                                tokenizer)
        else:
            token_ids = tokenizer.encode(decorated, add_special_tokens=False)

        if self.loss:
            label_ids = copy.deepcopy(token_ids)
        else:
            label_ids = [IGNORE_INDEX] * len(token_ids)

        image_urls = self.collect_img_urls()

        return {
            'input_ids': token_ids,
            'labels': label_ids,
            'image_urls': image_urls
        }









class TrainingMessages(BaseModel):
    messages: List[TrainingMsg]
    

    @classmethod
    def from_dict(cls, item) -> 'TrainingMessages':
        '''
        item
        {
            'messages':[
                {'role':'user', 'content':'hello'},
                {'role':'assistant', 'content':'hello!'},
            ],
            'funcitons': [],
        }
        '''

        assert 'messages' in item, item

        _messages = item['messages']
        messages = []
        functions = None

        for _msg in _messages:
            assert 'role' in _msg and 'content' in _msg
            _role = _msg['role']
            _content = _msg['content']

            if isinstance(_content, list):

                content = []
                for c_item in _content:
                    assert 'type' in c_item
                    _type = c_item['type']
                    if _type == 'text':
                        assert 'text' in c_item
                        _text = c_item['text']
                        content.append(TextContentItem(type=_type, text=_text))
                    elif _type == 'image_url':
                        assert 'image_url' in c_item
                        _url = c_item['image_url']
                        content.append(
                            ImageContentItem(type=_type, image_url=_url))
                    else:
                        raise NotImplementedError

                msg = TrainingMsg(role=_role, content=content)
                messages.append(msg)
                continue


            if isinstance(_content, str):
                msg = TrainingMsg(role=_role, content=_content)
                messages.append(msg)

            # TODO (pppppM) add format warning

        return cls(messages=messages)

    def collect_img_urls(self) -> List[str]:
        img_urls = []
        for msg in self.messages:
            img_urls.extend(msg.collect_img_urls())
        return img_urls

    def pop_latest_msg(self):
        return self.messages.pop()

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:

        prompt = ''

        if isinstance(self.functions, list) and len(self.functions) > 0:

            functions = [func.model_dump() for func in self.functions]

            prompt += chat_template.decorate_functions(functions)

        for msg in self.messages:
            if msg.role == 'system':
                prompt = msg.apply_chat_template(chat_template) + prompt
            else:
                prompt += msg.apply_chat_template(chat_template)

        return prompt

    def tokenize(self, tokenizer: PreTrainedTokenizer,
                 chat_template: HybridChatTemplate) -> RawTrainingData:

        input_ids = []
        labels = []
        image_urls = []

        bos_token_ids = get_bos_token_ids(tokenizer)
        input_ids.extend(bos_token_ids)
        labels.extend([IGNORE_INDEX] * len(bos_token_ids))

        for msg in self.messages:
            res = msg.tokenize(tokenizer, chat_template)
            token_ids, label_ids = res['input_ids'], res['labels']

            input_ids.extend(token_ids)
            labels.extend(label_ids)

            if 'image_urls' in res:
                image_urls.extend(res['image_urls'])

            # TODO (pppppM) Verify whether sep and suffix_as_eos are necessary

        training_data = {
            'input_ids': input_ids,
            'labels': labels,
            'image_urls': image_urls
        }
        return training_data