import copy
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from xtuner.types.chat_template import ChatTemplate
from xtuner.utils import IGNORE_INDEX
from xtuner.utils.tokenizer import get_bos_token_ids
from .base import BaseMessages


class ChatMsg(BaseModel):

    role: Literal['assistant', 'user', 'system']
    content: str
    loss: Optional[bool] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loss is None:
            if self.role == 'system':
                self.loss = False
            elif self.role == 'user':
                self.loss = False
            elif self.role == 'assistant':
                self.loss = True
            else:
                raise NotImplementedError

    def get_prompt(self, chat_template: ChatTemplate) -> str:

        if isinstance(self.content, str):
            text = self.content
        else:
            raise NotImplementedError

        if self.role == 'system':
            prompt = chat_template.decorate_system(text)
        elif self.role == 'user':
            prompt = chat_template.decorate_user(text)
        elif self.role == 'assistant':
            prompt = chat_template.decorate_assistant(text)
        else:
            raise NotImplementedError

        return prompt

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):

        decorated = self.get_prompt(chat_template)

        token_ids = tokenizer.encode(decorated, add_special_tokens=False)

        if self.loss:
            label_ids = copy.deepcopy(token_ids)
        else:
            label_ids = [IGNORE_INDEX] * len(token_ids)

        return {
            'input_ids': token_ids,
            'labels': label_ids,
        }


class ChatMessages(BaseMessages):

    messages: List[ChatMsg]

    def add(self, role, content, loss=False):
        self.messages.append(ChatMsg(role=role, content=content, loss=loss))

    def pop(self):
        return self.messages.pop()

    def get_prompt(self, chat_template: ChatTemplate) -> str:

        prompt = ''

        for msg in self.messages:
            prompt += msg.get_prompt(chat_template)

        return prompt

    def tokenize(self, tokenizer: PreTrainedTokenizer,
                 chat_template: ChatTemplate) -> Dict:

        input_ids = []
        labels = []

        bos_token_ids = get_bos_token_ids(tokenizer)
        input_ids.extend(bos_token_ids)
        labels.extend([IGNORE_INDEX] * len(bos_token_ids))

        for msg in self.messages:
            res = msg.tokenize(tokenizer, chat_template)
            token_ids, label_ids = res['input_ids'], res['labels']

            input_ids.extend(token_ids)
            labels.extend(label_ids)
            # TODO (pppppM) Verify whether sep and suffix_as_eos are necessary

        training_data = {
            'input_ids': input_ids,
            'labels': labels,
            'num_tokens': len(input_ids)
        }
        return training_data

    @classmethod
    def from_str(cls, prompt: str) -> 'ChatMessages':

        msg = ChatMsg(role='user', content=prompt)
        return cls(messages=[msg])

    @classmethod
    def from_dict(cls, item: dict) -> 'ChatMessages':
        '''
        item
        {
            'messages':[
                {'role':'user', 'content':'hello'},
                {'role':'assistant', 'content':'hello!'},
            ],
        }
        '''

        assert 'messages' in item, item

        _messages = item['messages']
        messages = []

        for _msg in _messages:
            assert 'role' in _msg and 'content' in _msg
            _role = _msg['role']
            _content = _msg['content']
            if 'loss' in _msg:
                _loss = _msg['loss']
                msg = ChatMsg(role=_role, content=_content, loss=_loss)
            else:
                msg = ChatMsg(role=_role, content=_content)
            messages.append(msg)

        return cls(messages=messages)


if __name__ == '__main__':

    data = {
        'messages': [
            {
                'role': 'user',
                'content': 'hello'
            },
            {
                'role': 'assistant',
                'content': 'hello!'
            },
        ]
    }

    messages = ChatMessages.from_dict(data)
    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
    )

    print(messages.get_prompt(chat_template))
