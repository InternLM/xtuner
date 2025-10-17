# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from transformers import PreTrainedTokenizer
from xtuner.utils import IGNORE_INDEX
from xtuner.v1.data_proto.messages.base import BaseMessages
from xtuner.v1.data_proto.templates import ChatTemplate, HybridChatTemplate
from xtuner.v1.utils import get_logger


logger = get_logger()


class TextContentItem(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return self.text


class ImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None
    image_wh: Optional[List[int]] = None  # width, height


class ImageContentItem(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL

    def apply_chat_template(self, *args, **kwarg) -> str:
        return ""


class VideoURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None
    video_length: Optional[int] = None  # duration or frame count


class VideoContentItem(BaseModel):
    type: Literal["video_url"] = "video_url"
    video_url: VideoURL

    def apply_chat_template(self, *args, **kwargs) -> str:
        return ""


MultModalContentType = Union[TextContentItem, ImageContentItem, VideoContentItem]
ContentType = Union[str, List[MultModalContentType]]


class ChatMsg(BaseModel):
    role: Literal["assistant", "user", "system", "developer"]
    content: ContentType
    loss: Optional[bool] = None
    thinking: Optional[str] = None  # only for assistant
    tool_calls: Optional[List[Dict] | Dict] = None  # only for assistant
    name: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.role != "assistant":
            assert self.thinking is None, "Only assistant can have thinking."
            assert self.tool_calls is None, "Only assistant can have tool_calls."
        if self.loss is None:
            if self.role == "system":
                self.loss = False
            elif self.role == "developer":
                self.loss = False
            elif self.role == "user":
                self.loss = False
            elif self.role == "assistant":
                self.loss = True
            else:
                raise NotImplementedError

def process_message(messages: List[ChatMsg], chat_template: ChatTemplate):
    if not messages:
        return messages

    # if chat_template.default_system is not None and messages[0].role != "system":
    #     messages.insert(0, ChatMsg(role="system", content=chat_template.default_system, loss=False))

    # Only look at the last round, if there is thinking, keep it, otherwise remove it all
    for msg in messages[:-1]:
        msg.thinking = None

    # only compute loss on the last assistant response when only_last_assistant_loss is True
    last_msg = messages[-1]
    if last_msg.role == "assistant" and chat_template.only_last_assistant_loss:
        for msg in messages[:-1]:
            if msg.role == "assistant":
                msg.loss = False


class ChatMessages(BaseMessages):
    messages: List[ChatMsg]
    tools: Optional[List[Dict]] = None

    def add(self, role, content, loss=False):
        self.messages.append(ChatMsg(role=role, content=content, loss=loss))

    def pop(self):
        return self.messages.pop()

    def get_prompt(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> str:
        process_message(self.messages, chat_template)
        prompt = tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt


    def tokenize(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> Dict:
        input_ids = tokenizer.encode("", add_special_tokens=False)
        labels = [IGNORE_INDEX for _ in input_ids]

        process_message(self.messages, chat_template)
        def split_conversation(
            messages: List[Dict[str, str]]
        ) -> List[List[Dict[str, str]]]:
            final_chunks = []
            context_chunk = []
            for message in messages:
                if message.role == "assistant" and message.loss:
                    if context_chunk:
                        final_chunks.append(context_chunk)
                    final_chunks.append([message])
                    context_chunk = []
                else:
                    context_chunk.append(message)
            return final_chunks

        for idx, msg in enumerate(split_conversation(self.messages)):
            if msg[0].role == 'assistant':
                assistant_with_gen = tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template.chat_template
                )
                assistant_without_gen = tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=chat_template.chat_template,
                )
                prompt = assistant_without_gen[len(assistant_with_gen) - len(assistant_without_gen):]
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                label_ids = copy.deepcopy(token_ids)
            else:
                token_ids = tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                    chat_template=chat_template.chat_template,
                    tools=self.tools if idx == 0 else None
                )
                label_ids = [IGNORE_INDEX] * len(token_ids)

            input_ids.extend(token_ids)
            labels.extend(label_ids)

        if len(input_ids) != len(labels):
            logger.error(f"[messages] {self.messages}")
            logger.error(f"[input_ids] {input_ids}")
            logger.error(f"[labels] {labels}")
            raise RuntimeError(
                f"The lengths of input_ids and labels must be equal, but  found {len(input_ids)} and {len(labels)}."
            )

        training_data = {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": len(input_ids),
        }
        return training_data

    @classmethod
    def from_str(cls, prompt: str) -> "ChatMessages":
        msg = ChatMsg(role="user", content=prompt)
        return cls(messages=[msg])

    @classmethod
    def from_dict(cls, item: dict) -> "ChatMessages":
        """Item { 'messages':[ {'role':'user', 'content':'hello'},
        {'role':'assistant', 'content':'hello!'}, ], }"""
        return cls(**item)


if __name__ == "__main__":
    data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello!"},
        ],
        "tools": [
            {
                'type': 'function',
                'function': {
                    'name': 'get_current_temperature',
                    'description': 'Get current temperature at a location.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                            },
                            'unit': {
                                'type': 'string',
                                'enum': [
                                    'celsius',
                                    'fahrenheit'
                                ],
                                'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                            }
                        },
                        'required': [
                            'location'
                        ]
                    }
                }
            }, {
                'type': 'function',
                'function': {
                    'name': 'get_temperature_date',
                    'description': 'Get temperature at a location and date.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                            },
                            'date': {
                                'type': 'string',
                                'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                            },
                            'unit': {
                                'type': 'string',
                                'enum': [
                                    'celsius',
                                    'fahrenheit'
                                ],
                                'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                            }
                        },
                        'required': [
                            'location',
                            'date'
                        ]
                    }
                }
            }]
    }

    from transformers import AutoTokenizer
    MODEL_PATH = None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    messages = ChatMessages.from_dict(data)
    chat_template = ChatTemplate(
    )
    print(messages.tokenize(tokenizer, chat_template))
