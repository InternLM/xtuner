# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.utils import IGNORE_INDEX
from xtuner.v1.data_proto.messages.base import BaseMessages
from xtuner.v1.data_proto.templates import ChatTemplate, HybridChatTemplate
from xtuner.v1.utils import get_logger


logger = get_logger()


class TextContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    text: str

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return self.text


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None
    image_wh: Optional[List[int]] = None  # width, height


class ImageContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL

    def apply_chat_template(self, *args, **kwarg) -> str:
        return ""


class VideoURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None
    video_length: Optional[int] = None  # duration or frame count


class VideoContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["video_url"] = "video_url"
    video_url: VideoURL

    def apply_chat_template(self, *args, **kwargs) -> str:
        return ""


MultModalContentType = Union[TextContentItem, ImageContentItem, VideoContentItem]
ContentType = Union[str, List[MultModalContentType]]


class ChatMsg(BaseModel):
    role: Literal["assistant", "user", "system", "developer", "pretrain"]
    model_config = ConfigDict(extra="forbid")
    content: ContentType
    loss: Optional[bool] = None
    thinking: Optional[str] = None  # only for assistant

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.role != "assistant":
            assert self.thinking is None, "Only assistant can have thinking."
        if self.loss is None:
            if self.role == "system":
                self.loss = False
            elif self.role == "developer":
                self.loss = False
            elif self.role == "user":
                self.loss = False
            elif self.role == "assistant":
                self.loss = True
            elif self.role == "pretrain":
                self.loss = True
            else:
                raise NotImplementedError

    def get_prompt(self, chat_template: ChatTemplate) -> str:
        if isinstance(self.content, str):
            text = self.content
        elif isinstance(self.content, list):
            text = ""
            for i, item in enumerate(self.content):
                text += item.apply_chat_template(chat_template)
        else:
            raise NotImplementedError

        if self.role == "system":
            prompt = chat_template.decorate_system(text)
        elif self.role == "developer":
            prompt = chat_template.decorate_developer(text)
        elif self.role == "user":
            prompt = chat_template.decorate_user(text)
        elif self.role == "pretrain":
            prompt = text
        elif self.role == "assistant":
            prompt = ""
            if self.thinking is not None:
                prompt = chat_template.decorate_thinking(self.thinking)

            if chat_template.loss_assistant_format_mapping is not None and self.loss:
                old_prompt = chat_template.decorate_assistant(text)
                for k, v in chat_template.loss_assistant_format_mapping.items():
                    old_prompt = old_prompt.replace(k, v)
            else:
                old_prompt = chat_template.decorate_assistant(text)
            prompt += old_prompt
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
            "input_ids": token_ids,
            "labels": label_ids,
        }


def process_message(messages: List[ChatMsg], chat_template: ChatTemplate):
    if not messages:
        return messages

    if chat_template.default_system is not None and messages[0].role != "system":
        messages.insert(0, ChatMsg(role="system", content=chat_template.default_system, loss=False))

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

    def add(self, role, content, loss=False):
        self.messages.append(ChatMsg(role=role, content=content, loss=loss))

    def pop(self):
        return self.messages.pop()

    def get_prompt(self, chat_template: ChatTemplate) -> str:
        process_message(self.messages, chat_template)

        prompt = ""
        for msg in self.messages:
            prompt += msg.get_prompt(chat_template)
            if msg.role == "assistant":
                prompt += chat_template.sep
        return prompt

    def tokenize(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> Dict:
        input_ids = tokenizer.encode("", add_special_tokens=False)
        labels = [IGNORE_INDEX for _ in input_ids]

        process_message(self.messages, chat_template)

        for msg in self.messages:
            res = msg.tokenize(tokenizer, chat_template)
            token_ids, label_ids = res["input_ids"], res["labels"]

            input_ids.extend(token_ids)
            labels.extend(label_ids)

            if msg.role == "assistant":
                sep = chat_template.sep
                sep_tokens = tokenizer.encode(sep, add_special_tokens=False)
                input_ids.extend(sep_tokens)
                labels.extend([IGNORE_INDEX] * len(sep_tokens))

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
        ]
    }

    messages = ChatMessages.from_dict(data)
    chat_template = ChatTemplate(
        system="<|im_start|>system\n{system}<|im_end|>\n",
        user="<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        assistant="{assistant}<|im_end|>\n",
        stop_words=["<|im_end|>"],
    )

    print(messages.get_prompt(chat_template))
