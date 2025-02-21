# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from xtuner._lite import get_logger
from xtuner.utils import IGNORE_INDEX

from ..templates import ChatTemplate, HybridChatTemplate
from .base import BaseMessages

logger = get_logger()


class TextContentItem(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return self.text


class ImageContentItem(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: str

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return chat_template.image_token


MultModalContentType = Union[TextContentItem, ImageContentItem]
ContentType = Union[str, List[MultModalContentType]]


class ChatMsg(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: ContentType
    loss: Optional[bool] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.loss is None:
            if self.role == "system":
                self.loss = False
            elif self.role == "user":
                self.loss = False
            elif self.role == "assistant":
                self.loss = True
            else:
                raise NotImplementedError

    def collect_img_urls(self) -> List[str]:
        img_urls = []
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, ImageContentItem):
                    img_urls.append(item.image_url)
        return img_urls

    def get_prompt(self, chat_template: ChatTemplate) -> str:
        if isinstance(self.content, str):
            text = self.content
        elif isinstance(self.content, list):
            text = ""
            for i, item in enumerate(self.content):
                if i == 0:
                    text += item.apply_chat_template(chat_template)
                else:
                    text += "\n" + item.apply_chat_template(chat_template)
        else:
            raise NotImplementedError

        if self.role == "system":
            prompt = chat_template.decorate_system(text)
        elif self.role == "user":
            prompt = chat_template.decorate_user(text)
        elif self.role == "assistant":
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
            "input_ids": token_ids,
            "labels": label_ids,
        }


class ChatMessages(BaseMessages):
    messages: List[ChatMsg]

    def add(self, role, content, loss=False):
        self.messages.append(ChatMsg(role=role, content=content, loss=loss))

    def pop(self):
        return self.messages.pop()

    def get_prompt(self, chat_template: ChatTemplate) -> str:
        prompt = ""

        for msg in self.messages:
            prompt += msg.get_prompt(chat_template)
            if msg.role == "assistant":
                prompt += chat_template.sep
        return prompt

    def tokenize(
        self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate
    ) -> Dict:
        input_ids = tokenizer.encode("", add_special_tokens=True)
        labels = [IGNORE_INDEX for _ in input_ids]
        image_urls = []

        for msg in self.messages:
            res = msg.tokenize(tokenizer, chat_template)
            token_ids, label_ids = res["input_ids"], res["labels"]

            input_ids.extend(token_ids)
            labels.extend(label_ids)

            image_urls.extend(msg.collect_img_urls())

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
                "The lengths of input_ids and labels must be "
                f"equal, but  found {len(input_ids)} and "
                f"{len(labels)}."
            )

        training_data = {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": len(input_ids),
        }

        if len(image_urls) > 0:
            training_data["image_urls"] = image_urls

        return training_data

    @classmethod
    def from_str(cls, prompt: str) -> "ChatMessages":
        msg = ChatMsg(role="user", content=prompt)
        return cls(messages=[msg])

    @classmethod
    def from_dict(cls, item: dict) -> "ChatMessages":
        """
        item
        {
            'messages':[
                {'role':'user', 'content':'hello'},
                {'role':'assistant', 'content':'hello!'},
            ],
        }
        """
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
