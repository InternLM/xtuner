# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from typing import Any, Dict, List, Literal, Optional, Union

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
    conversation_timestamps: Optional[list[float] | list[list[float]]] = None

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
    image_wh: Optional[List[int]] = None  # width, height
    frames_timestamp: Optional[List[float]] = None
    origin_video_length: Optional[int] = None  # duration or frame count
    origin_fps: Optional[float] = None
    processed_video_length: Optional[int] = None
    processed_fps: Optional[float] = None
    video_length: Optional[int] = None  # deprecated


class VideoContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["video_url"] = "video_url"
    video_url: VideoURL

    def apply_chat_template(self, *args, **kwargs) -> str:
        return ""


MultModalContentType = Union[TextContentItem, ImageContentItem, VideoContentItem]
ContentType = Union[str, List[MultModalContentType]]


def tool_formatter(tools: list[dict[str, Any]]) -> str:
    tool_text = ""
    for tool in tools:
        wrapped_tool = tool if tool.get("type") == "function" else {"type": "function", "function": tool}
        tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)
    return tool_text


def function_formatter(functions: list[dict[str, Any]]) -> str:
    function_texts = []
    for function in functions:
        name = function["function"]["name"]
        arguments = function["function"]["arguments"]
        function_texts.append(json.dumps({"name": name, "arguments": json.loads(arguments)}, ensure_ascii=False))
    return "\n".join([f"<tool_call>\n{text}\n</tool_call>" for text in function_texts])


class ChatMsg(BaseModel):
    role: Literal["assistant", "user", "system", "developer", "pretrain", "tool"]
    model_config = ConfigDict(extra="forbid")
    content: ContentType
    reasoning_content: Optional[str] = None  # TODO: 暂时无效
    tool_calls: Optional[List[Dict]] = None
    loss: Optional[bool] = None
    thinking: Optional[str] = None  # only for assistant
    name: Optional[str] = None
    meta: Optional[Dict] = None
    tool_call_id: Optional[str] = None

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
            elif self.role == "tool":
                self.loss = False
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
        elif self.role == "tool":
            prompt = chat_template.decorate_tool_extractor(text)
        elif self.role == "assistant":
            if self.tool_calls is not None:
                function_text = function_formatter(self.tool_calls)
                if text is not None and text != "":
                    function_text = "\n" + function_text
                text += function_text

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


def process_message(messages: List[ChatMsg], chat_template: ChatTemplate, tools: Optional[List[Dict]] = None):
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

    if tools:
        assert chat_template.tool_prompt is not None, "tool_prompt must be set in chat_template."
        tool_text = tool_formatter(tools)
        tool_text = chat_template.tool_prompt.format(tool_text=tool_text)
        if messages[0].role != "system":
            messages.insert(0, ChatMsg(role="system", content=tool_text, loss=False))
        else:
            assert isinstance(messages[0].content, str), "system message content must be str."
            messages[0].content += tool_text


class ChatMessages(BaseMessages):
    messages: List[ChatMsg]
    tools: Optional[List[Dict]] = None

    def add(self, role, content, loss=False):
        self.messages.append(ChatMsg(role=role, content=content, loss=loss))

    def pop(self):
        return self.messages.pop()

    def get_prompt(self, chat_template: ChatTemplate) -> str:
        process_message(self.messages, chat_template, self.tools)

        prompt = ""
        for msg in self.messages:
            prompt += msg.get_prompt(chat_template)
            if msg.role == "assistant":
                prompt += chat_template.sep
        return prompt

    def tokenize(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate) -> Dict:
        input_ids = tokenizer.encode("", add_special_tokens=False)
        labels = [IGNORE_INDEX for _ in input_ids]

        process_message(self.messages, chat_template, self.tools)

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
