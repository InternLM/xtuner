# Copyright (c) OpenMMLab. All rights reserved.
import re

from xtuner._lite.chat import ChatMessages


class XTunerFormat2Openai:
    @classmethod
    def source_format(cls):
        data = {
            "conversation": [
                {"system": "SYSTEM", "input": "INPUT", "output": "OUTPUT"},
                {"input": "INPUT", "output": "OUTPUT"},
            ]
        }
        return data

    @classmethod
    def target_format(cls):
        data = {
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "INPUT"},
                {"role": "assistant", "content": "OUTPUT"},
                {"role": "user", "content": "INPUT"},
                {"role": "assistant", "content": "OUTPUT"},
            ]
        }
        return data

    @staticmethod
    def convert(data):
        ROLE_MAPPING = {"system": "system", "input": "user", "output": "assistant"}
        messages = []
        for single_turn_conversation in data["conversation"]:
            for role, content in single_turn_conversation.items():
                messages.append({"role": ROLE_MAPPING[role], "content": content})
        return ChatMessages.from_dict({"messages": messages})


class Alpaca2Openai:
    @classmethod
    def source_format(cls):
        data = {
            "instruction": "INSTRUCTION",
            "input": "INPUT",
            "output": "OUTPUT",
        }
        return data

    @classmethod
    def target_format(cls):
        data = {
            "messages": [
                {"role": "user", "content": "INSTRUCTION\nINPUT"},
                {"role": "assistant", "content": "OUTPUT"},
            ]
        }
        return data

    @staticmethod
    def convert(data):
        if data.get("output") == "<nooutput>":
            return ChatMessages.from_dict({"messages": []})
        else:
            return ChatMessages.from_dict(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{data['instruction']}\n{data['input']}",
                        },
                        {"role": "assistant", "content": f"{data['output']}"},
                    ]
                }
            )


def llava_to_openai(data):
    image_token = "<image>"
    conversations = data["conversations"]
    messages = []

    if "image" in data:
        image_urls = data["image"]
        if isinstance(image_urls, str):
            image_urls = [image_urls]
    else:
        image_urls = None

    while conversations and conversations[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        conversations = conversations[1:]

    image_id = 0
    for convs in conversations:
        if convs["from"] == "human":
            pattern = f"({image_token})"
            chunks = re.split(pattern, convs["value"])

            text_content = []
            img_content = []

            for chunk in chunks:
                if chunk == image_token:
                    url = image_urls[image_id]
                    if not isinstance(url, str):
                        raise TypeError(data)
                    # assert , image_url
                    item = dict(type="image_url", image_url=url)
                    img_content.append(item)
                    image_id += 1
                elif len(chunk.strip()):
                    item = dict(type="text", text=chunk.strip())
                    text_content.append(item)

            msg = {"role": "user", "content": img_content + text_content}
            messages.append(msg)

        elif convs["from"] == "gpt":
            msg = {"role": "assistant", "content": convs["value"]}
            messages.append(msg)
        else:
            raise NotImplementedError

    return ChatMessages.from_dict({"messages": messages})


def llava_to_openai_interleave(data):
    image_token = "<image>"
    conversations = data["conversations"]
    messages = []

    if "image" in data:
        image_urls = data["image"]
        if isinstance(image_urls, str):
            image_urls = [image_urls]
    else:
        image_urls = None

    while conversations and conversations[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        conversations = conversations[1:]

    image_id = 0
    for convs in conversations:
        if convs["from"] == "human":
            pattern = f"({image_token})"
            chunks = re.split(pattern, convs["value"])

            content = []

            for chunk in chunks:
                if chunk == image_token:
                    url = image_urls[image_id]
                    if not isinstance(url, str):
                        raise TypeError(data)
                    # assert , image_url
                    item = dict(type="image_url", image_url=url)
                    content.append(item)
                    image_id += 1
                elif len(chunk.strip()):
                    item = dict(type="text", text=chunk.strip())
                    content.append(item)

            msg = {"role": "user", "content": content}
            messages.append(msg)

        elif convs["from"] == "gpt":
            msg = {"role": "assistant", "content": convs["value"]}
            messages.append(msg)
        else:
            raise NotImplementedError

    return ChatMessages.from_dict({"messages": messages})


def official_openai(data):
    if "messages" in data:
        return ChatMessages.from_dict(data)
    elif "message_data" in data:
        return ChatMessages.from_dict({"messages": data["message_data"]})
    elif "dialogs" in data:
        return ChatMessages.from_dict({"messages": data["dialogs"]})
    else:
        return ChatMessages.from_dict({"messages": data})


OPENAI_CONVERT_MAP = {
    "llava": llava_to_openai,
    "llava_interleave": llava_to_openai_interleave,
    "alpaca": Alpaca2Openai.convert,
    "xtuner": XTunerFormat2Openai.convert,
    "openai": official_openai,
}
