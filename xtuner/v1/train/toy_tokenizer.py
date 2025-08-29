from typing import Literal, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import BatchEncoding


def replace_token_ids(text, ids, special_str, special_id):
    if special_str in text:
        _positions = []
        start = 0
        while True:
            pos = text.find(special_str, start)
            if pos == -1:
                break
            _positions.append(pos)
            start = pos + len(special_str)

        new_ids = []
        current_pos = 0

        for _pos in _positions:
            _byte = len(text[:_pos].encode("utf-8"))
            new_ids.extend(ids[current_pos:_byte])
            new_ids.append(special_id)
            current_pos = _byte + len(special_str.encode("utf-8"))

        new_ids.extend(ids[current_pos:])
        ids = new_ids
    return ids


class UTF8ByteTokenizer:
    """字节级 UTF-8 tokenizer：

    - 普通字节的 token id 范围为 [0, 255]
    - 可选特殊符号会占用 256 以上的 id
    """

    def __init__(
        self,
        bos_token_id: int | None = 256,
        eos_token_id: int | None = 257,
        pad_token_id: int | None = 258,
        image_start_id_map: dict[str, int] | None = None,
        image_context_id_map: dict[str, int] | None = None,
        image_end_id_map: dict[str, int] | None = None,
    ):
        if image_start_id_map is None:
            image_start_id_map = {"<img>": 259}
        if image_context_id_map is None:
            image_context_id_map = {"<IMG_CONTEXT>": 260}
        if image_end_id_map is None:
            image_end_id_map = {"</img>": 261}

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.image_start_str = list(image_start_id_map.keys())[0]
        self.image_context_str = list(image_context_id_map.keys())[0]
        self.image_end_str = list(image_end_id_map.keys())[0]
        self.image_start_id = list(image_start_id_map.values())[0]
        self.image_context_id = list(image_context_id_map.values())[0]
        self.image_end_id = list(image_end_id_map.values())[0]

        self.special_ids = {x for x in [bos_token_id, eos_token_id, pad_token_id] if x is not None}
        self.vocab_size = 256 + len(self.special_ids) + 3

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 严格 UTF-8 编码，遇到非法字符会抛错；如需忽略可用 errors='ignore'
        b = text.encode("utf-8", errors="strict")
        ids = list(b)  # 每个字节 0..255

        ids = replace_token_ids(text, ids, self.image_start_str, self.image_start_id)
        ids = replace_token_ids(text, ids, self.image_context_str, self.image_context_id)
        ids = replace_token_ids(text, ids, self.image_end_str, self.image_end_id)

        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        bytes_list = []
        for t in ids:
            if t < 0:
                raise ValueError(f"invalid token id: {t}")
            if t < 256:
                bytes_list.append(t)
            else:
                if not skip_special_tokens and t in self.special_ids:
                    # 特殊符号默认不还原为字符；如有需要可自定义占位符
                    pass
                # 其余 >=256 的非特殊 id 直接忽略
        return bytes(bytes_list).decode("utf-8", errors="strict")

    def convert_tokens_to_ids(self, token: str) -> Union[int, list[int]]:
        ids = self.encode(token)
        if len(ids) == 1:
            return ids[0]
        return ids

    def __call__(self, text: list[str], return_tensors: Literal["np", "pt"] = "np", **kwargs):
        batch_ids = []
        for _text in text:
            ids = self.encode(_text, add_special_tokens=kwargs.get("add_special_tokens", False))
            if return_tensors == "np":
                ids = np.array(ids, dtype=np.int32)
            elif return_tensors == "pt":
                ids = torch.tensor(ids, dtype=torch.int32)  # type: ignore
            batch_ids.append(ids)

        return BatchEncoding({"input_ids": batch_ids}, tensor_type=return_tensors)
