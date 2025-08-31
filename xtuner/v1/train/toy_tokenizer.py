from typing import Literal, Union

import numpy as np
import torch

from transformers.tokenization_utils_base import BatchEncoding


def _replace_token_ids(text, ids, special_str, special_id):
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
    """A simple byte-level tokenizer that encodes text as UTF-8 bytes with
    special token handling.

    This tokenizer converts text into a sequence of byte values (0-255) and supports
    special tokens for text boundaries and image-related placeholders. It provides
    basic encoding/decoding functionality compatible with transformers' tokenizer interface.

    Args:
        bos_token_id (int, optional): Beginning of sequence token ID. Defaults to 256.
        eos_token_id (int, optional): End of sequence token ID. Defaults to 257.
        pad_token_id (int, optional): Padding token ID. Defaults to 258.
        image_start_id_map (dict[str, int], optional): Mapping from image start string
            to token ID. Defaults to {"<img>": 259}.
        image_context_id_map (dict[str, int], optional): Mapping from image context string
            to token ID. Defaults to {"<IMG_CONTEXT>": 260}.
        image_end_id_map (dict[str, int], optional): Mapping from image end string
            to token ID. Defaults to {"</img>": 261}.
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
        """Encode text into a sequence of token IDs.

        Converts text to UTF-8 bytes and replaces special image tokens with their
        corresponding IDs. Optionally adds BOS/EOS tokens.

        Args:
            text (str): The text to encode.
            add_special_tokens (bool, optional): Whether to add BOS/EOS tokens. Defaults to False.

        Returns:
            list[int]: List of token IDs representing the encoded text.
        """
        # 严格 UTF-8 编码，遇到非法字符会抛错；如需忽略可用 errors='ignore'
        b = text.encode("utf-8", errors="strict")
        ids = list(b)  # 每个字节 0..255

        ids = _replace_token_ids(text, ids, self.image_start_str, self.image_start_id)
        ids = _replace_token_ids(text, ids, self.image_context_str, self.image_context_id)
        ids = _replace_token_ids(text, ids, self.image_end_str, self.image_end_id)

        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs back to text.

        Converts byte-level token IDs back to UTF-8 text. Special tokens are either
        skipped or kept as-is based on skip_special_tokens parameter.

        Args:
            ids (list[int]): List of token IDs to decode.
            skip_special_tokens (bool, optional): Whether to skip special tokens during decoding. Defaults to True.

        Returns:
            str: The decoded text.

        Raises:
            ValueError: If any token ID is negative.
        """
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

    def _convert_tokens_to_ids(self, token: str) -> Union[int, list[int]]:
        ids = self.encode(token)
        if len(ids) == 1:
            return ids[0]
        return ids

    def __call__(self, text: list[str], return_tensors: Literal["np", "pt"] = "np", **kwargs):
        """Encode a batch of texts into token IDs with tensor format.

        Processes multiple texts simultaneously and returns them in the specified
        tensor format compatible with transformers library.

        Args:
            text (list[str]): List of texts to encode.
            return_tensors (Literal["np", "pt"], optional): Format of returned tensors.
                "np" for numpy arrays, "pt" for PyTorch tensors. Defaults to "np".
            **kwargs: Additional arguments passed to encode method, including
                add_special_tokens (bool).

        Returns:
            BatchEncoding: Object containing encoded token IDs with the specified tensor type.
        """
        batch_ids = []
        for _text in text:
            ids = self.encode(_text, add_special_tokens=kwargs.get("add_special_tokens", False))
            if return_tensors == "np":
                ids = np.array(ids, dtype=np.int32)
            elif return_tensors == "pt":
                ids = torch.tensor(ids, dtype=torch.int32)  # type: ignore
            batch_ids.append(ids)

        return BatchEncoding({"input_ids": batch_ids}, tensor_type=return_tensors)
