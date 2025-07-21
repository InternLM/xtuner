from typing import TypedDict

from typing_extensions import Required


class DataItem(TypedDict, total=False):
    input_ids: Required[list[int]]
    labels: Required[list[int]]
    num_tokens: Required[int]
