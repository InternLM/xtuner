from typing import Union

from .huggingface import HFTextIteratorStreamer, HFTextStreamer
from .lmdeploy import LMDeployTextIteratorStreamer, LMDeployTextStreamer

SteamerType = Union[HFTextIteratorStreamer, HFTextStreamer,
                    LMDeployTextIteratorStreamer, LMDeployTextStreamer]

__all__ = [
    'HFTextIteratorStreamer', 'HFTextStreamer', 'LMDeployTextIteratorStreamer',
    'LMDeployTextStreamer'
]
