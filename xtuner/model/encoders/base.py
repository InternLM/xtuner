# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractclassmethod, abstractmethod
from typing import List, Union

import torch
from PIL import Image
from torch import nn

_ImageType = Union[str, Image.Image]


class EncoderWrapper(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    @property
    def encoder(self):
        pass

    @abstractmethod
    @property
    def projector(self):
        pass

    @abstractmethod
    def post_init_proj(self, llm):
        pass

    @abstractmethod
    def preprocess(self, image: _ImageType) -> torch.Tensor:
        pass

    @abstractmethod
    def batch_infer(images: List[_ImageType]) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def gradient_checkpointing_enable(self):
        pass

    @abstractclassmethod
    def save_checkpoint(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def load_checkpoint(self, *args, **kwargs) -> 'EncoderWrapper':
        pass

    @abstractclassmethod
    def only_build_processor(self, *args, **kwargs):
        pass
