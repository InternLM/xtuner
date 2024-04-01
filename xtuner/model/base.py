# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractclassmethod, abstractmethod

from mmengine.model import BaseModel

from xtuner.types import ChatBackendProtocol, ChatMessages, SampleParams


class BaseAlgorithm(BaseModel, ChatBackendProtocol):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass

    def avoid_override_weights(self):
        self._is_init = True

    @abstractmethod
    def gradient_checkpointing_enable(self):
        pass

    @abstractmethod
    def chat(self, messages: ChatMessages, sample_params: SampleParams,
             streamer):
        pass

    @abstractmethod
    def batch_infer(self, messages: ChatMessages, sample_params: SampleParams,
                    streamer):
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir: str, to_hub: bool = False):
        pass

    @abstractmethod
    def load_checkpoint(self,
                        ckpt_dir: str,
                        from_hub: bool = False) -> 'BaseAlgorithm':
        pass

    @abstractclassmethod
    def dataloader_collate_fn(cls, data):
        pass


class BaseTune(BaseAlgorithm):

    pass
