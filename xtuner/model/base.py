# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from mmengine.model import BaseModel

from xtuner.types import (ChatBackendProtocol, ChatMessages, ChatTemplate,
                          SampleParams)


class BaseTune(BaseModel, ChatBackendProtocol):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass

    def avoid_override_weights(self):
        self._is_init = True

    @property
    @abstractmethod
    def chat_template(self) -> ChatTemplate:
        pass

    @property
    @abstractmethod
    def llm(self):
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        pass

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
                        from_hub: bool = False) -> 'BaseTune':
        pass
