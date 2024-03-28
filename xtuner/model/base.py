# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractclassmethod, abstractmethod

from mmengine.model import BaseModel

from xtuner.types import HybridChatMessages, HybridChatTemplate


class BaseTune(BaseModel):

    def __init__():
        super().__init__()

    def init_weights(self):
        """Parent class method.

        To avoid overwriting the loaded weights, overload it to an empty
        function.
        """
        pass

    def avoid_override_weights(self):
        self._is_init = True

    @abstractmethod
    @property
    def chat_template(self) -> HybridChatTemplate:
        pass

    @abstractmethod
    @property
    def llm(self):
        pass

    @abstractmethod
    @property
    def tokenizer(self):
        pass

    @abstractmethod
    def gradient_checkpointing_enable(self):
        pass

    def forward(self, data, data_samples=None, mode='loss'):
        """Overload parent class method, only support training."""

        if mode == 'loss':
            return self.compute_loss(data)
        else:
            raise NotImplementedError(
                f"{type(self)}'s forward is only supported for use during "
                'training. If you want to get predictions or chat, please '
                "directly use `llm`'s forward.")

    @abstractmethod
    def chat(self, messages: HybridChatMessages, sample_params, streamer):
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs) -> 'BaseTune':
        pass

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
