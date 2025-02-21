# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractclassmethod, abstractmethod
from typing import Dict

from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from ..templates import ChatTemplate


class BaseMessages(BaseModel):
    @abstractmethod
    def add(self, role: str, content):
        pass

    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def get_prompt(self, chat_template: ChatTemplate) -> str:
        pass

    @abstractmethod
    def tokenize(
        self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate
    ) -> Dict:
        pass

    @abstractclassmethod
    def from_dict(cls, item: Dict) -> "BaseMessages":
        pass
