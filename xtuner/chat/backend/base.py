from abc import abstractmethod
from typing import List, Optional

from xtuner.chat.streamer import SteamerType
from xtuner.types import (ChatBackendProtocol, ChatMessages, ChatTemplate,
                          SampleParams)


class BaseBackend(ChatBackendProtocol):

    @property
    def chat_template(self) -> ChatTemplate:
        pass

    @abstractmethod
    def create_streamer(self, iterable: bool = False) -> SteamerType:
        pass

    @abstractmethod
    def chat(self,
             messages: ChatMessages,
             sample_params: Optional[SampleParams] = None,
             streamer: Optional[SteamerType] = None):
        pass

    @abstractmethod
    def batch_infer(self,
                    messages: List[ChatMessages],
                    sample_params: Optional[SampleParams] = None):
        pass
