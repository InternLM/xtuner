from typing import List, Optional, Protocol

from xtuner.chat.streamer import SteamerType
from xtuner.types.chat_template import ChatTemplate
from xtuner.types.messages import ChatMessages
from xtuner.types.sample_params import SampleParams


class ChatBackendProtocol(Protocol):

    @property
    def chat_template(self) -> ChatTemplate:
        pass

    def create_streamer(self, iterable=False) -> SteamerType:
        ...

    def chat(self,
             messages: ChatMessages,
             sample_params: Optional[SampleParams] = None,
             streamer: Optional[SteamerType] = None):
        ...

    def batch_infer(self,
                    messages: List[ChatMessages],
                    sample_params: Optional[SampleParams] = None,
                    streamer: Optional[SteamerType] = None):
        ...
