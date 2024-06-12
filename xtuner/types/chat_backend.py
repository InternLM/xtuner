from typing import List, Optional, Protocol, Union

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
             prompt_or_messages: Union[str, ChatMessages],
             sample_params: Optional[SampleParams] = None,
             streamer: Optional[SteamerType] = None) -> str:
        ...

    def batch_infer(self,
                    prompt_or_messages_list: Union[List[str],
                                                   List[ChatMessages]],
                    sample_params: Optional[SampleParams] = None,
                    streamer: Optional[SteamerType] = None) -> List[str]:
        ...
