from abc import abstractmethod

from xtuner.types import HybridChatTemplate


class BaseBackend():

    @property
    def chat_template(self) -> HybridChatTemplate:
        pass

    @abstractmethod
    def create_streamer(self, iterable=False):
        pass

    @abstractmethod
    def chat(self, messages, streamer=None, generation_config=None):
        pass

    # @abstractmethod
    # def response_with_function_call(self, response: str):
    #     pass

    # @abstractmethod
    # def response_with_code_interpreter(self, response: str):
    #     pass
