import asyncio
import os
from typing import List, Optional, Union

from lmdeploy.utils import get_logger

from xtuner.types import HybridChatMessages, HybridChatTemplate, SampleParams
from ...streamer import LMDeployTextIteratorStreamer, LMDeployTextStreamer
from ..base import BaseBackend
from ._encoder import _AsyncEncoderWrapper
from ._engine import _MMAsyncEngine

os.environ['TM_LOG_LEVEL'] = 'ERROR'
logger = get_logger('lmdeploy')
logger.setLevel('ERROR')

_StreamerType = Union[LMDeployTextStreamer, LMDeployTextIteratorStreamer]


class LMDeployBackend(BaseBackend):

    def __init__(self,
                 chat_template,
                 llm_name_or_path,
                 vision_encoder=None) -> None:
        super().__init__()

        if vision_encoder:
            encoder = _AsyncEncoderWrapper(vision_encoder)
        else:
            encoder = None

        self._engine = _MMAsyncEngine(
            chat_template,
            encoder=encoder,
            model_path=llm_name_or_path,
            model_name='base')

        self._chat_template = chat_template

    @property
    def chat_template(self) -> HybridChatTemplate:
        return self._chat_template

    def create_streamer(self, iterable=False):

        if iterable:
            return LMDeployTextIteratorStreamer()
        else:
            return LMDeployTextStreamer()

    def parse_sample_params(self,
                            params: SampleParams) -> 'LMGenerationConfig':

        if params is None:
            params = SampleParams()

        stop_words = params.stop_words
        stop_words.extend(self.chat_template.stop_words)

        from lmdeploy.messages import GenerationConfig as LMDGenerationConfig
        lmd_gen_config = LMDGenerationConfig(
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            random_seed=params.seed,
            stop_words=stop_words)

        return lmd_gen_config

    def chat(self,
             messages: HybridChatMessages,
             streamer: Optional[_StreamerType] = None,
             sample_params: Optional[SampleParams] = None):

        lmd_gen_config = self.parse_sample_params(sample_params)
        self.session_id += 1
        import random

        generator = self._engine.generate(
            messages, random.randint(1, 100000), gen_config=lmd_gen_config)

        async def get_response():
            out = ''
            async for res in generator:
                out += res.response
                if streamer:
                    streamer.put(res.response)
            if streamer:
                streamer.end()
            return out

        loop = asyncio.new_event_loop()
        response = loop.run_until_complete(get_response())
        return response

    def batch_infer(self,
                    messages: List[HybridChatMessages],
                    sample_params: Optional[SampleParams] = None):

        lmd_gen_config = self.parse_sample_params(sample_params)

        results = self._engine.batch_infer(messages, gen_config=lmd_gen_config)

        return [r.text for r in results]
