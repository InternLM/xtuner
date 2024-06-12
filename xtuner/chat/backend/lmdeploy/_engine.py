# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.async_engine import AsyncEngine

from xtuner.types import ChatMessages, ChatTemplate


class _AsyncEngine(AsyncEngine):
    """Async inference engine."""

    def __init__(self, chat_template: ChatTemplate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.model_name == 'base'
        self.chat_template = chat_template

    async def _get_prompt_input(self, prompt: ChatMessages,
                                do_preprocess: bool, sequence_start: bool):
        """get input_ids, embeddings and offsets."""

        decorated = prompt.get_prompt(self.chat_template)

        results = {}

        input_ids = self.tokenizer.encode(decorated, add_bos=sequence_start)

        results['input_ids'] = input_ids
        results['prompt'] = decorated
        return results
