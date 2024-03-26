# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX

from xtuner.types import HybridChatMessages, HybridChatTemplate


class _MMAsyncEngine(AsyncEngine):
    """Visual Language Async inference engine."""

    def __init__(self,
                 chat_template: HybridChatTemplate,
                 *args,
                 encoder=None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.model_name == 'base'
        self.encoder = encoder
        self.chat_template = chat_template

    async def _get_prompt_input(self, prompt: HybridChatMessages,
                                do_preprocess: bool, sequence_start: bool):
        """get input_ids, embeddings and offsets."""

        decorated = prompt.apply_chat_template(self.chat_template)
        segs = decorated.split(self.chat_template.image_token)

        results = {}
        input_ids = []
        if len(segs) > 1:
            assert self.encoder is not None
            img_urls = prompt.collect_img_urls()
            features = await self.encoder.async_infer(img_urls)
            features = [x.cpu().numpy() for x in features]
            input_ids = []
            begins = []
            ends = []
            for i, seg in enumerate(segs):
                if i > 0:
                    image_dim = features[i - 1].shape[0]
                    begins.append(len(input_ids))
                    ends.append(begins[-1] + image_dim)
                    input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
                seg_ids = self.tokenizer.encode(
                    seg, add_bos=((i == 0) and sequence_start))
                input_ids.extend(seg_ids)
            ranges = np.stack([begins, ends], axis=1).tolist()
            results['input_embeddings'] = features
            results['input_embedding_ranges'] = ranges
        else:
            input_ids = self.tokenizer.encode(
                decorated, add_bos=sequence_start)

        results['input_ids'] = input_ids
        results['prompt'] = decorated
        return results

    # def batch_infer(self, prompts: Union[VLPromptType, List[Dict],
    #                                      List[VLPromptType], List[List[Dict]]],
    #                 **kwargs):
    #     """Inference a batch of prompts."""
    #     # prompts = self._convert_prompts(prompts)
    #     return super().batch_infer(prompts, **kwargs)

    # def stream_infer(self, prompts: Union[VLPromptType, List[Dict],
    #                                       List[VLPromptType],
    #                                       List[List[Dict]]], **kwargs):
    #     """Inference a batch of prompts with stream mode."""
    #     # prompts = self._convert_prompts(prompts)
    #     return super().stream_infer(prompts, **kwargs)

    # def __call__(self, prompts, **kwargs):
    #     """Inference a batch of prompts."""
    #     # prompts = self._convert_prompts(prompts)
    #     return super().__call__(prompts, **kwargs)

    # def chat(self, prompts: VLPromptType, **kwargs):
    #     """chat."""
    #     # _prompts = self._convert_prompts(prompts)
    #     sess = super().chat(_prompts, **kwargs)

    #     # recover prompts & history
    #     sess._prompt = prompts
    #     last_round = sess.history[-1]
    #     sess.history[-1] = (prompts, last_round[-1])
    #     return sess
