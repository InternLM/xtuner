# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
from .utils import (get_peft_model_state_dict, guess_load_checkpoint,
                    prepare_inputs_labels_for_multimodal)
from mmengine.model import BaseModel
import asyncio
from openai import AsyncOpenAI
from typing import List


class OpenaiBackend:

    def __init__(self, api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1'):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def request_completion(self, model_id, messages):
        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.8,
            top_p=0.8)
        return response  # .choices[0].message.content

    async def _batch_infer(self, messages: List[List]):
        model_cards = await self.client.models.list()._get_page()
        model_id = model_cards.data[0].id

        tasks = [self.request_completion(model_id, msg) for msg in messages]

        responses = await asyncio.gather(*tasks)

        return [res.choices[0].message.content for res in responses]

    def batch_infer(self, messages: List[List]):
        return asyncio.run(self._batch_infer(messages))


class OpenAIModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = OpenaiBackend(base_url='http://10.140.24.142:23333/v1')

    def forward(self, data, data_samples=None, mode='loss'):
        pixel_values = data['pixel_values'][0]
        text = data['text'][0]

        prompt = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': text},
                    {'type': 'image_url', 'image_url': {'url': pixel_values}}
                ]
            }
        ]
        prediction = self.model.batch_infer([prompt])[0]
        return dict(prediction=prediction)

    def gradient_checkpointing_disable(self):
        pass
    def preparing_for_generation(self, metainfo: dict = None):
        pass
