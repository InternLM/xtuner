import dataclasses

import torch
import transformers
from mmengine import print_log
from mmengine.model import BaseModel
from torch import nn

from mmchat.registry import LLM, TOKENIZER


def traverse_dict(d):
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                # print(key)
                if 'type' in value and dataclasses.is_dataclass(value['type']):
                    builder = value.pop('type')
                    new_value = builder(**value)
                    d[key] = new_value
                    print_log(f'{key} convert to {builder}')
                else:
                    traverse_dict(value)
    elif isinstance(d, list):
        for element in d:
            traverse_dict(element)


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size
    not be divisible by 64.
    """
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tokenizer))
    num_new_tokens = len(tokenizer) - model_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    elif num_new_tokens < 0:
        raise RuntimeError


class SupervisedFinetune(BaseModel):

    def __init__(self, llm, data_preprocessor=None, tokenizer=None):
        super().__init__(data_preprocessor)
        self.llm = self._build_from_cfg_or_module(llm, LLM)
        self.llm.config.use_cache = False
        self.llm.config.torch_dtype = torch.float32
        tokenizer = TOKENIZER.build(tokenizer)
        smart_tokenizer_and_embedding_resize(tokenizer, self.llm)

    def _build_from_cfg_or_module(self, cfg_or_mod, registry):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return registry.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        # import pdb;pdb.set_trace()
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
