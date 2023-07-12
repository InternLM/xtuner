
from typing import Dict
from torch import nn
from mmengine.model import BaseModel
from mmengine import Config
from mmchat.registry import MODELS, TOKENIZER, LLM
import torch
import transformers
import dataclasses
from mmengine import print_log

DEFAULT_PAD_TOKEN = "[PAD]"

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
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedFinetune(BaseModel):

    def __init__(self, llm, data_preprocessor):
        super().__init__(data_preprocessor)
        self.llm = self._build_from_cfg_or_module(llm, LLM)
        self.llm.config.use_cache = False
        self.llm.config.torch_dtype = torch.float32
        if self.tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=self.tokenizer,
                model=self.llm,
            )
        from transformers.models.llama import LlamaTokenizer
        
        if  isinstance(self.tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
            print('Adding special tokens.')
            self.tokenizer.add_special_tokens({
                    "eos_token": self.tokenizer.convert_ids_to_tokens(self.llm.config.eos_token_id),
                    "bos_token": self.tokenizer.convert_ids_to_tokens(self.llm.config.bos_token_id),
                    "unk_token": self.tokenizer.convert_ids_to_tokens(
                        self.llm.config.pad_token_id if self.llm.config.pad_token_id != -1 else self.tokenizer.pad_token_id
                    ),
            })

    @property
    def tokenizer(self):
        return self.data_preprocessor.tokenizer

        
    def _build_from_cfg_or_module(self, cfg_or_mod, registry):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return registry.build(cfg_or_mod)
        else:
            raise NotImplemented
        
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
        
        return outputs


    def compute_loss(self, data, data_samples=None):
        
        outputs = self.llm(**data)
        # import pdb;pdb.set_trace()
        loss_dict = {'loss_llm': outputs.loss}
        return loss_dict
        
    