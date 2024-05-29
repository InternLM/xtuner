from typing import Optional

import ray
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from ..config.config_consts import ENGINE_HUGGINGFACE, ENGINE_INTERNEVO
from ..model_backend.hf_model_runner import HfModelRunnerRayActorGroup
from ..model_backend.models.modeling_internlm2_p import InternLM2ForCausalLM
from ..tokenizer import tokenizer_utils

DEFAULT_GET_TIMEOUT = 600.0  # 10 min


class BaseModelServer:
    # Initialize
    def __init__(self, model_name: str, model_config: dict):
        self.model_name = model_name
        self.model_config = model_config
        self.tokenizer = None
        self.tokenizer_config = None
        self.trainer = None
        self.trainer_config = None
        self.model_ref = None
        self.is_initialized = False
        self.show_cuda_mem_stats = self.model_config.get(
            'show_cuda_mem_stats', False)
        logger.info(f'model_name={model_name}, model_config={model_config}')

    def init_tokenizer_and_config(self, model_config):
        tokenizer_config = model_config.get('tokenizer_config', {})
        if 'tokenizer_path' in tokenizer_config:
            tokenizer_path = tokenizer_config['tokenizer_path']
        elif 'tokenizer_path' in model_config:
            tokenizer_path = model_config['tokenizer_path']
        else:
            tokenizer_path = model_config['model_path']

        self.tokenizer = tokenizer_utils.get_tokenizer(
            tokenizer_path, trust_remote_code=True, **tokenizer_config)

        tokenizer_config['tokenizer_path'] = tokenizer_path
        tokenizer_config['pad_token_id'] = self.tokenizer.pad_token_id
        self.tokenizer_config = tokenizer_config

    def init_trainer_config(self, model_config, tokenizer_config):
        model_path = model_config['model_path']
        trainer_config: dict = model_config['trainer_config']  # requisite
        trainer_config['tokenizer_config'] = tokenizer_config
        trainer_config['tokenizer_path'] = tokenizer_config['tokenizer_path']
        trainer_config['model_path'] = model_path
        trainer_config['model_type'] = model_config['model_type']
        trainer_config['model_class'] = self.get_model_class(model_path)
        self.trainer_config = trainer_config

    def get_model_class(self, model_path):
        # will be changed in subclasses
        if model_path == 'internlm/internlm2-chat-1_8b-sft':
            return InternLM2ForCausalLM
        return AutoModelForCausalLM

    def initialize_async(self):
        self.init_tokenizer_and_config(self.model_config)
        self.init_trainer_config(self.model_config, self.tokenizer_config)

        trainer_type = self.trainer_config.get('trainer_type',
                                               'huggingface').lower()
        if trainer_type == ENGINE_HUGGINGFACE:
            self.trainer = HfModelRunnerRayActorGroup(
                name=f'{self.model_name}_trainer', config=self.trainer_config)
        elif trainer_type == ENGINE_INTERNEVO:
            raise NotImplementedError(f'{trainer_type}.')
        else:
            raise ValueError(
                f'No trainer is registered with type: {trainer_type}')

    def initialize_get(self):
        self.trainer.initialize_get()
        self.is_initialized = True
        logger.info(f'{self.model_name} has been initialized.')

    # Inference
    def infer_async(self, inputs, attention_mask=None, *args, **infer_kwargs):
        if not isinstance(inputs, torch.Tensor):
            input_ids, attention_mask = tokenizer_utils.encode(
                inputs, self.tokenizer)
        else:
            input_ids = inputs
        return self.trainer.infer_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **infer_kwargs)

    def infer_get(self, object_refs, timeout: Optional[float] = None):
        return self.trainer.infer_get(object_refs, timeout=timeout)

    def infer(self, inputs, *args, **infer_kwargs):
        object_refs = self.infer_async(inputs, *args, **infer_kwargs)
        results = self.infer_get(object_refs)
        self.log_cuda_mem_stats(remark='[infer] ')
        return results

    # Training
    def train_async(self,
                    input_ids,
                    labels=None,
                    attention_mask=None,
                    *args,
                    **train_kwargs):
        return self.trainer.train_async(input_ids, labels, attention_mask,
                                        *args, **train_kwargs)

    def train_get(self, object_refs, timeout: Optional[float] = None):
        return self.trainer.train_get(object_refs, timeout=timeout)

    def train(self,
              input_ids,
              labels=None,
              attention_mask=None,
              *args,
              **train_kwargs):
        object_refs = self.train_async(input_ids, labels, attention_mask,
                                       *args, **train_kwargs)
        loss = self.train_get(object_refs)
        self.log_cuda_mem_stats(remark='[train] ')
        return loss

    # Generation
    def generate_async(self,
                       inputs,
                       attention_mask=None,
                       *args,
                       **generate_kwargs):
        raise NotImplementedError

    def generate_get(self, object_refs, timeout: Optional[float] = None):
        raise NotImplementedError

    def generate(self, inputs, *args, **generate_kwargs):
        raise NotImplementedError

    # Model
    def model_get(self):
        if not self.model_ref:
            self.model_ref = self.trainer.get_model()  # an reference
        return ray.get(self.model_ref, timeout=DEFAULT_GET_TIMEOUT)

    def state_dict_get(self):
        return ray.get(
            self.trainer.get_state_dict(), timeout=DEFAULT_GET_TIMEOUT)

    def save_model(self, path):
        self.trainer.save_model(path)

    # Misc.
    def set_seed(self, seed: int = None):
        self.trainer.set_seed(seed)

    def log_cuda_mem_stats(self, remark=''):
        if self.show_cuda_mem_stats:
            trainer_mem = self.trainer.get_cuda_mem_stats()
            logger.info(
                f'{remark}{self.model_name} trainer allocated GPU memory: {trainer_mem.total_current_mb} MiB'  # noqa: E501
            )

    def clean_up(self):
        self.trainer.release_resources()
        logger.info(f'{self.model_name} is destroyed.')
