from typing import Optional

import torch
from loguru import logger

from ..config.config_consts import ENGINE_VLLM
from ..tokenizer import tokenizer_utils
from .base_model_server import BaseModelServer


class ActorModelServer(BaseModelServer):
    # Initialize
    def initialize_async(self):
        super().initialize_async()

        self.generator_eq_trainer = True
        # use trainer for self.generate() by default
        self.generator = self.trainer
        if 'generator_config' not in self.model_config:
            return  # self.generator = self.trainer

        generator_config = self.model_config['generator_config']  # optional
        if generator_config.get('shared_with_trainer', True):
            return  # self.generator = self.trainer

        generator_config['model_path'] = self.model_config['model_path']
        generator_config['tokenizer_config'] = self.tokenizer_config
        generator_config[
            'tokenizer_path'] = self.tokenizer_config.tokenizer_path
        generator_type = generator_config.get('generator_type', None)
        if generator_type == ENGINE_VLLM:
            from ..model_backend.vllm_model_runner import \
                VllmGeneratorRayActorGroup
            self.generator = VllmGeneratorRayActorGroup(
                f'{self.model_name}_generator', generator_config)
            # to sync model among trainer and generator
            self.trainer.initialize_get()
            self.trainer.init_process_group(self.generator)
        else:
            raise ValueError(
                f"No generator is registered with type '{generator_type}'")
        self.generator_eq_trainer = False

    def initialize_get(self):
        self.generator.initialize_get()
        self.is_initialized = True
        logger.info(f'{self.model_name} has been initialized. ')

    # Generation
    def generate_async(self,
                       inputs,
                       attention_mask=None,
                       *args,
                       **generate_kwargs):
        if isinstance(inputs, torch.Tensor):
            input_ids = inputs
        elif isinstance(inputs, list):
            if not self.generator_eq_trainer:
                input_ids, attention_mask = tokenizer_utils.encode(
                    inputs,
                    self.tokenizer,
                    return_tensors=None,
                    padding=False,
                    add_generation_prompt=True)
            else:
                input_ids, attention_mask = tokenizer_utils.encode(
                    inputs, self.tokenizer, add_generation_prompt=True)
        else:
            raise NotImplementedError(f'unknown inputs: {inputs}')

        return self.generator.generate_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **generate_kwargs)

    def generate_get(self, object_refs, timeout: Optional[float] = None):
        return self.generator.generate_get(object_refs, timeout=timeout)

    def generate(self, inputs, *args, **generate_kwargs):
        object_refs = self.generate_async(inputs, *args, **generate_kwargs)
        policy_output = self.generate_get(object_refs)
        self.log_cuda_mem_stats(remark='[generate] ')
        return policy_output

    # Sync
    def sync_model(self, *args, **kwargs):
        if not self.generator_eq_trainer:
            self.trainer.broadcast_model_to_generator(self.generator)

    # Misc.
    def log_cuda_mem_stats(self, remark=''):
        if self.show_cuda_mem_stats:
            trainer_mem = self.trainer.get_cuda_mem_stats()
            generator_mem = self.generator.get_cuda_mem_stats()
            logger.info(
                f'{remark}{self.model_name} trainer allocated GPU memory: {trainer_mem.total_current_mb} MiB, '  # noqa: E501
                f'generator allocated GPU memory: {generator_mem.total_current_mb} MiB, '  # noqa: E501
                f'generator_eq_trainer: {self.generator_eq_trainer}')
