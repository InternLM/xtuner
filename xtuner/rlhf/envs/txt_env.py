import torch
from loguru import logger

from ..model_server.base_model_server import BaseModelServer
from ..timer import Timer
from .base import EnvBase


class TxtEnv(EnvBase):
    """A generic RL environment to generate textual sequences."""

    def __init__(
        self,
        policy_model: BaseModelServer,
        max_new_tokens: int = 1024,
        policy_micro_bs: int = 32,
        generate_kwargs: dict = None,
        **_ignored,
    ):
        self.policy_model = policy_model
        self.max_new_tokens = max_new_tokens
        self.policy_micro_bs = policy_micro_bs
        self.generate_kwargs: dict = generate_kwargs

    def rollout(
        self, 
        prompt_input_messages, 
        pretrain_input_messages, 
        display=True
    ):
        # prompt data
        if display:
            logger.info(
                f'[TXT_ENV For Generate]: \n{prompt_input_messages[0]}')
        with Timer('policy_model.generate'):
            trajectories = self.policy_model.generate(
                inputs=prompt_input_messages,
                micro_batch_size=self.policy_micro_bs,
                step=self.max_new_tokens,
                output_str=True,
                generate_kwargs=self.generate_kwargs)
        logger.info(f'[Generate] len: {len(prompt_input_messages)}')

        # pretrain data
        if pretrain_input_messages is not None:
            from xtuner.rlhf.tokenizer import encode_inputs
            pt_input_ids, pt_attention_mask = encode_inputs(
                pretrain_input_messages, self.policy_model.tokenizer)
            pretrain_labels = torch.nn.functional.pad(
                pt_input_ids[:, 1:], (0, 1), mode='constant', value=-100)

            trajectories.pretrain_data = {
                'input_ids': pt_input_ids,
                'labels': pretrain_labels,
                'attention_mask': pt_attention_mask
            }
            logger.info(f'[TxtEnv] gets {pt_input_ids.shape} pretrain data.')
        else:
            trajectories.pretrain_data = None

        return trajectories

    def rollout_background(
        self, 
        prompt_input_messages, 
        pretrain_input_messages, 
        display=True
    ):
        self.pretrain_input_messages = pretrain_input_messages
        self.pretrain_idx = 0

        if display:
            logger.info(
                f'[TXT_ENV For Generate]: \n{prompt_input_messages[0]}')
        with Timer('txt_env.generate_background'):
            self.policy_model.generate_background(
                inputs=prompt_input_messages,
                micro_batch_size=self.policy_micro_bs,
                step=self.max_new_tokens,
                output_str=True,
                generate_kwargs=self.generate_kwargs)
        logger.info(f'[Generate] len: {len(prompt_input_messages)}')
    
    def rollout_get(self, num):
        # prompt data
        with Timer('txt_env.rollout_get'):
            trajectories = self.policy_model.get_generate_finish(num)
        
        # pretrain data
        # TODO: Get pretrain data proportionally
        if self.pretrain_input_messages is not None:
            assert self.pretrain_idx + num < len(self.pretrain_input_messages)
            pretrain_input_messages = self.pretrain_input_messages[
                self.pretrain_idx:self.pretrain_idx+num]
            #update pretrain idx
            self.pretrain_idx += num

            from xtuner.rlhf.tokenizer import encode_inputs
            pt_input_ids, pt_attention_mask = encode_inputs(
                pretrain_input_messages, self.policy_model.tokenizer)
            pretrain_labels = torch.nn.functional.pad(
                pt_input_ids[:, 1:], (0, 1), mode='constant', value=-100)

            trajectories.pretrain_data = {
                'input_ids': pt_input_ids,
                'labels': pretrain_labels,
                'attention_mask': pt_attention_mask
            }
            logger.info(f'[TxtEnv] gets {pt_input_ids.shape} pretrain data.')
        else:
            trajectories.pretrain_data = None
        
        return trajectories
