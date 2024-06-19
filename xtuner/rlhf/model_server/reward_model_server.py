import torch
from transformers import AutoConfig

from ..tokenizer import encode_inputs
from ..utils import expand_reward_token_id
from .base_model_server import BaseModelServer
from .utils import get_reward_model


class RewardModelServer(BaseModelServer):
    # Initialize
    def get_model_class(self, model_path):
        head_name = self.model_config.get('head_name', 'v_head')
        return get_reward_model(model_path, head_name)

    def init_tokenizer_and_config(self, model_config):
        super().init_tokenizer_and_config(self.model_config)

        # specify `reward_token_id`` to get scalar reward of a sequence
        # according to the `Rward Model` training strategy,
        # which is set to `pad_token_id` by default
        self.reward_token_id = self.tokenizer.pad_token_id
        model_path = model_config['model_path']
        auto_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True)
        if hasattr(auto_config, 'reward_token_id'):
            self.reward_token_id = auto_config.reward_token_id

    # Inference
    def infer_async(self, inputs, attention_mask=None, *args, **infer_kwargs):
        if not isinstance(inputs, torch.Tensor):
            input_ids, attention_mask = encode_inputs(inputs, self.tokenizer)
        else:
            input_ids = inputs

        # Reward model specific
        if self.reward_token_id is not None:
            input_ids, attention_mask = expand_reward_token_id(
                self.reward_token_id, input_ids, attention_mask)

        return self.trainer.infer_async(
            input_ids=input_ids,
            attention_mask=attention_mask,
            *args,
            **infer_kwargs)
