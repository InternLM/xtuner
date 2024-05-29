import torch
from transformers import AutoConfig

from ..model_backend.models.critical_and_reward import get_reward_model
from ..tokenizer import tokenizer_utils
from ..utils import expand_reward_token_id
from .base_model_server import BaseModelServer


class RewardModelServer(BaseModelServer):
    # Initialize
    def get_model_class(self, model_path):
        head_name = self.model_config.get('head_name', 'v_head')
        return get_reward_model(model_path, head_name)

    def init_tokenizer_and_config(self, model_config):
        super().init_tokenizer_and_config(self.model_config)

        self.reward_token_id = self.tokenizer.pad_token_id
        model_path = model_config['model_path']
        auto_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True)
        if hasattr(auto_config, 'reward_token_id'):
            self.reward_token_id = auto_config.reward_token_id

    # Inference
    def infer_async(self, inputs, attention_mask=None, *args, **infer_kwargs):
        if not isinstance(inputs, torch.Tensor):
            input_ids, attention_mask = tokenizer_utils.encode(
                inputs, self.tokenizer)
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
