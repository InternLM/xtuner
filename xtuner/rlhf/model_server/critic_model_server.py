from ..model_backend.models.critical_and_reward import get_critic_model
from .base_model_server import BaseModelServer


class CriticModelServer(BaseModelServer):
    # Initialize
    def get_model_class(self, model_path):
        head_name = self.model_config.get('head_name', 'v_head')
        return get_critic_model(model_path, head_name)
