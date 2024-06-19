from .base_model_server import BaseModelServer
from .utils import get_critic_model


class CriticModelServer(BaseModelServer):
    # Initialize
    def get_model_class(self, model_path):
        head_name = self.model_config.get('head_name', 'v_head')
        return get_critic_model(model_path, head_name)
