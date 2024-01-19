from xtuner.chat.utils import GenerationConfig
from .base import BaseBot


class OpenaiBot(BaseBot):

    def __init__(
        self,
        api_url,
        openai_api_key='EMPTY',
    ) -> None:

        super().__init__()
        from openai import OpenAI

        self.client = OpenAI(base_url=api_url, api_key=openai_api_key)

    @property
    def default_gen_config(self):
        return {}

    def generate(self, text, gen_config: GenerationConfig = None):

        openai_gen_config = dict(
            max_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )

        output = self.client.completions.create(text, **openai_gen_config)
        return output

    def predict(self, texts, gen_config: GenerationConfig = None, repeat=1):

        openai_gen_config = dict(
            n=repeat,
            max_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )
        outputs = self.client.completions.create(texts, **openai_gen_config)

        return outputs
