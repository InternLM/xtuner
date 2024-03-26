from xtuner.chat.utils import GenerationConfig
from .base import BaseBot


class VllmBot(BaseBot):
    # TODO support tp
    def __init__(
        self,
        model_name_or_path,
        max_length=4096,
        max_batch_size=1,
    ) -> None:
        super().__init__()
        from vllm import LLM
        self.pipeline = LLM(model_name_or_path, trust_remote_code=True)

    def generate(self, text, gen_config=None):
        from vllm import SamplingParams
        vllm_gen_config = SamplingParams(
            max_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )

        output = self.pipeline.generate([text], **vllm_gen_config)
        return output[0].outputs[0].text

    def predict(self, texts, gen_config: GenerationConfig = None, repeat=1):

        from vllm import SamplingParams
        vllm_gen_config = SamplingParams(
            n=repeat,
            max_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )

        outputs = self.pipeline.generate(texts, **vllm_gen_config)

        return [o.outputs[0].text for o in outputs]
