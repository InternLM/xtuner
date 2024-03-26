from transformers import AutoConfig

from xtuner.chat.streamer import (LMDeployTextIteratorStreamer,
                                  LMDeployTextStreamer)
from xtuner.chat.utils import GenerationConfig
from .base import BaseBot

TURBOMIND_SUPPORTED = [
    'InternLMForCausalLM',
    'QWenLMHeadModel',
    'BaiChuanForCausalLM',  # Baichuan 7B
    'BaichuanForCausalLM',  # Baichuan2 7B
    'LlamaForCausalLM',
]

PYTORCH_SUPPORTED = [
    'InternLMForCausalLM',
    'QWenLMHeadModel',
    'BaiChuanForCausalLM',  # Baichuan 7B
    'BaichuanForCausalLM',  # Baichuan2 7B
    'LlamaForCausalLM',
]


class LMDeployBot(BaseBot):
    # TODO support tp
    def __init__(
        self,
        model_name_or_path,
        max_batch_size=1,
        max_length=4096,
        use_logn_attn=False,
        rope_scaling_factor=0.0,
    ) -> None:
        super().__init__()

        from lmdeploy import pipeline
        from lmdeploy.messages import (PytorchEngineConfig,
                                       TurbomindEngineConfig)

        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True)
        hf_cls = hf_config.architectures[0]
        if hf_cls in TURBOMIND_SUPPORTED:

            backend_config = TurbomindEngineConfig(
                model_name='base',
                session_len=max_length,
                max_batch_size=max_batch_size,
                rope_scaling_factor=rope_scaling_factor,
                use_logn_attn=use_logn_attn)
        elif hf_cls in TURBOMIND_SUPPORTED:
            backend_config = PytorchEngineConfig(
                model_name='base',
                session_len=max_length,
                max_batch_size=max_batch_size,
                rope_scaling_factor=rope_scaling_factor,
                use_logn_attn=use_logn_attn)
        else:
            raise NotImplementedError

        self.pipeline = pipeline(
            model_name_or_path, backend_config=backend_config)

        self.session_id = 0

    def create_streamer(self, iterable=False):

        if iterable:
            return LMDeployTextIteratorStreamer()
        else:
            return LMDeployTextStreamer()

    def generate(self,
                 text,
                 streamer=None,
                 gen_config: GenerationConfig = None):

        from lmdeploy.messages import GenerationConfig as LMGenerationConfig
        lm_gen_config = LMGenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            random_seed=gen_config.seed,
        )

        self.session_id += 1

        generator = self.pipeline.generate(
            text, self.session_id, gen_config=lm_gen_config)

        results = []

        async def _streaming_generate():
            async for output in generator:
                results.append(output.response)
                if streamer:
                    streamer.put(output.response)

            if streamer:
                streamer.end()

        import asyncio
        asyncio.run(_streaming_generate())

        return ''.join(results)

    def predict(self, texts, gen_config: GenerationConfig = None, repeat=1):

        from lmdeploy.messages import GenerationConfig as LMGenerationConfig
        lm_gen_config = LMGenerationConfig(
            n=repeat,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            random_seed=gen_config.seed,
        )

        outputs = self.pipeline(texts, gen_config=lm_gen_config)

        return [o.text for o in outputs]


class SLoraBot(BaseBot):
    # TODO support tp
    def __init__(
        self,
        model_name_or_path,
        adapters,
        max_batch_size=1,
        max_length=4096,
        use_logn_attn=False,
        rope_scaling_factor=0.0,
    ) -> None:
        super().__init__()

        from lmdeploy import pipeline
        from lmdeploy.messages import PytorchEngineConfig

        backend_config = PytorchEngineConfig(
            model_name='base',
            session_len=max_length,
            max_batch_size=max_batch_size,
            rope_scaling_factor=rope_scaling_factor,
            use_logn_attn=use_logn_attn)

        self.pipeline = pipeline(
            model_name_or_path, backend_config=backend_config)

    def generate(self, text, gen_config=None):

        from lmdeploy.messages import GenerationConfig as LMGenerationConfig
        lm_gen_config = LMGenerationConfig(
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )

        output = self.pipeline([text], gen_config=lm_gen_config)
        return output[0]

    def predict(self, texts, gen_config=None, repeat=1):

        from lmdeploy.messages import GenerationConfig as LMGenerationConfig
        lm_gen_config = LMGenerationConfig(
            n=repeat,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
            seed=gen_config.seed,
        )

        outputs = self.pipeline(texts, gen_config=lm_gen_config)

        return [o.text for o in outputs]
