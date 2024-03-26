from transformers import AutoConfig

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


def run_lmdeploy_server(
    model_name_or_path,
    max_batch_size=1,
    max_length=4096,
    use_logn_attn=False,
    rope_scaling_factor=0.0,
    server_name: str = '0.0.0.0',
    server_port: int = 23333,
) -> None:

    from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
    from lmdeploy.serve.openai.api_server import serve

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
        backend = 'turbomind'
    elif hf_cls in PYTORCH_SUPPORTED:
        backend_config = PytorchEngineConfig(
            model_name='base',
            session_len=max_length,
            max_batch_size=max_batch_size,
            rope_scaling_factor=rope_scaling_factor,
            use_logn_attn=use_logn_attn)
        backend = 'pytorch'
    else:
        raise NotImplementedError

    serve(model_name_or_path, 'base', backend, backend_config)
