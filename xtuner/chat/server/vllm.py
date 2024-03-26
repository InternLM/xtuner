def run_vllm_server(
    model_name_or_path,
    max_batch_size=1,
    max_length=4096,
    use_logn_attn=False,
    rope_scaling_factor=0.0,
    server_name: str = '0.0.0.0',
    server_port: int = 23333,
) -> None:

    import subprocess
    subprocess.run([
        'python', '-m', 'vllm.entrypoints.openai.api_server', '--model',
        f'{model_name_or_path}', '--trust-remote-code'
    ])
