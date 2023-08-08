# Copyright (c) OpenMMLab. All rights reserved.
def internlm_map_fn(example):
    user = '<|User|>'
    eoh = '<eoh>'
    eoa = '<eoa>'  # noqa:F841
    assistant = '<|Bot|>'
    instruction = example.get('input', '')
    prompt = f'<BOS>{user}:{instruction}{eoh}\n{assistant}:'
    return {'input': prompt}
