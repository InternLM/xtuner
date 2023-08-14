# Copyright (c) OpenMMLab. All rights reserved.
def internlm_map_fn(example):
    user = '<|User|>'
    eoh = '<eoh>'
    eoa = '<eoa>'  # noqa:F841
    assistant = '<|Bot|>'
    instructions = example.get('input', [''])
    if isinstance(instructions, str):
        instructions = [instructions]
    prompts = [
        f'<BOS>{user}:{instruction}{eoh}\n{assistant}:'
        for instruction in instructions
    ]
    return {'input': prompts}
