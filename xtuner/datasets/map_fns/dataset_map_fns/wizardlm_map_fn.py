# Copyright (c) OpenMMLab. All rights reserved.
def wizardlm_map_fn(example):
    data = example['conversations']
    while data and data[0]['from'] != 'human':
        # Skip the first one if it is not from human
        data = data[1:]

    conversation = []
    for i in range(0, len(data), 2):
        assert data[i]['from'] == 'human' and data[i + 1]['from'] == 'gpt'
        human, gpt = data[i]['value'], data[i + 1]['value']
        conversation.append({'input': human, 'output': gpt})
    return {'conversation': conversation}
