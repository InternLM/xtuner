# Copyright (c) OpenMMLab. All rights reserved.
def oasst1_map_fn(example):
    r"""Example before preprocessing:
        example['text'] = '### Human: Can you explain xxx'
                          '### Assistant: Sure! xxx'
                          '### Human: I didn't understand how xxx'
                          '### Assistant: It has to do with a process xxx.'

    Example after preprocessing:
        example['conversation'] = [
            {
                'input': 'Can you explain xxx',
                'output': 'Sure! xxx'
            },
            {
                'input': 'I didn't understand how xxx',
                'output': 'It has to do with a process xxx.'
            }
        ]
    """
    data = []
    for sentence in example['text'].strip().split('###'):
        sentence = sentence.strip()
        if sentence[:6] == 'Human:':
            data.append(sentence[6:].strip())
        elif sentence[:10] == 'Assistant:':
            data.append(sentence[10:].strip())
    if len(data) % 2:
        # The last round of conversation solely consists of input
        # without any output.
        # Discard the input part of the last round, as this part is ignored in
        # the loss calculation.
        data.pop()
    conversation = []
    for i in range(0, len(data), 2):
        single_turn_conversation = {'input': data[i], 'output': data[i + 1]}
        conversation.append(single_turn_conversation)
    return {'conversation': conversation}
