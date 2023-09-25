# Copyright (c) OpenMMLab. All rights reserved.
def msagent_map_fn(example):
    text = eval(example['conversations'])
    conversation = []
    input_text = ''
    for t in text:
        if t['from'] == 'system':
            input_text += f"<|System|>:{t['value']}\n\n"
        elif t['from'] == 'user':
            input_text += f"<|User|>:{t['value']}\n\n"
        elif t['from'] == 'assistant':
            input_text += '<|Bot|>:'
            conversation.append({'input': input_text, 'output': t['value']})
            input_text = ''
    return {'conversation': conversation}
