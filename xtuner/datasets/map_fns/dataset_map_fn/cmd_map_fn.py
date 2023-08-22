# Copyright (c) OpenMMLab. All rights reserved.
def cmd_map_fn(example):
    PROMPT = ('请从一名专业医生的角度，对下述医学问题给出安全、可靠的回答。\n\n'
              '问：{ask}\n\n答：')
    if example.get('ask', '') == '无':
        return {'conversation': [{'input': '', 'output': ''}]}
    else:
        return {
            'conversation': [{
                'input': PROMPT.format(**example),
                'output': example['answer']
            }]
        }
