import re


def default_map_fn(example):
    return example


def hhrlhf_map_fn(example):
    string = example['chosen']
    pattern = r'(\n\nHuman|\n\nAssistant)(.+?)(?=(\n\nHuman|\n\nAssistant|$))'
    matches = re.findall(pattern, string, re.DOTALL)
    messages = []
    for match in matches:
        role, content = match[0].strip(), match[1].strip()
        if role == 'Human':
            messages.append(dict(role='user', content=content[2:]))
        elif role == 'Assistant':
            messages.append(dict(role='assistant', content=content[2:]))
        else:
            raise NotImplementedError('role must in Human or Assistant')
    return {'conversation': messages}


def H4_hhh_alignment_map_fn(example):
    input = example['input']
    choices = example['targets']['choices']
    labels = example['targets']['labels']
    for label, choice in zip(labels, choices):
        if label == 1:
            chosen = choice
    messages = [
        dict(role='user', content=input),
        dict(role='assistant', content=chosen)
    ]
    return {'conversation': messages}


def H4_summarize_map_fn(example):
    # prompt = example['prompt']
    chosen = example['chosen']
    # rejected = example['rejected']
    return {'conversation': chosen}


def stingning_ultrachat_map_fn(example):
    # id = example['id']
    data = example['data']
    messages = []
    for i, d in enumerate(data):
        if i % 2 == 0:
            role = 'user'
        else:
            role = 'assistant'
        messages.append(dict(role=role, content=d))

    return {'conversation': messages}


def nvidia_HelpSteer_map_fn(example):
    prompt = example['prompt']
    response = example['response']
    messages = [
        dict(role='user', content=prompt),
        dict(role='assistant', content=response)
    ]

    return {'conversation': messages}


def nvidia_OpenMathInstruct_map_fn(example):
    question = example['question']
    # expected_answer = example['expected_answer']
    generated_solution = example['generated_solution']
    messages = [
        dict(role='user', content=question),
        dict(role='assistant', content=generated_solution)
    ]

    return {'conversation': messages}


def nvidia_sft_datablend_v1_map_fn(example):
    conversations = example['conversations']
    # system = example['system']
    messages = []
    for conv in conversations:
        if conv['from'] == 'User':
            role = 'user'
        elif conv['from'] == 'Assistant':
            role = 'assistant'
        messages.append(dict(role=role, content=conv['value']))

    return {'conversation': messages}


def argilla_prompt_map_fn(example):
    prompt = example['prompt']
    messages = [dict(role='user', content=prompt)]
    return {'conversation': messages}


def dibt_prompt_map_fn(example):
    prompt = example['prompt']
    messages = [dict(role='user', content=prompt)]
    return {'conversation': messages}


def FW_fineweb_edu_map_fn(example):
    question = ''
    answer = example['text']
    token_count = example['token_count']
    messages = [
        dict(role='user', content=question),
        dict(role='assistant', content=answer)
    ]

    return {'conversation': messages, 'token_count': token_count}
