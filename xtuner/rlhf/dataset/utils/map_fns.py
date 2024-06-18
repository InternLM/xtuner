import re


def hhrlhf_map_fn(example):
    string = example['chosen']
    pattern = r'(\n\nHuman|\n\nAssistant)(.+?)(?=(\n\nHuman|\n\nAssistant|$))'  # noqa: E501
    matches = re.findall(pattern, string, re.DOTALL)
    messages = []
    for match in matches:
        role, content = match[0].strip(), match[1].strip()
        if role == 'Human':
            messages.append({'role': 'user', 'content': content[2:]})
        elif role == 'Assistant':
            messages.append({'role': 'assistant', 'content': content[2:]})
        else:
            raise NotImplementedError('role must in Human or Assistant')
    return {'conversation': messages}


def H4_summarize_map_fn(example):
    # prompt = example['prompt']
    chosen = example['chosen']
    # rejected = example['rejected']
    return {'conversation': chosen}
