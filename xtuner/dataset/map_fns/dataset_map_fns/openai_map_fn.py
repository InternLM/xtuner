# Copyright (c) OpenMMLab. All rights reserved.
def openai_map_fn(example):
    """
    Example before preprocessing:
        example["messages"] = [
            { "role": "system", "content": "You are an assistant that
                occasionally misspells words." },
            { "role": "user", "content": "Tell me a story." },
            { "role": "assistant", "content": "One day a student
                went to schoool." }
        ]
    Example after preprocessing:
        example["conversation"] = [
            {
                "system": "You are an assistant that occasionally misspells
                    words.",
                "input": "Tell me a story.",
                "output": "One day a student went to schoool."
            }
        ]
    """
    messages = example['messages']
    system = ''
    input = ''
    conversation = []
    while messages and messages[0]['role'] == 'assistant':
        # Skip the first one if it is from assistant
        messages = messages[1:]
    for msg in messages:
        if msg['role'] == 'system':
            system = msg['content']
        elif msg['role'] == 'user':
            input += msg['content']
        elif msg['role'] == 'assistant':
            output_with_loss = msg.get('loss', 'True')
            output_with_loss = str(output_with_loss)
            output_with_loss = output_with_loss.lower() == 'true'
            conversation.append({
                'system': system,
                'input': input,
                'output': msg['content'],
                'output_with_loss': output_with_loss
            })
            system = ''
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}
