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
                "input": "You are an assistant that occasionally misspells
                    words. Tell me a story.",
                "output": "One day a student went to schoool."
            }
        ]
    """
    messages = example['messages']
    if len(messages) == 0:
        return {'conversation': [{'input': '', 'output': ''}]}
    if messages[0]['role'] == 'system':
        messages[1][
            'content'] = messages[0]['content'] + ' ' + messages[1]['content']
        messages = messages[1:]
    if len(messages) % 2:
        # The last round of conversation solely consists of input
        # without any output.
        # Discard the input part of the last round, as this part is ignored in
        # the loss calculation.
        messages.pop()
    conversation = []
    for i in range(0, len(messages), 2):
        single_turn_conversation = {
            'input': messages[i]['content'],
            'output': messages[i + 1]['content']
        }
        conversation.append(single_turn_conversation)
    return {'conversation': conversation}
