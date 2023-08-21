# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict

PROMPT_TEMPLATE = ConfigDict(
    title=dict(
        INSTRUCTION_START=(
            'If you are an expert in writing papers, please generate '
            "a good paper title for this paper based on other authors' "
            'descriptions of their abstracts.\n\n'
            '### Descriptions:\n{input}\n\n### Title: '),
        INSTRUCTION='### Descriptions:\n{input}\n\n### Title: '),
    moss_sft=dict(
        INSTRUCTION_START=(
            'You are an AI assistant whose name is {bot_name}.\n'
            'Capabilities and tools that {bot_name} can possess.\n'
            '- Inner thoughts: enabled.\n'
            '- Web search: enabled. API: Search(query)\n'
            '- Calculator: enabled. API: Calculate(expression)\n'
            '- Equation solver: enabled. API: Solve(equation)\n'
            '- Text-to-image: disabled.\n'
            '- Image edition: disabled.\n'
            '- Text-to-speech: disabled.\n'
            '<|Human|>: {input}<eoh>'),
        INSTRUCTION='<|Human|>: {input}<eoh>'),
    alpaca=dict(
        INSTRUCTION_START=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '### Instruction:\n{input}\n\n'
            '### Response: '),
        INSTRUCTION='### Instruction:\n{input}\n\n### Response: '),
    openassistant=dict(
        INSTRUCTION_START='### Human: {input}\n### Assistant: ',
        INSTRUCTION='### Human: {input}\n### Assistant: '),
    medical=dict(
        INSTRUCTION_START=(
            '请从一名专业医生的角度，对下述医学问题给出安全、可靠的回答。\n\n问：{input}\n\n答：'),
        INSTRUCTION='问：{input}\n\n答：'),
    llama_2_chat=dict(
        INSTRUCTION_START=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature. \n<</SYS>>\n\n{input} [/INST]'),
        INSTRUCTION='[INST] {input} [/INST]'),
    internlm_chat=dict(
        INSTRUCTION_START='<|User|>:{input}<eoh>\n<|Bot|>:',
        INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:'),
    code_alpaca=dict(
        INSTRUCTION_START='### Human: {input}\n### Bot: ',
        INSTRUCTION='### Human: {input}\n### Bot: '),
    tiny_codes=dict(
        INSTRUCTION_START='### Human: {input}\n### Bot: ',
        INSTRUCTION='### Human: {input}\n### Bot: '),
    colors=dict(
        INSTRUCTION_START=(
            'You are a professional color designer. Please provide the '
            'corresponding colors based on the description of Human.\n'
            '### Human: {input}\n### Bot: '),
        INSTRUCTION=('### Human: {input}\n### Bot: ')),
    lawyer=dict(
        INSTRUCTION_START=('你现在是一名专业的中国律师，请根据Human的问题给出准确、'
                           '有理有据的回复。\n\n### Human: {input}\n### Bot: '),
        INSTRUCTION=('### Human: {input}\n### Bot: ')),
)
