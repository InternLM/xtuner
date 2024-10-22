# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict

# - Turn 0: SYSTEM + INSTRUCTION, [output + SUFFIX], SEP
# - Turn 1: INSTRUCTION, [output + SUFFIX], SEP
# - Turn ...
# Note: [] means having supervised loss during the fine-tuning
PROMPT_TEMPLATE = ConfigDict(
    default=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}\n<|Bot|>:',
        SEP='\n'),
    zephyr=dict(
        SYSTEM='<|system|>\n{system}\n',
        INSTRUCTION='<|user|>\n{input}\n<|assistant|>\n',
        SEP='\n'),
    internlm_chat=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
        SUFFIX='<eoa>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<eoa>']),
    internlm2_chat=dict(
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>']),
    moss_sft=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<|Human|>: {input}<eoh>\n',
        SEP='\n',
        STOP_WORDS=['<eoc>', '<eom>']),
    llama2_chat=dict(
        SYSTEM=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        INSTRUCTION='[INST] {input} [/INST]',
        SEP='\n'),
    code_llama_chat=dict(
        SYSTEM='{system}\n', INSTRUCTION='[INST] {input} [/INST]'),
    chatglm2=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='[Round {round}]\n\n问：{input}\n\n答：',
        SEP='\n\n'),
    chatglm3=dict(
        SYSTEM='<|system|>\n{system}',
        INSTRUCTION='<|user|>\n{input}<|assistant|>\n',
        SEP='\n'),
    qwen_chat=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    baichuan_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_102>{input}<reserved_103>',
        SEP='\n'),
    baichuan2_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_106>{input}<reserved_107>',
        SEP='\n'),
    wizardlm=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    wizardcoder=dict(
        SYSTEM=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '{system}\n '),
        INSTRUCTION=('### Instruction:\n{input}\n\n### Response:'),
        SEP='\n\n'),
    vicuna=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    deepseek_coder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    # TODO: deprecation, v0.2.0
    deepseekcoder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    deepseek_moe=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    deepseek_v2=dict(
        SYSTEM='{system}\n\n',
        INSTRUCTION='User: {input}\n\nAssistant: ',
        SUFFIX='<｜end▁of▁sentence｜>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<｜end▁of▁sentence｜>']),
    mistral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mixtral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    minicpm=dict(INSTRUCTION=('<用户> {input} <AI>'), SEP='\n'),
    minicpm3=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    gemma=dict(
        # `system` field is extended by xtuner
        SYSTEM=('<start_of_turn>system\n{system}<end_of_turn>\n'),
        INSTRUCTION=('<start_of_turn>user\n{input}<end_of_turn>\n'
                     '<start_of_turn>model\n'),
        SUFFIX='<end_of_turn>',
        SUFFIX_AS_EOS=False,
        SEP='\n',
        STOP_WORDS=['<end_of_turn>']),
    cohere_chat=dict(
        SYSTEM=('<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}'
                '<|END_OF_TURN_TOKEN|>'),
        INSTRUCTION=(
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{input}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'),
        SUFFIX='<|END_OF_TURN_TOKEN|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|END_OF_TURN_TOKEN|>']),
    llama3_chat=dict(
        SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
                '{system}<|eot_id|>'),
        INSTRUCTION=(
            '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'),
        SUFFIX='<|eot_id|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|eot_id|>']),
    phi3_chat=dict(
        SYSTEM='<|system|>\n{system}<|end|>\n',
        INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
        SUFFIX='<|end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|end|>']),
)

SYSTEM_TEMPLATE = ConfigDict(
    moss_sft=('You are an AI assistant whose name is {bot_name}.\n'
              'Capabilities and tools that {bot_name} can possess.\n'
              '- Inner thoughts: enabled.\n'
              '- Web search: enabled. API: Search(query)\n'
              '- Calculator: enabled. API: Calculate(expression)\n'
              '- Equation solver: enabled. API: Solve(equation)\n'
              '- Text-to-image: disabled.\n'
              '- Image edition: disabled.\n'
              '- Text-to-speech: disabled.\n'),
    alpaca=('Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n'),
    arxiv_gentile=('If you are an expert in writing papers, please generate '
                   "a good paper title for this paper based on other authors' "
                   'descriptions of their abstracts.\n'),
    colorist=('You are a professional color designer. Please provide the '
              'corresponding colors based on the description of Human.\n'),
    coder=('You are a professional programer. Please provide the '
           'corresponding code based on the description of Human.\n'),
    lawyer='你现在是一名专业的中国律师，请根据用户的问题给出准确、有理有据的回复。\n',
    medical='如果你是一名医生，请根据患者的描述回答医学问题。\n',
    sql=('If you are an expert in SQL, please generate a good SQL Query '
         'for Question based on the CREATE TABLE statement.\n'),
)
