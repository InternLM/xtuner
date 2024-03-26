from mmengine import ConfigDict

from .chat import ChatTemplateConfig

CHAT_TEMPLATE = ConfigDict(
    default=ChatTemplateConfig(
        system='<|System|>:{system}\n',
        instruction='<|User|>:{input}\n<|Bot|>:',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    pretrain=ChatTemplateConfig(
        system='{system}',
        instruction='{input}',
        suffix='',
        suffix_as_eos=False,
        sep='',
        stop_words=[]),
    zephyr=ChatTemplateConfig(
        system='<|system|>\n{system}\n',
        instruction='<|user|>\n{input}\n<|assistant|>\n',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    internlm_chat=ChatTemplateConfig(
        system='<|System|>:{system}\n',
        instruction='<|User|>:{input}<eoh>\n<|Bot|>:',
        suffix='<eoa>',
        suffix_as_eos=True,
        sep='\n',
        stop_words=['<eoa>']),
    internlm2_chat=ChatTemplateConfig(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        instruction=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        suffix='<|im_end|>',
        suffix_as_eos=True,
        sep='\n',
        stop_words=['<|im_end|>']),
    moss_sft=ChatTemplateConfig(
        system='{system}\n',
        instruction='<|Human|>: {input}<eoh>\n',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=['<eoc>', '<eom>']),
    llama2_chat=ChatTemplateConfig(
        system=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        instruction='[INST] {input} [/INST]',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    code_llama_chat=ChatTemplateConfig(
        system='{system}\n',
        instruction='[INST] {input} [/INST]',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    chatglm2=ChatTemplateConfig(
        system='{system}\n',
        instruction='[Round {round}]\n\n问：{input}\n\n答：',
        suffix='',
        suffix_as_eos=False,
        sep='\n\n',
        stop_words=[]),
    chatglm3=ChatTemplateConfig(
        system='<|system|>\n{system}',
        instruction='<|user|>\n{input}<|assistant|>\n',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    qwen_chat=ChatTemplateConfig(
        system=('\n<|im_start|>system\n{system}<|im_end|>'),
        instruction=(
            '\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'),
        suffix='<|im_end|>',
        suffix_as_eos=True,
        sep='\n',
        stop_words=['<|im_end|>', '<|endoftext|>']),
    baichuan_chat=ChatTemplateConfig(
        system='{system}\n',
        instruction='<reserved_102>{input}<reserved_103>',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    baichuan2_chat=ChatTemplateConfig(
        system='{system}\n',
        instruction='<reserved_106>{input}<reserved_107>',
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    wizardlm=ChatTemplateConfig(
        system=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        instruction=('USER: {input} ASSISTANT:'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    wizardcoder=ChatTemplateConfig(
        system=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '{system}\n '),
        instruction=('### Instruction:\n{input}\n\n### Response:'),
        suffix='',
        suffix_as_eos=False,
        sep='\n\n',
        stop_words=[]),
    vicuna=ChatTemplateConfig(
        system=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        instruction=('USER: {input} ASSISTANT:'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    deepseek_coder=ChatTemplateConfig(
        system=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        instruction=('### Instruction:\n{input}\n### Response:\n'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    # TODO: deprecation, v0.2.0
    deepseekcoder=ChatTemplateConfig(
        system=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        instruction=('### Instruction:\n{input}\n### Response:\n'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    deepseek_moe=ChatTemplateConfig(
        system=('[INST] {system} [/INST]\n'),
        instruction=('[INST] {input} [/INST]'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    mistral=ChatTemplateConfig(
        system=('[INST] {system} [/INST]\n'),
        instruction=('[INST] {input} [/INST]'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
    mixtral=ChatTemplateConfig(
        system=('[INST] {system} [/INST]\n'),
        instruction=('[INST] {input} [/INST]'),
        suffix='',
        suffix_as_eos=False,
        sep='\n',
        stop_words=[]),
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
