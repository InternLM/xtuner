import json

from xtuner.types import HybridChatTemplate, TrainingHybridChatMessages


chat_template = HybridChatTemplate(
    system='<|im_start|>system\n{system}<|im_end|>\n',
    user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
    assistant='{assistant}<|im_end|>\n',
    stop_words=['<|im_end|>'],
    image_token='<image>',
    files='<|im_start|>user name=file\n{files}<|im_end|>\n', 
    function_call='{assistant}<|action_start|><|plugin|>\n{function_call}<|action_end|><|im_end|>\n',  # noqa: E501, E251
    function_result='<|im_start|>environment name=<|plugin|>\n{function_result}<|im_end|>\n<|im_start|>assistant\n',  # noqa: E501, E251
    functions='<|im_start|>system name=<|plugin|>\n{functions}<|im_end|>\n',
    code_interpreter_call='{assistant}<|action_start|><|interpreter|>\n{code_interpreter_call}<|action_end|><|im_end|>\n',  # noqa: E501, E251
    code_interpreter_result='<|im_start|>environment name=<|interpreter|>\n{code_interpreter_result}<|im_end|>\n<|im_start|>assistant\n',  # noqa: E501, E251
    code_interpreter='<|im_start|>system name=<|interpreter|>\n{code_interpreter}<|im_end|>\n'

)

agent_data = json.load(open('agent.json'))

msg = TrainingHybridChatMessages.from_dict(agent_data)
print(msg.apply_chat_template(chat_template))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)
print(msg.tokenize(tokenizer, chat_template))