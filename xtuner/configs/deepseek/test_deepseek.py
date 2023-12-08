import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pdb

# model_name = "deepseek-ai/deepseek-llm-67b-chat"
model_name = "/workspace/models/deepseek-coder-33b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


def chat(msg: str):
    messages = [
        {"role": "user", "content": msg}
    ]
    pdb.set_trace()
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    print(input_tensor.shape)
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1024, top_k=1)
    
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(result)

chat(msg='什么是 mmdeploy ?')
chat(msg='判断以下句子是否是个有主题的疑问句，结果用 1～10 表示。直接提供得分不要解释。问题“如何安装 mmdeploy ?”')
