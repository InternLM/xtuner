# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

from mmengine.config import ConfigDict
from transformers import AutoTokenizer

from xtuner.utils import PromptTemplateConfig


@dataclass
class AgentPromptTemplateConfig(PromptTemplateConfig):
    system_with_meta: str
    environment: str
    environment_with_meta: str
    instruction_with_file: str
    interpreter_token: str
    plugin_token: str
    action_start_token: str
    action_end_token: str

    def template_map_fn_v2(self, example):
        messages_original = example['messages']
        messages = []
        n_turn = 1
        for data_idx, data in enumerate(messages_original):
            role_original = data.get('role')
            content_original = data.get('content')
            extra = data.get('extra')
            if role_original == 'system':
                role = 'system'
                if extra is None:
                    content = self.system.format(system=content_original)
                else:
                    meta_type = extra.get('meta_type')
                    if meta_type == 'interpreter':
                        content = self.system_with_meta.format(
                            system=content_original,
                            meta=self.interpreter_token)
                    elif meta_type == 'plugin':
                        content = self.system_with_meta.format(
                            system=content_original, meta=self.plugin_token)
                    else:
                        raise NotImplementedError
            elif role_original == 'environment':
                role = 'system'
                if extra is None:
                    content = self.environment.format(
                        environment=content_original)
                else:
                    meta_type = extra.get('meta_type')
                    if meta_type == 'interpreter':
                        content = self.environment_with_meta.format(
                            environment=content_original,
                            meta=self.interpreter_token)
                    elif meta_type == 'plugin':
                        content = self.environment_with_meta.format(
                            environment=content_original,
                            meta=self.plugin_token)
                    else:
                        raise NotImplementedError
            elif role_original == 'user':
                role = 'user'
                if extra is None:
                    content = self.instruction.format(
                        input=content_original, round=n_turn)
                else:
                    upload_type = extra.get('upload_type')
                    upload_content = extra.get('upload_content')
                    assert upload_content is not None
                    if upload_type == 'file':
                        content = self.instruction_with_file.format(
                            input=content_original,
                            file=upload_content,
                            round=n_turn)
                    else:
                        raise NotImplementedError
                n_turn += 1
            elif role_original == 'assistant':
                role = 'assistant'
                if extra is None:
                    content = content_original
                else:
                    action_type = extra.get('action_type')
                    action_content = extra.get('action_content')
                    assert action_content is not None
                    content = content_original + '\n' + self.action_start_token
                    if action_type == 'interpreter':
                        content += self.interpreter_token + '\n'
                    elif action_type == 'plugin':
                        content += self.plugin_token + '\n'
                    else:
                        raise NotImplementedError
                    content += action_content + self.action_end_token + '\n'
                if self.suffix != '':
                    content += self.suffix
            else:
                raise NotImplementedError
            messages.append({'role': role, 'content': content})
        return {'messages': messages}


AGENT_PROMPT_TEMPLATE = ConfigDict(
    internlm2_chat=AgentPromptTemplateConfig(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        system_with_meta='<|im_start|>system name={meta}\n{system}<|im_end|>\n',
        environment='<|im_start|>environment\n{environment}<|im_end|>\n<|im_start|>assistant\n',
        environment_with_meta='<|im_start|>environment name={meta}\n{environment}<|im_end|>\n<|im_start|>assistant\n',
        instruction='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
        instruction_with_file='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>user name=file\n{file}<|im_end|>\n<|im_start|>assistant\n',
        interpreter_token='<|interpreter|>',
        plugin_token='<|plugin|>',
        action_start_token='<|action_start|>',
        action_end_token='<|action_end|>',
        suffix='<|im_end|>',
        suffix_as_eos=True,
        sep='\n',
        stop_words=['<|im_end|>', '<|action_end|>']), )

data = [{
    "messages": [
        {
            "role": "system",
            "content": "你是书生浦语2，一个无害的人工智能助手"
        },
        {
            "role": "system",
            "content": "YOUR_INTERPRETER, YOUR_INTERPRETER, YOUR_INTERPRETER, YOUR_INTERPRETER",
            "extra": {
                "meta_type": "interpreter"
            }
        },
        {
            "role": "system",
            "content": "YOUR_PLUGIN, YOUR_PLUGIN, YOUR_PLUGIN, YOUR_PLUGIN",
            "extra": {
                "meta_type": "plugin"
            }
        },
        {
            "role": "user",
            "content": "请帮我对该数据集进行数据处理并可视化。",
            "extra": {
                "upload_type": "file",
                "upload_content": "[{\"path\": \"data.csv\", size=\"10K\"}]"
            }
        },
        {
            "role": "assistant",
            "content": "我已经帮您处理了数据并进行了可视化。",
            "extra": {
                "action_type": "interpreter",
                "action_content": "YOUR_CODE, YOUR_CODE, YOUR_CODE, YOUR_CODE"
            }
        },
        {
            "role": "environment",
            "content": "![image](xxx.png)",
            "extra": {
                "meta_type": "interpreter"
            }
        },
        {
            "role": "assistant",
            "content": "由于代码输出没有包含在这里，我无法提供具体的图表内容。但是，如果代码执行无误，它会显示一个极坐标图，其中包含两个填充区域，一个代表上午9点的风向与降雨量的关系，另一个代表下午3点的风向与降雨量的关系。每个方向上的值是根据提供的数据集计算得到的平均降雨量。图表应该有一个清晰的标题、图例，并且能够直观地比较不同风向下的降雨量。由于使用了暗色主题，图表的整体外观应该是深色背景上的亮色线条和填充区域。"
        },
        {
            "role": "user",
            "content": "你能解释一下代码吗"
        },
        {
            "role": "assistant",
            "content": "我使用的是Python的Plotly库来创建一个极坐标图，以展示风向与降雨量之间的关系。代码中首先创建了一个Plotly的Figure对象。接着，我添加了两个`Scatterpolar`追踪。第一个追踪展示的是上午9点的风向与降雨量的平均值的关系，第二个追踪展示的是下午3点的风向与降雨量的平均值的关系。在这两个追踪中，`r`参数代表降雨量的平均值，`theta`参数代表风向。`fill=\"toself\"`表示图形以自身为边界进行填充。然后我更新了图表的布局，隐藏了径向轴，添加了图例，设置了标题和标题字体，选择了暗色主题，并设定了图表的背景色、宽度和高度。最后，使用`fig.show()`展示了图表。"
        },
        {
            "role": "user",
            "content": "我想了解今天上海的天气"
        },
        {
            "role": "assistant",
            "content": "好的，我将为你查询上海的天气。",
            "extra": {
                "action_type": "plugin",
                "action_content": "{\"name\": \"get_current_weather\", \"parameters\": {\"location\": \"Shanghai\"}}"
            }
        },
        {
            "role": "environment",
            "content": "{\"temperature\": 22}",
            "extra": {
                "meta_type": "plugin"
            }
        },
        {
            "role": "assistant",
            "content": "上海的天气是 22 摄氏度"
        }]
}]


tokenizer = AutoTokenizer.from_pretrained('internlm/internlm2-chat-7b', trust_remote_code=True)

template = AGENT_PROMPT_TEMPLATE.internlm2_chat
messages = template.template_map_fn_v2(data[0])
ret = template.encode_map_fn(messages, tokenizer=tokenizer, max_length=2048)
print('==================================================')
print('input_ids')
print('==================================================')
print(tokenizer.decode(ret['input_ids']))
print()

print('==================================================')
print('labels')
print('==================================================')
labels = [i if i > 0 else 0 for i in ret['labels']]
print(tokenizer.decode(labels))
breakpoint()
