# Copyright (c) OpenMMLab. All rights reserved.
import json
import re

think_regex = r'(.*?)(<\|startofthink\|\>)(.*?)(<\|endofthink\|\>)'
exec_regex = r'(<\|startofexec\|\>)(.*?)(<\|endofexec\|\>)(.*?)$'


def replace_think(match):
    out_text = ''
    if match.group(1).strip() != '':
        out_text += f'Thought:{match.group(1).strip()}\n'
    think_text = match.group(3).replace('```JSON',
                                        '').replace('```',
                                                    '').replace('\n', '')
    think_json = json.loads(think_text)
    out_text += (f"Action:{think_json['api_name']}\n"
                 f"Action Input:{think_json['parameters']}\n")
    return out_text


def replace_exec(match):
    out_text = ''
    exec_text = match.group(2).replace('```JSON',
                                       '').replace('```',
                                                   '').replace('\n', '')
    exec_json = json.loads(exec_text)
    out_text += f'Response:{exec_json}\n'
    if match.group(4).strip() != '':
        out_text += f'Final Answer:{match.group(4).strip()}\n'
    return out_text


def extract_json_objects(text, decoder=json.JSONDecoder()):
    pos = 0
    results = []
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            if 'name' in result and 'description' in result:
                results.append(result)
                pos = match + index
            else:
                pos = match + 1
        except ValueError:
            pos = match + 1
    return results


def msagent_react_map_fn(example):
    text = example['conversations']
    if isinstance(text, str):
        text = eval(text)
    if len(text) < 2:  # Filter out invalid data
        return {'conversation': []}
    conversation = []
    system_text = ''
    input_text = ''
    for t in text:
        if t['from'] == 'system':
            system_text += '你是一个可以调用外部工具的助手，可以使用的工具包括：\n'
            json_objects = extract_json_objects(t['value'])
            api_dict = {}
            for obj in json_objects:
                api_dict[obj['name']] = obj['description']
                try:
                    params = {
                        i['name']: i['description']
                        for i in obj['paths'][0]['parameters']
                    }
                    api_dict[obj['name']] += f'\n输入参数: {params}'
                except Exception:
                    pass
            system_text += f'{api_dict}\n'
            system_text += (
                '如果使用工具请遵循以下格式回复：\n```\n'
                'Thought:思考你当前步骤需要解决什么问题，是否需要使用工具\n'
                f'Action:工具名称，你的工具必须从 [{str(list(api_dict.keys()))}] 选择\n'
                'Action Input:工具输入参数\n```\n工具返回按照以下格式回复：\n```\n'
                'Response:调用工具后的结果\n```\n如果你已经知道了答案，或者你不需要工具，'
                '请遵循以下格式回复\n```\n'
                'Thought:给出最终答案的思考过程\n'
                'Final Answer:最终答案\n```\n开始!\n')
        elif t['from'] == 'user':
            input_text += f"{t['value']}\n"
        elif t['from'] == 'assistant':
            output = t['value']
            output_response = None
            try:
                if '<|startofexec|>' in output:
                    output, output_response = output.split('<|startofexec|>')
                    output_response = '<|startofexec|>' + output_response
                output, think_cnt = re.subn(
                    think_regex, replace_think, output, flags=re.DOTALL)
            except Exception:
                return {'conversation': []}

            if think_cnt == 0:
                output = f'Final Answer:{output}\n'
            else:
                output = f'{output}\n'
            conversation.append({
                'system': system_text,
                'input': input_text,
                'output': output
            })
            system_text = ''
            input_text = ''
            if output_response is not None:
                try:
                    output_response, exec_cnt = re.subn(
                        exec_regex,
                        replace_exec,
                        output_response,
                        flags=re.DOTALL)
                    if 'Final Answer:' in output_response:
                        output_response, output_answer = output_response.split(
                            'Final Answer:')
                        output_answer = 'Final Answer:' + output_answer
                        conversation.append({
                            'system': output_response,
                            'output': output_answer
                        })
                except Exception:
                    pass
    return {'conversation': conversation}
