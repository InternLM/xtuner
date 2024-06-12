from .base import BaseConverter


class AlpacaConverter(BaseConverter):

    @classmethod
    def source_format(cls):
        data = {
            'instruction': 'INSTRUCTION',
            'input': 'INPUT',
            'output': 'OUTPUT',
        }
        return data

    @classmethod
    def target_format(cls):
        data = {
            'messages': [
                {
                    'role': 'user',
                    'content': 'INSTRUCTION\nINPUT'
                },
                {
                    'role': 'assistant',
                    'content': 'OUTPUT'
                },
            ]
        }
        return data

    @staticmethod
    def convert(data):
        if data.get('output') == '<nooutput>':
            return {'messages': []}
        else:
            return {
                'messages': [
                    {
                        'role': 'user',
                        'content': f"{data['instruction']}\n{data['input']}"
                    },
                    {
                        'role': 'assistant',
                        'content': f"{data['output']}"
                    },
                ]
            }
