import re

from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
from .base import BaseChat
from .moss_plugins import plugins_api


class MossChat(BaseChat):

    def __init__(self,
                 bot,
                 bot_name,
                 chat_template,
                 system_template,
                 plugins=['calculate', 'solve', 'search']) -> None:
        super().__init__(bot, bot_name, chat_template, system_template)

        assert self.chat_template == PROMPT_TEMPLATE['moss_sft']
        assert self.system_template == SYSTEM_TEMPLATE['moss_sft']

        self.plugins = plugins

        self.calculate_open = 'calculate' in plugins
        self.solve_open = 'solve' in plugins
        self.search_open = 'search' in plugins

    def apply_template(self, text, system=None):
        templated_text = super().apply_template(text, system)

        if not self.calculate_open:
            templated_text.replace(('- Calculator: enabled. API: '
                                    'Calculate(expression)'),
                                   '- Calculator: disabled.')
        if not self.solve_open:
            templated_text.replace(
                '- Equation solver: enabled. API: Solve(equation)',
                '- Equation solver: disabled.')
        if not self.search_open:
            templated_text.replace('- Web search: enabled. API: Search(query)',
                                   '- Web search: disabled.')

    def chat(self, text, system=None, generation_config=None):

        templated_text = self.apply_template(text, system)
        self.history_text += templated_text

        response = self.bot.generate(self.history_text, generation_config)
        self.history_text += response

        pattern = r'<\|Commands\|>:(.*?)<eoc>'
        command_text = ', '.join(re.findall(pattern, response))

        extent_text = plugins_api(
            command_text,
            calculate_open=self.calculate_open,
            solve_open=self.solve_open,
            search_open=self.search_open)
        self.history_text += extent_text

        output = self.bot.generate(self.history_text, generation_config)
        self.history_text += output

        self.history_text += self.chat_template.get('SEP', '')
        self.num_turns += 1

        return output
