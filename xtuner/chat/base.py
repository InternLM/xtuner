from xtuner.chat.utils import GenerationConfig


class BaseChat():

    def __init__(self, bot, bot_name, chat_template, system_template) -> None:

        self.chat_template = chat_template
        self.system_template = system_template
        self.bot_name = bot_name
        self.bot = bot

        self.num_turns = 0
        self.history_text = ''

    def reset_history(self):

        self.num_turns = 0
        self.history_text = ''

    def apply_template(self, text, system=None):

        prompt_text = ''

        if 'SYSTEM' in self.chat_template and self.num_turns == 0:
            system_text = None
            if self.system_template is not None:
                system_text = self.system_template.format(
                    round=self.num_turns + 1, bot_name=self.bot_name)
            elif system is not None:
                system_text = system
            if system_text is not None:
                prompt_text += self.system_template.format(
                    system=system_text,
                    round=self.num_turns + 1,
                    bot_name=self.bot_name)
                prompt_text += self.chat_template['INSTRUCTION'].format(
                    input=text,
                    round=self.num_turns + 1,
                    bot_name=self.bot_name)

        prompt_text += self.chat_template['INSTRUCTION'].format(
            input=text, round=self.num_turns + 1, bot_name=self.bot_name)
        return prompt_text

    def chat(self, text, system=None, gen_config=GenerationConfig()):

        templated_text = self.apply_template(text, system)
        self.history_text += templated_text

        gen_config.stop_words.extend(self.chat_template.stop_words)
        output = self.bot.generate(self.history_text, gen_config)
        self.history_text += output

        self.history_text += self.chat_template.get('SEP', '')
        self.num_turns += 1
        return output

    def completion(self, text, system=None, gen_config=GenerationConfig()):

        self.history_text += text

        gen_config.stop_words.extend(self.chat_template.stop_words)
        output = self.bot.generate(self.history_text, gen_config)
        self.history_text += output

        self.history_text += self.chat_template.get('SEP', '')
        self.num_turns += 1
        return output

    def predict(self, texts, system=None, generation_config=None, repeat=1):

        templated_texts = [self.apply_template(t, system) for t in texts]
        outputs = self.bot.predict(templated_texts, generation_config, repeat)
        return outputs
