from xtuner.chat.utils import GenerationConfig


class BaseChat():

    def __init__(self,
                 bot,
                 bot_name=None,
                 chat_template=None,
                 system_template=None) -> None:

        self.chat_template = chat_template
        self.system_template = system_template
        self.bot_name = bot_name
        self.bot = bot

        self.num_turns = 0
        self.history = []

    def reset_history(self):

        self.num_turns = 0
        self.history = []

    def apply_template(self, messages):

        messages = self.chat_template.template_map_fn_v2(messages)

        return ''.join([msg['content'] for msg in messages])

    def update_history(self, role, content):

        self.history.append({'role': role, 'content': content})

    def create_streamer(self, iterable=False):
        return self.bot.create_streamer(self.chat_template, iterable=iterable)

    def chat(self, text, system=None, streamer=None, gen_config=None):

        assert self.chat_template

        if self.num_turns == 0 and system:
            self.update_history('system', system)

        self.update_history('user', text)

        if gen_config is None:
            gen_config = GenerationConfig()

        prompt = self.apply_template(self.history)

        gen_config.stop_words.extend(self.chat_template.stop_words)
        output = self.bot.generate(prompt, streamer, gen_config)
        self.update_history('assistant', output)

        self.num_turns += 1
        return output

    def predict(self, texts, system=None, gen_config=None, repeat=1):

        if gen_config is None:
            gen_config = GenerationConfig()

        gen_config.stop_words.extend(self.chat_template.stop_words)

        prompts = []
        for text in texts:
            msg = []
            if system:
                msg.append({'role': 'system', 'content': system})

            msg.append({'role': 'user', 'content': text})
            prompts.append(self.apply_template(msg))

        outputs = self.bot.predict(prompts, gen_config, repeat)
        return outputs
