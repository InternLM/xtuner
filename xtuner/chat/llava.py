from xtuner.chat.utils import GenerationConfig
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from .base import BaseChat


class LlavaChat(BaseChat):

    def __init__(self,
                 bot,
                 image,
                 bot_name=None,
                 chat_template=None,
                 system_template=None) -> None:
        super().__init__(bot, bot_name, chat_template, system_template)

        self.pixel_values = self.bot.process_img(image)
        self.history = []

    def update_history(self, role, content):

        self.history.append({'role': role, 'content': content})

    def create_streamer(self, chat_template=None, iterable=False):
        return self.bot.create_streamer(self.chat_template, iterable=iterable)

    def reset_history(self):

        self.num_turns = 0
        self.history = []

    def reset_image(self, image):
        self.pixel_values = self.bot.process_img(image)
        self.reset_history()

    def apply_template(self, messages):

        messages = self.chat_template.template_map_fn_v2(messages)
        prompt = ''.join([msg['content'] for msg in messages])
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        return prompt

    def chat(self, text, system=None, streamer=None, gen_config=None):

        assert self.chat_template

        if self.num_turns == 0 and system:
            self.update_history('system', system)

        self.update_history('user', text)

        if gen_config is None:
            gen_config = GenerationConfig()

        prompt = self.apply_template(self.history)

        inputs = self.bot.prepare_inputs(prompt, self.pixel_values,
                                         self.num_turns)

        gen_config.stop_words.extend(self.chat_template.stop_words)
        output = self.bot.generate(inputs, streamer, gen_config)
        self.update_history('assistant', output)

        self.num_turns += 1
        return output
