from transformers import TextIteratorStreamer, TextStreamer
from transformers.models.auto import AutoTokenizer


class HFTextIteratorStreamer(TextIteratorStreamer):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 skip_prompt: bool = False,
                 timeout=None,
                 chat_template=None,
                 **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.chat_template = chat_template

    def on_finalized_text(self, text: str, stream_end: bool = False):

        for word in self.chat_template.stop_words:
            text = text.rstrip(word)
        super().on_finalized_text(text, stream_end)


class HFTextStreamer(TextStreamer):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 skip_prompt: bool = False,
                 chat_template=None,
                 **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.chat_template = chat_template

    def on_finalized_text(self, text: str, stream_end: bool = False):

        for word in self.chat_template.stop_words:
            text = text.rstrip(word)
        super().on_finalized_text(text, stream_end)
