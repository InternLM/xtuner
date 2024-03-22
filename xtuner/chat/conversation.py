from xtuner.chat.backend import HFBackend
from xtuner.types.chat import (ChatMsg, HybridChatMessages, ImageContentItem,
                               TextContentItem)


class Conversation():

    def __init__(self,
                 backend: HFBackend,
                 name=None,
                 system=None,
                 functions=None,
                 code_interpreter=None) -> None:

        self.name = name
        self.backend = backend
        self.system = system
        self.functions = functions
        self.code_interpreter = code_interpreter
        self._messages = HybridChatMessages()

        if system:
            msg = ChatMsg(role='system', content=system)
            self._messages.messages.append(msg)

    @property
    def messages(self):
        return self._messages

    def add_message(self, role, content):
        if role == 'system':
            assert isinstance(content, str)
            msg = ChatMsg(role='system', content=content)
            self._messages.messages.append(msg)
        elif role == 'user':
            self._add_user(content)
        elif role == 'assistant':
            assert isinstance(content, str)
            msg = ChatMsg(role='assistant', content=content)
            self._messages.messages.append(msg)

    def _add_user(self, content):

        if isinstance(content, str):
            msg = ChatMsg(role='user', content=content)
            self._messages.messages.append(msg)
        elif isinstance(content, list):
            _content = []
            for item in content:
                if isinstance(item, (ImageContentItem, TextContentItem)):
                    _content.append(item)
                    continue

                assert isinstance(item, dict)
                assert 'type' in item
                assert item['type'] in item
                if item['type'] == 'image_url':
                    _item = ImageContentItem(image_url=item['image_url'])
                    _content.append(_item)
                elif item['type'] == 'text':
                    _item = TextContentItem(text=item['text'])
                    _content.append(_item)
                else:
                    raise NotImplementedError

            msg = ChatMsg(role='user', content=_content)
            self._messages.messages.append(msg)
        else:
            raise TypeError

    def run(self, sample_params=None, streamer=None):

        self.add_message(role='user', content=content)
        response = self.backend.chat(self.messages)
        self.add_message(role='assistant', content=response)
        return response

    def regenerate(self):

        assert self._messages.messages[-1].role == 'assistant'
        self._messages.messages.pop()
        return self.backend.chat(self.messages)

    def create_streamer(self, iterable=False):
        return self.backend.create_streamer(iterable=iterable)


if __name__ == '__main__':

    from xtuner.types import HybridChatTemplate
    chat_template = HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
        image_token='<image>',
        function_call=
        '{assistant}<|action_start|><|plugin|>\n{function_call}<|action_end|><|im_end|>\n',  # noqa: E501, E251
        function_result=
        '<|im_start|>environment name=<|plugin|>\n{function_result}<|im_end|>\n<|im_start|>assistant\n',  # noqa: E501, E251
        functions='<|im_start|>system name=<|plugin|>\n{functions}<|im_end|>\n'
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from xtuner.chat.backend import HFBackend, VisionEncoderForDeploy

    llm = AutoModelForCausalLM.from_pretrained(
        '/mnt/petrelfs/share_data/linzhihao/model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f',
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        '/mnt/petrelfs/share_data/linzhihao/model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f',
        trust_remote_code=True)
    vision_tower = VisionEncoderForDeploy(
        model_name_or_path='openai/clip-vit-large-patch14-336',
        adapter_name_or_path=
        '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/visual_encoder_adapter',
        projector_name_or_path=
        '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/projector'
    )

    llm.cuda()

    backend = HFBackend(
        chat_template,
        llm,
        tokenizer,
        vision_tower,
    )

    conv = Conversation(backend)
    print(conv.chat('who are you?'))

    from xtuner.chat.backend import LMDeployBackend
    backend = LMDeployBackend(
        chat_template,
        '/mnt/petrelfs/share_data/linzhihao/model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f',
        vision_tower)
    conv = Conversation(backend)
    print(conv.chat('who are you?'))

    content = [
        TextContentItem(text='Please describe this image'),
        ImageContentItem(image_url='llava.jpeg')
    ]

    print(conv.chat(content))
