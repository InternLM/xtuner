from typing import Dict, List, Literal, Union

from pydantic import BaseModel

from .chat_template import HybridChatTemplate


class TextContentItem(BaseModel):
    type: Literal['text']
    text: str

    def format_content(self, chat_template: HybridChatTemplate) -> str:
        return self.text


class ImageContentItem(BaseModel):
    type: Literal['image_url']
    image_url: str

    def format_content(self, chat_template: HybridChatTemplate) -> str:
        return chat_template.image_token


MultModalContentType = Union[TextContentItem, ImageContentItem]
ContentType = Union[str, List[MultModalContentType]]


class ChatMsg(BaseModel):
    role: Literal['assistant', 'user', 'system']
    content: ContentType

    def collect_img_urls(self) -> List[str]:
        img_urls = []
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, ImageContentItem):
                    img_urls.append(item.image_url)
        return img_urls

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:

        if isinstance(self.content, str):
            text = self.content
        elif isinstance(self.content, list):
            text = ''
            for i, item in enumerate(self.content):
                if i == 0:
                    text += item.format_content(chat_template)
                else:
                    text += '\n' + item.format_content(chat_template)
        else:
            raise NotImplementedError

        if self.role == 'system':
            prompt = chat_template.decorate_system(text)
        elif self.role == 'user':
            prompt = chat_template.decorate_user(text)
        elif self.role == 'assistant':
            prompt = chat_template.decorate_assistant(text)
        else:
            raise NotImplementedError

        return prompt


# Function Call


class FunctionCall(BaseModel):
    name: str
    arguments: Dict


class FunctionCallMsg(BaseModel):

    role: Literal['assistant']
    content: str
    function_call: Union[str, Dict]

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:

        return chat_template.decorate_function_call(self.content,
                                                    self.function_call)


class FunctionResultMsg(BaseModel):
    role: Literal['function']
    name: str
    content: Union[str, Dict]

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return chat_template.decorate_function_result(self.content)


class CodeInterpreterCallMsg(BaseModel):

    role: Literal['assistant']
    content: str
    conde_interpreter_call: Union[str, Dict]

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:

        return chat_template.decorate_code_interpreter_call(
                    self.content, self.conde_interpreter_call)



class CodeInterpreterResultMsg(BaseModel):
    role: Literal['function']
    name: str
    content: Union[str, Dict]

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:
        return chat_template.decorate_code_internpreter_result(self.content)




class Functions(BaseModel):

    # class Parameters(BaseModel):

    #     class Property(BaseModel):
    #         type: str
    #         description: str
    #         enum: Optional[List] = None

    #     type: Literal['object']
    #     properties: Dict[str, Property]
    #     required: List[str]

    name: str
    description: Union[str, Dict]
    parameters: Union[str, Dict]
    


class CodeInterpreter(BaseModel):

    # class Parameters(BaseModel):

    #     class Property(BaseModel):
    #         type: str
    #         description: str
    #         enum: Optional[List] = None

    #     type: Literal['object']
    #     properties: Dict[str, Property]
    #     required: List[str]

    name: str
    description: Union[str, Dict]
    



HybridChatMsgType = Union[ChatMsg, FunctionCallMsg, FunctionResultMsg]


class HybridChatMessages(BaseModel):

    messages: List[HybridChatMsgType] = []
    # images: List[Image.Image] = []
    functions: List[Functions] = []

    # TODO (pppppM) add audio and video

    def collect_img_urls(self) -> List[str]:
        img_urls = []
        for msg in self.messages:
            img_urls.extend(msg.collect_img_urls())
        return img_urls

    def pop_latest_msg(self):
        return self.messages.pop()

    def apply_chat_template(self, chat_template: HybridChatTemplate) -> str:

        prompt = ''
        if len(self.functions) > 0:

            functions = [func.model_dump() for func in self.functions]

            prompt += chat_template.decorate_functions(functions)

        for msg in self.messages:
            prompt += msg.apply_chat_template(chat_template)

        return prompt
