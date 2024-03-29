from typing import Callable, Dict, List, Type

from mmengine.config.lazy import LazyObject

from xtuner.types import ChatMessages


def map_protocol(
    input_keys: Dict[str, Type] = {},
    output_keys: Dict[str, Type] = {},
    added_keys: Dict[str, Type] = {},
) -> Callable:

    def decorator(func):

        def wrapper(data, *args, **kwargs):

            for key, _type in input_keys.items():
                assert key in data
                if not isinstance(data[key], _type):
                    breakpoint()

            result = func(data, *args, **kwargs)

            for key, _type in output_keys.items():
                assert key in result
                assert isinstance(result[key], _type)

            return result

        return wrapper

    setattr(decorator, 'input_keys', input_keys)
    setattr(decorator, 'output_keys', output_keys)
    setattr(decorator, 'added_keys', added_keys)

    return decorator


def map_sequential(mappings: List[Callable]):

    if not isinstance(mappings, List):
        mappings = list(mappings)

    for i in range(len(mappings)):
        if isinstance(mappings[i], LazyObject):
            mappings[i] = mappings[i].build()

    def _sequential(item, tokenizer, chat_template):

        for func in mappings:
            item = func(item, tokenizer, chat_template)

        return item

    return _sequential


@map_protocol(
    input_keys=dict(messages=list),
    output_keys=dict(input_ids=list, labels=list, image_urls=list),
)
def openai_to_raw_training(item: dict, tokenizer, chat_template) -> Dict:

    data = ChatMessages.from_dict(item)
    data = data.tokenize(tokenizer, chat_template)

    return data
