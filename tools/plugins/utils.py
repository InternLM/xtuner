import copy

from transformers import StoppingCriteria


def get_plugins_stop_criteria(base, tokenizer, command_stop_word,
                              answer_stop_word):
    command = copy.deepcopy(base)
    answer = copy.deepcopy(base)
    if command_stop_word is not None:
        command.append(StopWordStoppingCriteria(tokenizer, command_stop_word))
    if answer_stop_word is not None:
        answer.append(StopWordStoppingCriteria(tokenizer, answer_stop_word))
    return command, answer


class StopWordStoppingCriteria(StoppingCriteria):
    """Stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word
