from typing import List, Literal, Iterable
from cyclopts import App
from pathlib import Path

from xtuner.v1.datasets import PretrainTokenizeFunction, FTDPTokenizeFnConfig, OpenaiTokenizeFunction, Qwen3VLTokenizeFunction, CachableTokenizeFunction
import jsonlines


from transformers import AutoTokenizer
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from pypager import Pager
from pypager.source import GeneratorSource


app = App()


def show_iterable(data: Iterable[dict], tokenize_fn: CachableTokenizeFunction, tokenizer):
    sep = "=" * 80
    color_prefix = "\033[31m"
    color_suffix = "\033[0m"
    current_string = ""

    current_type: Literal["positive", "negative"]
    token_type: Literal["positive", "negative"]

    def flush_tokens(current_tokens: List[int]) -> str:
        if not current_tokens:
            return ""
        text = tokenizer.decode(current_tokens, skip_special_tokens=False)
        if current_type == "positive":
            return f"{color_prefix}{text}{color_suffix}"
        return text

    for messages in data:
        res = tokenize_fn(messages)
        token_ids, labels = res["input_ids"], res["labels"]
        current_string = ""
        current_tokens: List[int] = []
        current_type = "negative"

        for i, label in zip(token_ids, labels):
            token_type = "positive" if label >= 0 else "negative"
            if token_type != current_type:
                current_string += flush_tokens(current_tokens)
                current_type = token_type
                current_tokens = []
            current_tokens.append(i)

        current_string += flush_tokens(current_tokens)
        current_string += f"\n{sep}\n"
        yield to_formatted_text(ANSI(current_string))



@app.default
def main(tokenizer_path: Path, data_path: Path, debug=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenize_fn = Qwen3VLTokenizeFunction(tokenizer, processor_path=str(tokenizer_path), anno_name="visualize")

    with jsonlines.open(data_path) as reader:
        if not debug:
            pager = Pager()
            pager.add_source(GeneratorSource(show_iterable(reader, tokenize_fn, tokenizer)))
            pager.run()
        else:
            list(show_iterable(reader, tokenize_fn, tokenizer))


if __name__ == "__main__":
    app()

