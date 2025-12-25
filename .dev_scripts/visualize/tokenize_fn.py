from typing import List, Literal

from xtuner.v1.datasets import PretrainTokenizeFunction, FTDPTokenizeFnConfig, OpenaiTokenizeFunction, Qwen3VLTokenizeFunction


from transformers import AutoTokenizer


def show():
    tokenizer_path = ""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # tokenize_fn = FtdpTokenizeFunction(tokenizer)
    # messages = {"dialogs": [{"role": "user", "content": "<think></think>Hello, <think> how are you?</think>"}, {"role": "assistant", "content": "<think></think>Hello, <think> how are you?</think>"}]}

    # tokenize_fn = OpenaiTokenizeFunction(tokenizer, chat_template="qwen3")
    # messages = {"messages": [{"role": "user", "content": "<think></think>Hello, <think> how <think></think> are you?</think>"}, {"role": "assistant", "content": "<think></think>Hello, <think> how <think></think> are you?</think>"}]}

    # tokenize_fn = PretrainTokenizeFunction(tokenizer)
    # messages = {"messages": [{"role": "pretrain", "content": "<think></think>Hello, <think> how <think></think> are you?</think>"},]}

    # tokenize_fn = Qwen3VLTokenizeFunction(tokenizer, processor_path=tokenizer_path, anno_name="visualize")
    # messages = {"messages": [{"role": "user", "content": "<think></think>Hello, <think> how <think></think> are you?</think>"}, {"role": "assistant", "content": "<think></think><think>Hello, <think> how<think></think> are you?</think>"}]}

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
    print(current_string)


def main():
    show()


if __name__ == "__main__":
    main()

