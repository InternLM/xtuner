"""Preprocess the GSM8k dataset to parquet format."""

import argparse
import os
import re

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="openai/gsm8k")
    parser.add_argument("--out-dir")

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.input_dir, "default")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    # Adapted from https://github.com/verl-project/verl/blob/c37d4d53850906aced4c071666340ec26966d707/examples/data_preprocess/gsm8k_tool_agent_loop.py#L62
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": "openai/gsm8k",
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_gsm8k_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer in the format of `#### <answer>`."
                        ),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_gsm8k_reward": {
                            "create_kwargs": {"ground_truth": solution},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    train_dataset.to_json(os.path.join(out_dir, "train.jsonl"), orient="records", lines=True)
    test_dataset.to_json(os.path.join(out_dir, "test.jsonl"), orient="records", lines=True)
