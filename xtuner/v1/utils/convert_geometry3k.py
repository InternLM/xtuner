"""Preprocess the GSM8k dataset to parquet format."""

import argparse
import os
import datasets


# Adapted from https://github.com/volcengine/verl/blob/main/examples/data_preprocess/geo3k.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="hiyouga/geometry3k")
    parser.add_argument("--out-dir")

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.input_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            data = {
                "data_source": "hiyouga/geometry3k",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    train_dataset.to_json(os.path.join(out_dir, "train.jsonl"), orient="records", lines=True)
    test_dataset.to_json(os.path.join(out_dir, "test.jsonl"), orient="records", lines=True)
