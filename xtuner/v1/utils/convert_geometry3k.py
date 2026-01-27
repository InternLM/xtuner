"""Preprocess the geometry3k dataset to parquet format."""

import argparse
import json
import os

import datasets
from PIL import Image


def save_jsonl(data_list, output_file):
    with open(output_file, "w", encoding="utf-8") as writer:
        for d in data_list:
            writer.write(json.dumps(d, ensure_ascii=False) + "\n")


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

    image_root = os.path.join(args.out_dir, "images")
    os.makedirs(image_root, exist_ok=True)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem").replace("<image>", "<IMG_CONTEXT>")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            assert len(images) == 1, f"image {len(images)}"
            image = images[0]
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                raise NotImplementedError

            image_path = os.path.join("images", f"{split}_{idx}.jpg")
            image.save(os.path.join(args.out_dir, image_path))
            image_wh = [image.width, image.height]

            # openai format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path, "image_wh": image_wh}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            data = {
                "data_source": "hiyouga/geometry3k",
                "prompt": messages,
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

    out_dir = args.out_dir
    new_data = []
    for i, data_item in enumerate(train_dataset):
        new_dict = {
            "prompt": [],
            "data_source": data_item["data_source"],
            "ability": data_item["ability"],
            "reward_model": data_item["reward_model"],
            "extra_info": data_item["extra_info"],
        }
        content = data_item["prompt"][0]["content"]
        del content[0]["text"]
        del content[1]["image_url"]
        prompt = data_item["prompt"]
        new_dict["prompt"] = prompt
        new_data.append(new_dict)
    save_jsonl(new_data, os.path.join(out_dir, "train.jsonl"))

    new_data = []
    for i, data_item in enumerate(test_dataset):
        new_dict = {
            "prompt": [],
            "data_source": data_item["data_source"],
            "ability": data_item["ability"],
            "reward_model": data_item["reward_model"],
            "extra_info": data_item["extra_info"],
        }
        content = data_item["prompt"][0]["content"]
        del content[0]["text"]
        del content[1]["image_url"]
        prompt = data_item["prompt"]
        new_dict["prompt"] = prompt
        new_data.append(new_dict)
    save_jsonl(new_data, os.path.join(out_dir, "test.jsonl"))

    # train_dataset.to_json(os.path.join(out_dir, "train.jsonl"), orient="records", lines=True)
    # test_dataset.to_json(os.path.join(out_dir, "test.jsonl"), orient="records", lines=True)
