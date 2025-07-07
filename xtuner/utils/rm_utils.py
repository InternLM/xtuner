# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import List, Union

import requests
from transformers import AutoTokenizer


class RewardModelClient:
    """This class is used to process the input sequences for the reward
    model."""

    def __init__(
        self,
        path,
        max_length=16384,
        max_response_length=4096,
        response_cut_side="left",
        server_type="sglang",
        server_address="127.0.0.1:30000",
    ):
        """
        Args:
            path: Path to the reward model.
            max_length: Maximum length of the input sequence.
            max_response_length: Maximum length of the response sequence.
            response_cut_side: Side to cut the response sequence if it exceeds the maximum length.
            server_type: Type of the server, can be "sglang", "vllm", or "lmdeploy".
            server_address: Address of the reword model server.
        """
        self.rm_name = path.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # for final reward token and one <|reward|> token and two '\n' tokens
        self.max_length = max_length - 4
        self.max_response_length = max_response_length
        self.response_cut_side = response_cut_side
        self.server_type = server_type
        self.server_address = server_address

    def _encode(self, prompt, reference, output, wrapper="sft") -> str:
        """Construct the input string for the reward model.

        Args:
            prompt: Prompt.
            reference: Reference trajectory.
            output: Candidate trajectory.
            wrapper: The wrapper type. Can be "sft" or "pretrain".
        Returns:
            The constructed input string for RM.
        """
        p = (
            "\n".join([e["content"] for e in prompt])
            if isinstance(prompt, list)
            else prompt
        )
        r1 = (
            "\n".join([e["content"] for e in reference])
            if isinstance(reference, list)
            else reference
        )
        r2 = (
            "\n".join([e["content"] for e in output])
            if isinstance(output, list)
            else output
        )

        p_ids = self.tokenizer.encode(p, add_special_tokens=True)
        r1_ids = self.tokenizer.encode(r1, add_special_tokens=True)
        r2_ids = self.tokenizer.encode(r2, add_special_tokens=True)

        if len(r1_ids) > self.max_response_length:
            print(
                f"Reference sequence length {len(r1_ids)} is "
                f"larger than max_response_length {self.max_response_length}",
            )
            if self.response_cut_side == "right":
                r1_ids = r1_ids[: self.max_response_length]
            else:
                r1_ids = r1_ids[-self.max_response_length :]
        if len(r2_ids) > self.max_response_length:
            print(
                f"Output sequence length {len(r2_ids)} is "
                f"larger than max_response_length {self.max_response_length}",
            )
            if self.response_cut_side == "right":
                r2_ids = r2_ids[: self.max_response_length]
            else:
                r2_ids = r2_ids[-self.max_response_length :]

        max_prompt_length = (self.max_length - len(r1_ids) - len(r2_ids)) // 2

        if len(p_ids) > max_prompt_length:
            print(
                f"Prompt sequence length {len(p_ids)} is "
                f"larger than max_prompt_length {max_prompt_length}",
            )
            p_ids = p_ids[-max_prompt_length:]

        p = self.tokenizer.decode(p_ids, skip_special_tokens=True)
        r1 = self.tokenizer.decode(r1_ids, skip_special_tokens=True)
        r2 = self.tokenizer.decode(r2_ids, skip_special_tokens=True)

        # Fit the template of RM
        _reference_cat = (
            p + r1 if wrapper == "pretrain" or len(r1) == "" else p + "\n" + r1
        )
        _output_cat = (
            p + r2 if wrapper == "pretrain" or len(r2) == "" else p + "\n" + r2
        )

        final_txt = _reference_cat + "<|reward|>" + _output_cat + "[UNUSED_TOKEN_130]"

        return final_txt

    def encode(self, data) -> Union[str, List[str]]:
        """Encode the input data into a format suitable for RM.

        Args:
            data: A dictionary or a list of dictionary containing the keys
                  'prompt', 'reference', 'output', and optionally 'wrapper'.
        Returns:
            The encoded input string for RM.
        """
        if isinstance(data, dict):
            return self._encode(**data)
        elif isinstance(data, list):
            return [
                self._encode(**item) if isinstance(item, dict) else item
                for item in data
            ]
        else:
            raise ValueError(
                "Input data must be a dictionary or a list of dictionaries."
            )

    def sglang_request_reward(
        self, data, retry_delay=0.2, max_retries=8
    ) -> List[float]:
        for i in range(max_retries):
            try:
                res = requests.post(
                    f"http://{self.server_address}/classify",
                    json={
                        "model": self.rm_name,
                        "text": data,
                    },
                )
                rewards = [e["embedding"][0] for e in res.json()]
                return rewards
            except Exception as e:
                print(f"Error requesting reward: {e}")
                print(f"Raw response: {data}")
                time.sleep(retry_delay)
                continue
        print(f"Failed to request reward after {max_retries} retries")
        return None

    def vllm_request_reward(self, data, retry_delay=0.2, max_retries=8) -> List[float]:
        for i in range(max_retries):
            try:
                res = requests.post(
                    f"http://{self.server_address}/pooling",
                    json={
                        "input": data,
                    },
                )
                rewards = [e["data"][-1][0] for e in res.json()["data"]]
                return rewards
            except Exception as e:
                print(f"Error requesting reward: {e}")
                print(f"Raw response: {data}")
                time.sleep(retry_delay)
                continue
        print(f"Failed to request reward after {max_retries} retries")
        return None

    def lmdeploy_request_reward(
        self, data, retry_delay=0.2, max_retries=8
    ) -> List[float]:
        for i in range(max_retries):
            try:
                res = requests.post(
                    f"http://{self.server_address}/pooling",
                    json={
                        "input": data,
                    },
                )
                rewards = [e["data"] for e in res.json()["data"]]
                return rewards
            except Exception as e:
                print(f"Error requesting reward: {e}")
                print(f"Raw response: {data}")
                time.sleep(retry_delay)
                continue
        print(f"Failed to request reward after {max_retries} retries")
        return None

    def __call__(self, data) -> List[float]:
        """Call the input wrapper to construct the input string for RM.

        Args:
            data: A list of dictionaries containing the keys
                  'prompt', 'reference', 'output', and optionally 'wrapper'.
            retry_delay: Delay in seconds before retrying the request.
            max_retries: Maximum number of retries for the request.
        Returns:
            scores: The list of reward scores returned by the RM server.
                    If the request fails, it returns None.
        """
        data = self.encode(data)
        if self.server_type == "sglang":
            scores = self.sglang_request_reward(data)
        elif self.server_type == "vllm":
            scores = self.vllm_request_reward(data)
        elif self.server_type == "lmdeploy":
            scores = self.lmdeploy_request_reward(data)
        else:
            raise ValueError(f"Unsupported server type: {self.server_type}")

        return scores


if __name__ == "__main__":
    # Example usage
    ex1 = [
        {
            "prompt": "How many 'r's are in the word 'strawberry'?",
            "output": "There are three 'r's in the word 'strawberry'.",
            "reference": "3.",
        },
        {
            "prompt": "How many 'r's are in the word 'strawberry'?",
            "reference": "3.",
            "output": "There are two 'r's in the word 'strawberry'.",
        },
    ]

    ex2 = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "How many 'r's are in the word 'strawberry'?",
                }
            ],
            "reference": [{"role": "assistant", "content": "3."}],
            "output": [
                {
                    "role": "assistant",
                    "content": "There are three 'r's in the word 'strawberry'.",
                }
            ],
        },
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "How many 'r's are in the word 'strawberry'?",
                }
            ],
            "reference": [{"role": "assistant", "content": "3."}],
            "output": [
                {
                    "role": "assistant",
                    "content": "There are two 'r's in the word 'strawberry'.",
                }
            ],
        },
    ]

    # sglang
    client = RewardModelClient(
        "internlm/POLAR-7B", server_type="sglang", server_address="127.0.0.1:30000"
    )

    scores = client(ex1)
    print(scores)

    encoded_text = client.encode(ex2)
    scores = client.sglang_request_reward(encoded_text)
    print(scores)

    # vllm
    client = RewardModelClient(
        "internlm/POLAR-7B", server_type="vllm", server_address="127.0.0.1:30000"
    )

    scores = client(ex1)
    print(scores)

    encoded_text = client.encode(ex2)
    scores = client.vllm_request_reward(encoded_text)
    print(scores)

    # lmdeploy
    client = RewardModelClient(
        "internlm/POLAR-7B", server_type="lmdeploy", server_address="127.0.0.1:30000"
    )

    scores = client(ex1)
    print(scores)

    encoded_text = client.encode(ex2)
    scores = client.lmdeploy_request_reward(encoded_text)
    print(scores)
