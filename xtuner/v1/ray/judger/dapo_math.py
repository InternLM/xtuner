import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from .native import NativeJudger


# _SOLUTION_CLIP_CHARS = 300


# def extract_solution(solution_str, method="strict"):
#     assert method in ["strict", "flexible"]

#     # Optimization: Regular expression matching on very long strings can be slow.
#     # For math problems, the final answer is usually at the end.
#     # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
#     if len(solution_str) > _SOLUTION_CLIP_CHARS:
#         solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

#     if method == "strict":
#         # this also tests the formatting of the model
#         solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
#         if len(solutions) == 0:
#             final_answer = None
#         else:
#             # take the last solution
#             final_answer = solutions[-1].replace(",", "").replace("$", "")
#     elif method == "flexible":
#         answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
#         final_answer = None
#         if len(answer) == 0:
#             # no reward is there is no answer
#             pass
#         else:
#             invalid_str = ["", "."]
#             # find the last number that is not '.'
#             for final_answer in reversed(answer):
#                 if final_answer not in invalid_str:
#                     break
#     return final_answer


# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    # if gt_need_extract:
    #     gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    # else:
    assert not gt_need_extract
    gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    # print("==========", extracted_pred, gt)

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, str | None]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    # acc = correct

    return reward
    # return {
    #     "score": reward,
    #     "acc": acc,
    #     "pred": pred,
    # }


def compute_reward(response, label, extra_info):
    predict_str = response
    # ground_truth = label

    reward = compute_score(response, label)
    overlong_reward = 0
    if extra_info.get("enable_overlong_buffer", None):
        overlong_buffer_len = extra_info["overlong_buffer_len"]
        expected_len = extra_info["max_response_len"] - overlong_buffer_len
        valid_response_length = len(
            extra_info["tokenizer"](predict_str, return_tensors="pt")["input_ids"].flatten().tolist()
        )
        exceed_len = valid_response_length - expected_len
        overlong_penalty_factor = extra_info["overlong_penalty_factor"]
        overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    reward += overlong_reward
    return reward


class DapoMathJudgerConfig(BaseModel):
    extra_info: dict = Field(default={"score": 1, "format_score": 0})
    enable_overlong_buffer: bool
    max_response_len: Optional[int] = None
    overlong_buffer_len: Optional[int] = None
    overlong_penalty_factor: Optional[float] = None
    tokenizer: Any = None

    def __init__(
        self,
        enable_overlong_buffer: bool,
        max_response_len: Optional[int],
        overlong_buffer_len: Optional[int],
        overlong_penalty_factor: Optional[float],
        tokenizer: Any,
    ):
        # 初始化基类
        super().__init__(
            enable_overlong_buffer=enable_overlong_buffer,
            max_response_len=max_response_len,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
            tokenizer=tokenizer,
        )

        # 根据条件更新 extra_info
        if enable_overlong_buffer:
            assert max_response_len is not None
            assert overlong_buffer_len is not None
            assert overlong_penalty_factor is not None
            assert tokenizer is not None
            self.extra_info.update(
                {
                    "enable_overlong_buffer": enable_overlong_buffer,
                    "max_response_len": max_response_len,
                    "overlong_buffer_len": overlong_buffer_len,
                    "overlong_penalty_factor": overlong_penalty_factor,
                    "tokenizer": tokenizer,
                }
            )

    def build(self):
        return NativeJudger(reward_func=compute_reward, extra_info=self.extra_info)


if __name__ == "__main__":
    import json

    data = []
    with open(
        "/cpfs01/user/lishuaibin/projects/202508/xtuner/work_dirs/dapo_math_qwen25-7B/20250828103011/t_1.jsonl",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
    _data = data[0]
    # print(_data)
    prompt = _data["prompt"]
    responses = _data["response"]
    label = _data["label"]
    for res in responses:
        reward = compute_reward(res, label, {})
        # reward = compute_score(res, label, True)

        print(reward)
