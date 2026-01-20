import re
from typing import Any, Callable, List, Optional, Tuple

from pydantic import ConfigDict, Field

from .native import NativeJudgerConfig


# Adapted from https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math_dapo.py


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
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))  # type: ignore[arg-type]
    else:
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
) -> Tuple[bool, Optional[str]]:
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
        return correct == 1, pred  # type: ignore[arg-type]

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred  # type: ignore[arg-type]


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
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
    acc = correct

    return {"score": reward, "acc": acc}


def compute_reward(response, label, extra_info):
    predict_str = response

    eos_token = extra_info["eos_token"]
    if isinstance(eos_token, list):
        for eos in eos_token:
            if response.endswith(eos):
                response = response[: -len(eos)]
                break
    else:
        if response.endswith(eos_token):
            response = response[: -len(eos_token)]

    out = compute_score(response, label)
    reward = out["score"]

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
    return {"score": reward, "acc": out["acc"]}


class DapoMathJudgerConfig(NativeJudgerConfig):
    model_config = ConfigDict(extra="forbid")
    eos_token: List[str] | str
    enable_overlong_buffer: bool
    score: int = 1
    format_score: int = 0
    max_response_len: Optional[int] = None
    overlong_buffer_len: Optional[int] = None
    overlong_penalty_factor: Optional[float] = None
    tokenizer: Any = Field(default=None, exclude=True)
    reward_func: Callable = Field(default=compute_reward, exclude=True)

    def __init__(
        self,
        judger_name: str,
        eos_token: List[str] | str,
        enable_overlong_buffer: bool,
        max_response_len: Optional[int],
        overlong_buffer_len: Optional[int],
        overlong_penalty_factor: Optional[float],
        tokenizer: Any,
        score: int = 1,
        format_score: int = 0,
    ):
        if isinstance(eos_token, str):
            assert eos_token.strip() != "", "eos_token string must not be empty"
        elif isinstance(eos_token, list):
            assert all(isinstance(e, str) and e.strip() != "" for e in eos_token), (
                "All eos_token list elements must be non-empty strings"
            )
            assert len(eos_token) > 0, "eos_token list must not be empty"
        else:
            raise TypeError("eos_token must be a non-empty string or a non-empty list of strings")

        # 初始化基类
        super().__init__(
            judger_name=judger_name,
            eos_token=eos_token,
            enable_overlong_buffer=enable_overlong_buffer,
            score=score,
            format_score=format_score,
            max_response_len=max_response_len,
            overlong_buffer_len=overlong_buffer_len,
            overlong_penalty_factor=overlong_penalty_factor,
            tokenizer=tokenizer,
        )

        self.extra_info.update(
            {
                "eos_token": eos_token,
                "score": score,
                "format_score": format_score,
            }
        )

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
