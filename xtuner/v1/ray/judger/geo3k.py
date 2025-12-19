import re
from typing import Callable


try:
    from mathruler.grader import extract_boxed_content, grade_answer
except Exception:
    extract_boxed_content = None
    grade_answer = None

from .native import NativeJudgerConfig


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if extract_boxed_content is None:
        raise ImportError("Please install mathruler by 'pip install mathruler pylatexenc'")
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_reward(response, label, extra_info) -> dict:
    format_score = extra_info["format_score"]
    use_boxed = extra_info["use_boxed"]
    acc = acc_reward(response, label, use_boxed)
    score = (1.0 - format_score) * acc + format_score * format_reward(response)
    return {"score": score, "acc": acc}


class GEO3KJudgerConfig(NativeJudgerConfig):
    """Configuration for the GEO3K judger."""

    judger_name: str = "hiyouga/geometry3k"
    extra_info: dict = {"format_score": 0.1, "use_boxed": True}
    reward_func: Callable = compute_reward
