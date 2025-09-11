import re

from pydantic import BaseModel


try:
    from mathruler.grader import extract_boxed_content, grade_answer
except Exception:
    extract_boxed_content = None
    grade_answer = None

from .native import NativeJudger


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if extract_boxed_content is None:
        raise ImportError("Please install mathruler by 'pip install mathruler'")
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_reward(response, label, extra_info) -> float:
    format_score = extra_info["format_score"]
    use_boxed = extra_info["use_boxed"]
    return (1.0 - format_score) * acc_reward(response, label, use_boxed) + format_score * format_reward(response)


class GEO3KJudgerConfig(BaseModel):
    """Configuration for the GSM8K judger."""

    extra_info: dict = {"format_score": 0.1, "use_boxed": True}

    def build(self):
        """Build a NativeJudger instance from the configuration.

        Returns:
            NativeJudger: An instance of the NativeJudger configured for GSM8K.
        """
        return NativeJudger(reward_func=compute_reward, extra_info=self.extra_info)
