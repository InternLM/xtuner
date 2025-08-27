import re

from pydantic import BaseModel

from .native import NativeJudger


_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_reward(response, label, extra_info):
    predict_str = response
    ground_truth = label
    answer = extract_solution(predict_str)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return extra_info["score"]
        else:
            return extra_info["format_score"]


class GSM8KJudgerConfig(BaseModel):
    extra_info: dict = {"score": 1, "format_score": 0}

    def build(self):
        return NativeJudger(reward_func=compute_reward, extra_info=self.extra_info)
