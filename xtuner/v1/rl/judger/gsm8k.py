import re
from typing import Callable

from .native import JudgerConfig


_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """Extract the numerical solution from a string.

    Args:
        solution_str (str): The string containing the solution.
        method (str): The extraction method, either "strict" or "flexible".
            "strict" requires the solution to be in the format "#### <number>".
            "flexible" extracts the last numerical value found.
            Defaults to "strict".

    Returns:
        str or None: The extracted numerical solution as a string, or None if
            not found.
    """
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
    """Compute the reward for a given response based on the GSM8K dataset and
    criteria.

    Args:
        response (str): The model's generated response.
        label (str): The ground-truth answer.
        extra_info (dict): A dictionary containing scoring information,
            e.g., `{"score": 1, "format_score": 0}`.

    Returns:
        int or float: The calculated reward.
    """
    predict_str = response
    ground_truth = label
    answer = extract_solution(predict_str)
    if answer is None:
        return {"score": 0}
    else:
        if answer == ground_truth:
            return {"score": extra_info["score"]}
        else:
            return {"score": extra_info["format_score"]}


class GSM8KJudgerConfig(JudgerConfig):
    """Configuration for the built-in GSM8K judger.

    ``GSM8KJudgerConfig`` scores mathematical reasoning responses by extracting
    the final numeric answer and comparing it with the ground-truth answer. It
    is a preset ``JudgerConfig`` for the ``openai/gsm8k`` task.

    Args:
        judger_name (str): Logical judger name. Defaults to "openai/gsm8k".
        extra_info (dict): Reward values used by the GSM8K reward function.
            Defaults to ``{"score": 1, "format_score": 0}``.
        reward_handler (Callable | str): Reward handler used to compute the
            score. Defaults to ``compute_reward``.
        request_timeout (float): Timeout in seconds for HTTP reward handlers.
            Defaults to 30.0.
        num_ray_actors (int): Number of remote Ray actor judger replicas.
            ``0`` runs the judger locally. Defaults to 0.
        num_cpus_per_actor (int): CPU cores requested by each remote judger
            actor. Defaults to 1.
        cpu_memory_per_actor (int): CPU memory in bytes requested by each
            remote judger actor. Defaults to 1 GiB.

    **Examples:**

    Example GSM8K judger::

        config = GSM8KJudgerConfig()
    """

    judger_name: str = "openai/gsm8k"
    extra_info: dict = {"score": 1, "format_score": 0}
    reward_handler: Callable | str = compute_reward
