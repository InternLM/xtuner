import re
from typing import List

import ray
from pydantic import BaseModel, ConfigDict
from ray.actor import ActorClass

from .native import NativeJudger


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


class GSM8KJudgerConfig(BaseModel):
    """Configuration for the GSM8K judger."""

    judger_name: str = "openai/gsm8k"
    model_config = ConfigDict(extra="forbid")
    extra_info: dict = {"score": 1, "format_score": 0}
    num_ray_actors: int = 1

    def build_actor(self, pg, start_bundle_idx) -> List[ActorClass]:
        """Build the actor class for the judger.

        Returns:
            List[ActorClass]: The actor class for the judger.
        """
        workers_list = []
        pg_options = {"num_cpus": pg.bundle_specs[0].get("CPU", 1)}
        for idx in range(self.num_ray_actors):
            worker = (
                ray.remote(NativeJudger)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=(start_bundle_idx + idx),
                    **pg_options,
                )
                .remote(judger_name=self.judger_name, reward_func=compute_reward, extra_info=self.extra_info)
            )
            workers_list.append(worker)
        return workers_list
