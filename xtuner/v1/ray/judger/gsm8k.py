import re

import ray

from xtuner.v1.ray.judger.worker import JudgerWorker


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


# TODO(hha): 所有的函数式 reward 应该用同一个 judge worker 而不是每个任务都创建一个新的 worker
@ray.remote
class GSM8KJudgerWorker(JudgerWorker):
    def __init__(
        self,
        config,
        rank: int,
        master_addr: str,
        master_port: int,
        world_size: int,
        accelerator: str = "CPU",
        format_score: float = 0.0,
        score: float = 1.0,
    ):
        self.format_score = format_score
        self.score = score
        super().__init__(config, rank, master_addr, master_port, world_size, accelerator)

    def judge_function(self, response, label):
        predict_str = response
        ground_truth = label
        answer = extract_solution(predict_str)
        if answer is None:
            return 0
        else:
            if answer == ground_truth:
                return self.score
            else:
                return self.format_score
