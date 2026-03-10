import asyncio
import random
from typing import List

import aiohttp
import ray
import requests  # type: ignore[import-untyped]
from pydantic import ConfigDict
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.rl.judger.native import RouterJudgerConfig
from xtuner.v1.utils.type_helper import ray_method


verify_prompt = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""


class CompassVerifierV2:
    def __init__(
        self,
        hosts: list[str],
        request_timeout: float = 30.0,
        max_retries: int = 3,
        thinking_finish_words: list[str] | None = None,
    ):
        if not hosts:
            raise ValueError("CompassVerifierV2 requires at least one host.")
        self.hosts = hosts
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.thinking_finish_words = thinking_finish_words or ["<conclude>", "**Final Answer**", "</think>"]
        self.model_name = requests.get(
            f"http://{self.hosts[0]}/v1/models",
            headers={"Authorization": "Bearer "},
            timeout=request_timeout,
        ).json()["data"][0]["id"]
        self.judger_name = "compass_verifier_v2"

    @ray_method
    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        if rollout_state.status != Status.COMPLETED or rollout_state.response is None:
            rollout_state.reward = {"score": -1}
            return rollout_state

        question = rollout_state.message[-1]["content"]
        model_answer = rollout_state.response.replace("<|im_end|>", "").strip()
        for thinking_finish_word in self.thinking_finish_words:
            if thinking_finish_word in model_answer:
                model_answer = model_answer.split(thinking_finish_word)[-1]

        answer_lines = model_answer.split("\n")
        if len(answer_lines) > 10:
            model_answer = "\n".join(answer_lines[-10:])
        if len(model_answer) > 1000:
            model_answer = model_answer[-1000:]

        assert rollout_state.reward_model is not None and "ground_truth" in rollout_state.reward_model, (
            "RolloutState must have reward_model with 'ground_truth' for CompassVerifierV2."
        )
        outcome_reward = await self._judge_with_llm(question, model_answer, rollout_state.reward_model["ground_truth"])
        rollout_state.reward = {"score": outcome_reward}
        return rollout_state

    async def _judge_with_llm(self, question: str, model_response: str, label: str):
        headers = {"Content-Type": "application/json"}
        prompt = verify_prompt.format(question=question, llm_response=model_response, gold_answer=label)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
        for i in range(self.max_retries):
            host = random.choice(self.hosts)
            url = f"http://{host}/v1/chat/completions"
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                    ) as response:
                        response_json = await response.json()
                        if response.status != 200:
                            error_msg = response_json.get("error", {}).get("message", "Unknown error")
                            raise RuntimeError(f"API request failed with status {response.status}: {error_msg}")
                        res_str = response_json["choices"][0]["message"]["content"]
                        return 1 if res_str.strip() == "A" else -1
                except Exception as e:
                    await asyncio.sleep(1)
                    print(f"[Judger]: Error try {i}: {str(e)}")
        raise RuntimeError(f"Cannot connect to judger service for {self.max_retries} times.")

    def get_judger_name(self) -> str:
        return self.judger_name


class CompassVerifierV2Config(RouterJudgerConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    hosts: list[str]
    judger_name: str = "compass_verifier_v2"
    request_timeout: float = 30.0
    max_retries: int = 3
    thinking_finish_words: list[str] = ["<conclude>", "**Final Answer**", "</think>"]

    def _build_workers(self, pg: PlacementGroup | None = None, start_bundle_idx: int = 0) -> List[ray.actor.ActorHandle]:
        if pg is None:
            from xtuner.v1.rl.utils.ray_worker import CPUResourcesConfig

            cpu_resource_cfg = CPUResourcesConfig(
                num_workers=self.num_ray_actors,
                num_cpus_per_worker=self.num_cpus_per_actor,
                cpu_memory_per_worker=self.cpu_memory_per_actor,
            )
            pg = cpu_resource_cfg.build_placement_group()
            ray.get(pg.ready())
            start_bundle_idx = 0

        workers_list = []
        assert len(pg.bundle_specs) >= self.num_ray_actors, (
            "Placement group does not have enough bundles for the number of ray actors."
        )
        for idx in range(self.num_ray_actors):
            bundle_idx = start_bundle_idx + idx
            pg_options = {"num_cpus": self.num_cpus_per_actor, "memory": self.cpu_memory_per_actor}
            worker = (
                ray.remote(CompassVerifierV2)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    **pg_options,
                )
                .remote(
                    hosts=self.hosts,
                    request_timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    thinking_finish_words=self.thinking_finish_words,
                )
            )
            workers_list.append(worker)
        return workers_list
