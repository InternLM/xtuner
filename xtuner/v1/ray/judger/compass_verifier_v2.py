import asyncio
import random
from typing import List

import aiohttp
import ray
import requests
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLJudgerResponseItem
from xtuner.v1.ray.judger.native import NativeJudgerConfig

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
    """Base class for judgers, providing a standard interface for executing a
    judging process, which can be either a local function or a remote service.

    The judger orchestrates a three-step pipeline:
    1. Pre-process the input data.
    2. Execute the core logic (local function or remote HTTP call).
    3. Post-process the result.
    """

    def __init__(
        self,
        hosts=[],
        request_timeout: float = 30.0,
        max_retries: int = 3,
        thinking_finish_words=["<conclude>", "**Final Answer**", "</think>"],
    ):
        self.hosts = hosts
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.thinking_finish_words = thinking_finish_words
        self.model_name = requests.get(
            f"http://{self.hosts[0]}/v1/models",
            headers={"Authorization": "Bearer "},
        ).json()["data"][0]["id"]
        self.judger_name = "compass_verifier_v2"

    async def judge(self, data_item: List[RLDataFlowItem]) -> List[RLJudgerResponseItem]:
        response_future = [self.judge_single(d) for d in data_item]
        judger_responses = await asyncio.gather(*response_future)
        return judger_responses

    async def judge_single(self, data_item: RLDataFlowItem) -> RLJudgerResponseItem:
        # print(f"[Judger]: input {data_item}")
        if data_item.env.rollout.finish_reason not in ["finished", "stop"]:
            return RLJudgerResponseItem(uid=data_item.uid.observation_id, reward={"score": -1})
        question = data_item.data.messages[-1]["content"]
        model_answer = data_item.env.rollout.response.replace("<|im_end|>", "").strip()
        for thinking_finish_word in self.thinking_finish_words:
            if thinking_finish_word in model_answer:
                model_answer = model_answer.split(thinking_finish_word)[-1]

        # only keep last 10 lines
        num_lines = len(model_answer.split("\n"))
        if num_lines > 10:
            model_answer = "\n".join(model_answer.split("\n")[-10:])

        if len(model_answer) > 1000:
            model_answer = model_answer[-1000:]

        label = data_item.data.reward_model["ground_truth"]
        outcome_reward = await self._judge_with_llm(question, model_answer, label)
        # print(f"[Judger]: final reward {final_reward}")
        return RLJudgerResponseItem(uid=data_item.uid.observation_id, reward={"score": outcome_reward})

    async def _judge_with_llm(self, question: str, response: str, label: str):
        headers = {"Content-Type": "application/json"}
        prompt = verify_prompt.format("", "", question=question, llm_response=response, gold_answer=label)
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
                            raise Exception(f"API request failed with status {response.status}: {error_msg}")
                        res_str = response_json["choices"][0]["message"]["content"]
                        if res_str.strip() == "A":
                            return 1
                        else:
                            return -1
                except Exception as e:
                    asyncio.sleep(1)
                    print(f"[Judger]: Error try {i}: {str(e)}")
        raise RuntimeError(f"Cannot connect to judger service for {self.max_retries} times.")

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self.judger_name


class CompassVerifierV2Config(NativeJudgerConfig):
    """Configuration for the CompassVerifierV2 judger."""

    hosts: list
    judger_name: str = "compass_verifier_v2"

    def build_actor(self, pg: PlacementGroup, start_bundle_idx: int) -> List[ray.actor.ActorClass]:
        workers_list = []
        for idx in range(self.num_ray_actors):
            bundle_idx = start_bundle_idx + idx
            pg_options = {"num_cpus": pg.bundle_specs[bundle_idx].get("CPU", 1)}
            worker = (
                ray.remote(CompassVerifierV2)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    **pg_options,
                )
                .remote(hosts=self.hosts)
            )
            workers_list.append(worker)
        return workers_list
