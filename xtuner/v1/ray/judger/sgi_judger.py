"""LLM judge judger for SGI evaluation.

Refactored to follow the async judger pattern used by CompassVerifierV2:
- Async HTTP calls via aiohttp with multi-host load balancing and retries
- RLDataFlowItem / RLJudgerResponseItem data models
- Pydantic Config with .build() pattern
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import json
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import ray
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RLDataFlowItem, RLJudgerResponseItem
from xtuner.v1.ray.judger.native import NativeJudgerConfig


logger = logging.getLogger(__name__)

JudgeResponseParser = Callable[[str], bool]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

QUESTION_QUALITY_PROMPT_EN_COT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly.
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT:
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_judge_response(text: str) -> bool:
    """Default parser for the built-in A/B/C judge protocol."""
    if not text:
        raise ValueError("Judge response is empty")

    s = str(text)
    seg = s.rsplit("Final Judgment", 1)[-1] if "Final Judgment" in s else s

    boxed = re.findall(r"\\boxed\s*\{\s*([A-C])\s*\}", seg)
    if boxed:
        return boxed[-1].upper() == "A"

    if seg.strip() in ("A", "B", "C"):
        return seg.strip() == "A"

    paren = re.findall(r"\(([A-C])\)", seg)
    if paren:
        return paren[-1].upper() == "A"

    hyphen = re.findall(
        r"\b([A-C])\b\s*-\s*(CORRECT|INCORRECT|INCOMPLETE|REPETITIVE|REFUSAL)",
        seg,
        flags=re.IGNORECASE,
    )
    if hyphen:
        return hyphen[-1][0].upper() == "A"

    any_letter = re.findall(r"([A-C])", seg)
    if any_letter:
        return any_letter[-1].upper() == "A"

    raise ValueError(f"Unable to parse judge response with default parser: {text!r}")


# ---------------------------------------------------------------------------
# Main judger class
# ---------------------------------------------------------------------------

class SGIJudger:
    """LLM-based judger that uses a judge model to evaluate answers.

    Follows the same async pattern as CompassVerifierV2:
    - Multi-host load balancing with random selection
    - aiohttp-based async HTTP calls
    - Retry logic on failure
    - RLDataFlowItem input / RLJudgerResponseItem output
    """

    def __init__(
        self,
        hosts: List[str] = [],
        model_name: str = "",
        judger_name: str = "sgi_judger",
        request_timeout: float = 60.0,
        max_retries: int = 3,
        prompt_template: Optional[str] = None,
        response_parser: Optional[JudgeResponseParser] = None,
        thinking_finish_words: List[str] = None,
    ):
        self.hosts = hosts
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.prompt_template = prompt_template or QUESTION_QUALITY_PROMPT_EN_COT
        self.response_parser = response_parser or _parse_judge_response
        self.thinking_finish_words = thinking_finish_words or [
            "<conclude>", "**Final Answer**", "</think>"
        ]
        self.judger_name = judger_name

    # ---- batch entry point (asyncio.gather) ----

    async def judge(self, data_items: List[RLDataFlowItem]) -> List[RLJudgerResponseItem]:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"hello {self.judger_name}")
        """Judge a batch of data items concurrently."""
        response_futures = [self.judge_single(d) for d in data_items]
        return await asyncio.gather(*response_futures)

    # ---- single-item entry point ----

    async def judge_single(self, data_item: RLDataFlowItem) -> RLJudgerResponseItem:
        """Judge a single data item."""
        uid = data_item.uid.observation_id

        # Early exit for unfinished rollouts
        if data_item.env.rollout.finish_reason not in ["finished", "stop"]:
            return RLJudgerResponseItem(uid=uid, reward={"score": -1})

        # Extract question from the last user message
        question = data_item.data.messages[-1]["content"]
        if isinstance(question, list):
            # Multimodal: extract text portion
            question = " ".join(
                item.get("text", "") for item in question
                if isinstance(item, dict) and item.get("type") == "text"
            )

        # Extract and clean model answer
        model_answer = data_item.env.rollout.response.replace("<|im_end|>", "").strip()
        for word in self.thinking_finish_words:
            if word in model_answer:
                model_answer = model_answer.split(word)[-1]

        # Ground truth
        reward_model = data_item.data.reward_model or {}
        ground_truth = reward_model.get("ground_truth", "")

        # Build judge prompt
        prompt = self.prompt_template.format(
            question=question or "",
            gold_answer=ground_truth or "",
            llm_response=model_answer or "",
        )

        # Call judge LLM
        outcome_reward, response_msg = await self._call_judge_llm(prompt)
        return RLJudgerResponseItem(uid=uid, reward={"score": outcome_reward, "judger_response": response_msg})

    # ---- Async LLM call with load balancing & retries ----

    async def _call_judge_llm(self, prompt: str):
        """Call the judge LLM via OpenAI-compatible chat/completions endpoint.

        Uses random host selection and retry logic, matching CompassVerifierV2.
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16384,
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
                        response_msg = json.dumps(response_json["choices"][0]["message"], ensure_ascii=False)
                        try:
                            correct = bool(self.response_parser(res_str or ""))
                        except (ValueError, Exception):
                            return -1, response_msg
                        if correct:
                            return 1, response_msg
                        else:
                            return -1, response_msg
                except Exception as e:
                    await asyncio.sleep(1)
                    print(f"[SGIJudger]: Error try {i}: {str(e)}")
        raise RuntimeError(f"Cannot connect to judger service for {self.max_retries} times.")

    def get_judger_name(self) -> str:
        """Get the name of the judger.

        Returns:
            str: The name of the judger.
        """
        return self.judger_name


# ---------------------------------------------------------------------------
# Pydantic config with .build() pattern
# ---------------------------------------------------------------------------

class SGIJudgerConfig(NativeJudgerConfig):
    """Configuration for the SGI judger."""

    hosts: List[str] = []
    model_name: str = ""
    judger_name: str = "sgi_judger"
    max_retries: int = 3
    thinking_finish_words: List[str] = [
        "<conclude>", "**Final Answer**", "</think>"
    ]

    def build_actor(self, pg: PlacementGroup, start_bundle_idx: int) -> List[ray.actor.ActorClass]:
        """Create and launch Ray actor instances for the SGI judger.

        Args:
            pg: The Ray PlacementGroup used to allocate resources for the actors.
            start_bundle_idx: The starting bundle index in the placement group for actor placement.

        Returns:
            List[ActorClass]: A list of Ray actor handles representing the launched judger workers.
        """
        workers_list = []
        for idx in range(self.num_ray_actors):
            bundle_idx = start_bundle_idx + idx
            pg_options = {"num_cpus": pg.bundle_specs[bundle_idx].get("CPU", 1)}
            worker = (
                ray.remote(SGIJudger)
                .options(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_idx,
                    **pg_options,
                )
                .remote(
                    hosts=self.hosts,
                    model_name=self.model_name,
                    judger_name=self.judger_name,
                    request_timeout=self.request_timeout,
                    max_retries=self.max_retries,
                    thinking_finish_words=self.thinking_finish_words,
                )
            )
            workers_list.append(worker)
        return workers_list
