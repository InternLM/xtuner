import json
import logging
import math
import re
from typing import Dict, List, Union

import requests
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Refer to https://github.com/InternLM/POLAR/blob/main/src/polar/reward_func.py
class Client:
    def __init__(
        self,
        ip: str,
        port: str = "8000",
        model_name: str = "internlm/POLAR-7B",
        max_length: int = 16384,
        max_response_length: int = 4096,
        response_cut_side: str = "right",
        tokenizer_path: str | None = None,
    ):
        self.url = f"{ip}:{port}"
        self.model_name = model_name

        if tokenizer_path is None:
            tokenizer_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # for final reward token and one <|reward|> token and two '\n' tokens
        self.max_length = max_length - 4
        self.max_response_length = max_response_length
        self.response_cut_side = response_cut_side

    def _encode(self, prompt, reference, output, wrapper="sft"):
        """Construct the input string for the reward model.

        Args:
            prompt: Prompt.
            reference: Reference trajectory.
            output: Candidate trajectory.
            wrapper: The wrapper type. Can be "sft" or "pretrain".
        Returns:
            The constructed input string for RM.
        """
        p = "\n".join([e["content"] for e in prompt]) if isinstance(prompt, list) else prompt
        r1 = "\n".join([e["content"] for e in reference]) if isinstance(reference, list) else reference
        r2 = "\n".join([e["content"] for e in output]) if isinstance(output, list) else output

        p_ids = self.tokenizer.encode(p, add_special_tokens=True)
        r1_ids = self.tokenizer.encode(r1, add_special_tokens=True)
        r2_ids = self.tokenizer.encode(r2, add_special_tokens=True)

        if len(r1_ids) > self.max_response_length:
            print(
                f"Reference sequence length {len(r1_ids)} is "
                f"larger than max_response_length {self.max_response_length}",
            )
            if self.response_cut_side == "right":
                r1_ids = r1_ids[: self.max_response_length]
            else:
                r1_ids = r1_ids[-self.max_response_length :]

        if len(r2_ids) > self.max_response_length:
            print(
                f"Output sequence length {len(r2_ids)} is "
                f"larger than max_response_length {self.max_response_length}",
            )
            if self.response_cut_side == "right":
                r2_ids = r2_ids[: self.max_response_length]
            else:
                r2_ids = r2_ids[-self.max_response_length :]

        max_prompt_length = (self.max_length - len(r1_ids) - len(r2_ids)) // 2

        if len(p_ids) > max_prompt_length:
            print(
                f"Prompt sequence length {len(p_ids)} is " f"larger than max_prompt_length {max_prompt_length}",
            )
            p_ids = p_ids[-max_prompt_length:]

        p = self.tokenizer.decode(p_ids, skip_special_tokens=True)
        r1 = self.tokenizer.decode(r1_ids, skip_special_tokens=True)
        r2 = self.tokenizer.decode(r2_ids, skip_special_tokens=True)

        # Fit the template of RM
        _reference_cat = p + r1 if wrapper == "pretrain" or len(r1) == "" else p + "\n" + r1
        _output_cat = p + r2 if wrapper == "pretrain" or len(r2) == "" else p + "\n" + r2

        final_txt = _reference_cat + "<|reward|>" + _output_cat + "[UNUSED_TOKEN_130]"

        return final_txt

    def encode(self, data) -> Union[str, List[str]]:
        """Encode the input data into a format suitable for RM.

        Args:
            data: A dictionary or a list of dictionary containing the keys
                  'prompt', 'reference', 'output', and optionally 'wrapper'.
        Returns:
            The encoded input string for RM.
        """
        if isinstance(data, dict):
            return self._encode(**data)
        elif isinstance(data, list):
            return [self._encode(**item) if isinstance(item, dict) else item for item in data]
        else:
            raise ValueError("Input data must be a dictionary or a list of dictionaries.")

    def sglang_request_reward(self, data, max_retries=8):
        for _ in range(max_retries):
            try:
                res = requests.post(
                    f"http://{self.url}/classify",
                    json={
                        "model": self.model_name,
                        "text": data,
                    },
                    proxies={"http": None, "https": None},
                    timeout=540,
                )
                rewards = [e["embedding"][0] for e in res.json()]
                return rewards
            except Exception as e:
                print(f"Error requesting reward: {e}")
                print(f"Raw response: {data}")
                continue

        print(f"Failed to request reward after {max_retries} retries")
        return None

    def vllm_request_reward(self, data, max_retries=8):
        for _ in range(max_retries):
            try:
                res = requests.post(
                    f"http://{self.url}/classify",
                    json={
                        "model": self.model_name,
                        "text": data,
                    },
                    proxies={"http": None, "https": None},  # Explicitly disable proxy
                    timeout=540,  # Add timeout
                )
                rewards = [e["embedding"][0] for e in res.json()]
                return rewards
            except Exception as e:
                print(f"Error requesting reward: {e}")
                print(f"Raw response: {data}")
                continue
        print(f"Failed to request reward after {max_retries} retries")
        return None

    def __call__(self, data) -> List[float]:
        """Call the input wrapper to construct the input string for RM.

        Args:
            data: A list of dictionaries containing the keys
                  'prompt', 'reference', 'output', and optionally 'wrapper'.
                  E.g.
                    data = [
                        {
                            "prompt": [{"role": "user", "content": "What is the capital of China?"}],
                            "reference": [{"role": "assistant", "content": "Beijing."}],
                            "output": [{"role": "assistant", "content": "Beijing."}]
                        },
                        {
                            "prompt": [{"role": "user", "content": "What is the capital of China?"}],
                            "reference": [{"role": "assistant", "content": "Beijing."}],
                            "output": [{"role": "assistant", "content": "Shanghai."}]
                        }
                    ]
        Returns:
            scores: The list of reward scores returned by the RM server.
                    If the request fails, it returns None.
        """
        data = self.encode(data)
        # scores = self.vllm_request_reward(data)
        scores = self.sglang_request_reward(data)
        return scores


# _IP = "10.102.209.13"
# _PORT = "24110"
# _MODEL_NAME = "/mnt/shared-storage-user/agent4review-share/xuerui/models/POLAR-7B"
# client = Client(
#     ip=_IP,
#     port=_PORT,
#     model_name=_MODEL_NAME,
#     max_length=131072,
#     max_response_length=8192,
#     # tokenizer_path='/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/POLAR-7B',
# )

import random

_MODEL_NAME = "/mnt/shared-storage-user/agent4review-share/xuerui/models/POLAR-7B"
CLIENT_POOL = []


def init_clients():
    global CLIENT_POOL
    SERVER_POOL = [
        {"ip": "10.102.209.13", "port": "24110"},
        {"ip": "10.102.209.13", "port": "26426"},
        {"ip": "10.102.209.13", "port": "26427"},
        {"ip": "10.102.208.16", "port": "26428"},
        {"ip": "10.102.208.16", "port": "26429"},
    ]
    for s in SERVER_POOL:
        CLIENT_POOL.append(
            Client(
                ip=s["ip"],
                port=s["port"],
                model_name=_MODEL_NAME,
                max_length=131072,
                max_response_length=8192,
                # tokenizer_path='/mnt/shared-storage-user/llmit1/user/wangziyi/exp/mindcopilot_rl/work_dirs/ckpt/POLAR-7B',
            )
        )


def get_client():
    return random.choice(CLIENT_POOL)


init_clients()

REVIEWER_PROMPT = (
    "You are an expert reviewer in the field of {domain} for the {ven_id} conference. "
    "Your responsibilitie is conducting an initial review. "
    "Analyze the paper, then output the final review inside `<reviewer>` and `</reviewer>` tags. "
    "The review must include: Summary, Strengths, Weaknesses, Questions, and References. "
    "### CITATION & REFERENCE STANDARDS (CRITICAL)\n"
    "You must adhere to the following strict formatting rules for the final review:\n"
    "1. **Sequential Numbering**: Citations in the text must be numbered sequentially starting from [1] based on the order they first appear (e.g., [1], [2], [3]).\n"
    "2. **Inline Citation**: Every external claim must have an inline citation. Example: 'Recent studies [1] have shown that...'\n"
    "3. **Reference List**: The 'References' section at the end must strictly match the inline citations. Each entry must contain:\n"
    "   - Format: `[ID] Authors. Title. URL`\n"
    "   - Example: `[1] J. Smith. Deep Learning. https://arxiv.org/abs/...`\n"
    "   - Ensure the Authors, Title, and URL are complete. Do not use placeholders like '...'.\n\n"
    "### OUTPUT FORMAT (`<reviewer>`)\n"
    "<reviewer>\n"
    "## Summary\n"
    "...\n"
    "## Strengths\n"
    "...\n"
    "## Weaknesses\n"
    "...\n"
    "## Questions\n"
    "...\n"
    "## References\n"
    "[1] ...\n"
    "[2] ...\n"
    "</reviewer>\n\n"
    "Blow is the paper to be reviewed\n{paper}\n\n"
)


def extract_review(traj: str):
    reviews = re.findall(
        r"<reviewer>(.*?)</reviewer>",
        traj,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not reviews:
        return ""

    return reviews[-1].strip()


def extract_search_info(traj: str):
    valid_sources = []
    tool_responses = re.findall(
        r"<tool_response>(.*?)</tool_response>",
        traj,
        flags=re.DOTALL,
    )

    for sanitized_text in tool_responses:
        chunks = re.split(r"SOURCE \d+: ", sanitized_text)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            try:
                data = json.loads(chunk)
                source_info = {
                    "title": data["title"].strip(),
                    "id": data["id"].strip(),
                    "authors": data["authors"].strip(),
                }
                if source_info not in valid_sources:
                    valid_sources.append(source_info)
            except:
                pass

    return valid_sources


def extract_tool_responses_from_agent_trace(agent_trace):
    tool_texts = []
    if not isinstance(agent_trace, list):
        return tool_texts

    for record in agent_trace:
        if not isinstance(record, dict):
            continue
        sender = str(record.get("sender", "")).lower()
        content = record.get("content")

        # Main path from extra_info.log:
        # sender == "EnvAgent", content is ActionReturn list.
        if sender == "envagent" and isinstance(content, list):
            for action_ret in content:
                if not isinstance(action_ret, dict):
                    continue
                results = action_ret.get("result")
                if not isinstance(results, list):
                    continue
                for res in results:
                    if not isinstance(res, dict):
                        continue
                    if res.get("type") != "text":
                        continue
                    text = res.get("content", "")
                    if isinstance(text, str) and text.strip():
                        tool_texts.append(text)

    return tool_texts


def build_search_trace(solution_str: str, extra_info: dict = None):
    fragments = []
    if isinstance(extra_info, dict):
        # print("src/reward_fn.py build_search_trace extra_info agent_trace:", extra_info.get("agent_trace"))
        agent_trace = extra_info.get("agent_trace")
        tool_texts = extract_tool_responses_from_agent_trace(agent_trace)
        # print("src/reward_fn.py build_search_trace extra_info agent_trace:", tool_texts)
        for text in tool_texts:
            fragments.append(f"<tool_response>{text}</tool_response>")

    # Final fallback to keep compatibility when trace is unavailable.
    if not fragments and isinstance(solution_str, str) and solution_str:
        fragments.append(solution_str)

    return "\n".join(fragments)


def reward_format_review(review_content: str):
    required_sections = [
        "## Summary",
        "## Strengths",
        "## Weaknesses",
        "## Questions",
        "## References",
    ]
    missing_sections = [s for s in required_sections if s not in review_content]
    section_score = 1.0 - (len(missing_sections) / len(required_sections))

    clean_text = review_content
    for section in required_sections:
        clean_text = clean_text.replace(section, "")

    char_count = len(clean_text.strip())

    if char_count < 2000 or char_count > 10000:
        length_reward = -0.5
    else:
        length_reward = 0.5

    return section_score + length_reward


def reward_inline_citations(content: str):
    if "## References" not in content:
        return -0.5

    parts = content.split("## References")
    body_text = parts[0]
    ref_section = parts[1] if len(parts) > 1 else ""

    body_citations = re.findall(r"\[(\d+)\]", body_text)
    ref_list_ids = re.findall(r"^\s*\[(\d+)\]", ref_section, re.MULTILINE)

    body_set = set(body_citations)
    ref_set = set(ref_list_ids)

    if not body_set and not ref_set:
        return -1.0

    if len(body_set) == len(ref_set):
        score = 1.0
    else:
        score = -1.0

    penalties = 0.0

    if len(ref_set) < 2:
        penalties += 0.2
    if len(ref_set) > 5:
        penalties += 0.1 * (len(ref_set) - 5)

    if len(body_set) > 0:
        density_ratio = len(body_citations) / len(body_set)
        if density_ratio > 3.5:
            penalties += 0.6

    try:
        numeric_refs = sorted([int(x) for x in ref_list_ids])
        if numeric_refs:
            expected = list(range(1, len(numeric_refs) + 1))
            if numeric_refs != expected:
                penalties += 0.3
    except:
        penalties += 0.3

    # Duplication Penalty in Reference List
    if len(ref_list_ids) != len(ref_set):
        penalties += 0.6

    score = score - penalties
    return score


def reward_inline_citations_xtuner(content: str):
    if "## References" not in content:
        return 0.0  # 改动：原 -0.5 → 0.0，作为最低分

    parts = content.split("## References")
    body_text = parts[0]
    ref_section = parts[1] if len(parts) > 1 else ""

    body_citations = re.findall(r"\[(\d+)\]", body_text)
    ref_list_ids = re.findall(r"^\s*\[(\d+)\]", ref_section, re.MULTILINE)

    body_set = set(body_citations)
    ref_set = set(ref_list_ids)

    if not body_set and not ref_set:
        return 0.0  # 改动：原 -1.0 → 0.0

    score = 1.0

    # 正文引用与参考列表数量不一致（较严重）
    if len(body_set) != len(ref_set):
        score *= 0.25

    # 引用数量过少
    if len(ref_set) < 2:
        score *= 0.8

    # 引用数量过多（线性衰减，但有下限）
    if len(ref_set) > 5:
        excess = len(ref_set) - 5
        score *= max(0.5, 1.0 - 0.05 * excess)

    # 正文过度重复引用同一条
    if len(body_set) > 0:
        density_ratio = len(body_citations) / len(body_set)
        if density_ratio > 3.5:
            score *= 0.7

    # 编号不连续
    try:
        numeric_refs = sorted([int(x) for x in ref_list_ids])
        if numeric_refs:
            expected = list(range(1, len(numeric_refs) + 1))
            if numeric_refs != expected:
                score *= 0.25
    except:
        score *= 0.85

    # 引用列表有重复条目（较严重）
    if len(ref_list_ids) != len(ref_set):
        score *= 0.25

    return score


def reward_hallucination(
    content: str,
    search_info: List[Dict[str, str]],
):
    if "## References" not in content:
        return -0.4

    ref_section = content.split("## References")[-1].strip()
    ref_lines = [line.strip() for line in ref_section.split("\n") if line.strip()]

    if not ref_lines:
        return -0.5

    for line in ref_lines:
        line_lower = line.lower()

        found_match_for_this_line = False

        for paper in search_info:
            has_title = paper["title"].lower() in line_lower
            has_url = paper["id"].split("arxiv: ")[-1] in line
            # TODO: has_author

            if has_title and has_url:  # and has_author:
                found_match_for_this_line = True
                break

        if not found_match_for_this_line:
            return -1.0

    return 1.0


def compute_review_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    # print("#" * 80, "solution_str:", solution_str, "extra_info['num_turns']:", extra_info["num_turns"], "#" * 80)
    # print("#"*80, "len of solution_str:", len(solution_str), "#"*80)
    review_content = extract_review(solution_str)
    # print("#" * 80, "len of review_content:", len(review_content), "#" * 80)
    data = {
        "prompt": [
            {
                "role": "user",
                "content": REVIEWER_PROMPT.format(
                    domain=extra_info["domain"],
                    ven_id=extra_info["ven_id"],
                    paper=extra_info["paper"],
                ),
            },
        ],
        "reference": [
            {
                "role": "assistant",
                "content": "<reviewer>\n" + ground_truth + "\n</reviewer>",
            },
        ],
        "output": [
            {
                "role": "assistant",
                "content": "<reviewer>\n" + review_content.split("## References")[0] + "\n</reviewer>",
            },
        ],
        "wrapper": "sft",
    }

    client = get_client()
    # text qulity
    polar_score = math.tanh(client([data])[0] / 10) * 2 + 2  # math.tanh(client([data])[0] / 10) * 2
    format_score = reward_format_review(review_content) + 0.5  # reward_format_review(review_content)
    # tool call
    if extra_info["num_turns"] == 2:
        tool_score = -1.0  # 0.0 (会退化为非工具调用) # -1.0
    # elif 4 <= extra_info["num_turns"] <= 6:
    elif 8 <= extra_info["num_turns"]:
        tool_score = 2.0  # 1.0
    elif extra_info["num_turns"] == 6:
        tool_score = 1.0  # 1.0
    else:
        tool_score = 0.4  # -0.6
    # citation
    citations_score = reward_inline_citations_xtuner(review_content)  # reward_inline_citations(review_content)
    search_trace = build_search_trace(solution_str, extra_info)
    search_info = extract_search_info(search_trace)
    hallucination_score = (
        reward_hallucination(review_content, search_info) + 1
    )  # reward_hallucination(review_content, search_info)
    reviewer_score = polar_score + format_score + tool_score + citations_score + hallucination_score
    # return reviewer_score
    print(
        "compute_review_score compute_review_score:",
        {
            'score': reviewer_score,
            'polar_score': polar_score,
            "format_score": format_score,
            'tool_score': tool_score,
            "citations_score": citations_score,
            "hallucination_score": hallucination_score,
        },
    )
    return {
        'score': reviewer_score,
        'polar_score': polar_score,
        "format_score": format_score,
        'tool_score': tool_score,
        "citations_score": citations_score,
        "hallucination_score": hallucination_score,
    }


# import sys
# sys.path.insert(0, "/mnt/shared-storage-user/agent4review-share/xuerui/agentreview_code/260210_MoE_Review/xtuner")
from typing import Callable, List, Literal, TypeAlias, cast

from xtuner.v1.data_proto.rl_data import RLDataFlowItem
from xtuner.v1.ray.judger.native import NativeJudgerConfig


def review_preprocess(data_item: List[RLDataFlowItem], extra_info: dict):
    """Custom preprocess function that extracts extra_info from data item."""
    assert len(data_item) == 1, "Review preprocess only supports single data item."
    response = data_item[0].env.rollout.response
    assert data_item[0].data.reward_model is not None
    label = data_item[0].data.reward_model["ground_truth"]
    # print("src/reward_fn.py review_preprocess data_item[0].env.rollout.extra_info:", [data_item[0].env.rollout.extra_info])

    # Extract domain, ven_id, paper from data.extra_info (not reward_model!)
    item_extra_info = {}
    if data_item[0].data.extra_info:
        for key in ['domain', 'ven_id', 'paper']:
            if key in data_item[0].data.extra_info:
                item_extra_info[key] = data_item[0].data.extra_info[key]
    # Judger path (before postprocess_func): trace is attached into rollout.extra_info.
    rollout_extra_info = data_item[0].env.rollout.extra_info or {}
    if "agent_trace" in rollout_extra_info:
        item_extra_info["agent_trace"] = rollout_extra_info["agent_trace"]

    # Prefer num_turns explicitly attached to rollout extra_info by agent env/judger wrapper.
    if data_item[0].env.rollout.extra_info and "num_turns" in data_item[0].env.rollout.extra_info:
        item_extra_info["num_turns"] = int(data_item[0].env.rollout.extra_info["num_turns"])
    else:
        # Default to 2 for single-turn completion.
        item_extra_info["num_turns"] = 2

    return {
        "response": response,
        "label": label,
        "extra_info": item_extra_info,
    }


def review_reward_handler(response, label, extra_info):
    # NativeJudger 调用时的签名适配
    return compute_review_score(
        data_source=extra_info.get("data_source", ""),
        solution_str=response,
        ground_truth=label,
        extra_info=extra_info,
    )


class ReviewJudgerConfig(NativeJudgerConfig):
    judger_name: str = "openreview"
    num_ray_actors: int = 1
    reward_func: Callable = review_reward_handler
    preprocess_func: Callable = review_preprocess


AUTHOR_PROMPT = (
    "You are the corresponding author of the paper submitted to the {ven_id} conference. "
    "The paper you submitted is:\n{paper}\n\n"
    "Your paper has been reviewed and you have received critical feedback. "
    "Your goal is to write a persuasive, professional, and evidence-based rebuttal that addresses each reviewer concern.\n\n"
    "Produce the final rebuttal enclosed in <rebuttal> and </rebuttal> tags. "
    "The final rebuttal must be a point-by-point response that is polite, concise, and evidence-based. For each reviewer point include: "
    "(a) the reviewer's comment (short), (b) your response, (c) any actions or changes made to the manuscript.\n"
    "### CITATION & REFERENCE STANDARDS (CRITICAL)\n"
    "You MUST strictly follow these citation rules in the final <rebuttal> output:\n"
    "1. Sequential Numbering: References must be numbered sequentially starting from [1] in the order they are first cited in the rebuttal text.\n"
    "2. Inline Citation: Every external claim or evidence must have an inline citation using the bracketed number form. "
    "Example: 'Recent work [1] demonstrates...'\n"
    "3. Reference List: The 'References' section must exactly match the inline citation numbers. "
    "Each reference entry must follow this exact format: [N] Authors. Title. URL\n"
    "EXPERIMENT-REQUEST HANDLING (choose one per reviewer experiment request):\n"
    "A) Argue experiments are unnecessary — give principled reasons. Optionally propose lightweight analyses (no fabricated results).\n\n"
    "B) Acknowledge additional experiments are needed but not run. Never fabricate experiments, data, or citations. "
    "Keep proposed experiments minimal and prioritized.\n\n"
    "FINAL <rebuttal> FORMAT:\n"
    "<rebuttal>\n"
    "## Responses\n"
    "1. Reviewer comment: ...\nResponse: ... (use inline citations [N], include changes to be made)\n\n"
    "2. Reviewer comment: ...\nResponse: ... (use inline citations [N], include changes to be made)\n\n"
    "## References\n"
    "[1] ...\n"
    "[2] ...\n"
    "</rebuttal>\n"
    "Blow is the review content\n{review}\n\n"
)


def extract_rebuttal(traj: str):
    rebuttals = re.findall(
        r"<rebuttal>(.*?)</rebuttal>",
        traj,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not rebuttals:
        rebuttal = ""
    else:
        rebuttal = rebuttals[-1].strip()

    return rebuttal


def reward_format_rebuttal(rebuttal_content: str):
    required_sections = [
        "## Responses",
        "## References",
    ]
    missing_sections = [s for s in required_sections if s not in rebuttal_content]
    section_score = 1.0 - (len(missing_sections) / len(required_sections))

    clean_text = rebuttal_content
    for section in required_sections:
        clean_text = clean_text.replace(section, "")

    char_count = len(clean_text.strip())

    if char_count < 2000 or char_count > 10000:
        length_reward = -0.5
    else:
        length_reward = 0.5

    return section_score + length_reward


def compute_rebuttal_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
):
    rebuttal_content = extract_rebuttal(solution_str)
    data = {
        "prompt": [
            {
                "role": "user",
                "content": AUTHOR_PROMPT.format(
                    ven_id=extra_info["ven_id"],
                    paper=extra_info["paper"],
                    review=extra_info["review"],
                ),
            },
        ],
        "reference": [
            {
                "role": "assistant",
                "content": "<rebuttal>\n" + ground_truth + "\n</rebuttal>",
            },
        ],
        "output": [
            {
                "role": "assistant",
                "content": "<rebuttal>\n" + rebuttal_content.split("## References")[0] + "\n</rebuttal>",
            },
        ],
        "wrapper": "sft",
    }

    client = get_client()
    # text qulity
    polar_score = math.tanh(client([data])[0] / 10) * 2
    format_score = reward_format_rebuttal(rebuttal_content)
    # tool call
    if extra_info["num_turns"] == 2:
        tool_score = -1.0
    elif 4 <= extra_info["num_turns"] <= 6:
        tool_score = 1.0
    else:
        tool_score = -0.6
    # citation
    citations_score = reward_inline_citations(rebuttal_content)
    search_trace = build_search_trace(solution_str, extra_info)
    search_info = extract_search_info(search_trace)
    hallucination_score = reward_hallucination(rebuttal_content, search_info)
    rebuttal_score = polar_score + format_score + tool_score + citations_score + hallucination_score
    # return rebuttal_score
    return {
        'score': rebuttal_score,
        'polar_score': polar_score,
        "format_score": format_score,
        'tool_score': tool_score,
        "citations_score": citations_score,
        "hallucination_score": hallucination_score,
    }
