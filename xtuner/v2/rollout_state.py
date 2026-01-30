from dataclasses import dataclass
from enum import Enum
from xtuner.v1.data_proto.rl_data import SampleParams
from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizerBase
from transformers.image_processing_utils import ProcessorMixin
from xtuner.v1.ray.rollout.controller import SampleParams


class Status(Enum):
    INIT = "init"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    SKIPPED = "skipped"


@dataclass
class RolloutState:
    uid: int
    session_id: int | None = None
    prompt_ids: list[int]
    response: str
    response_ids: list[int] 
    logprobs: list[float] 
    routed_experts: list[int] | None = None
    state: Status = Status.INIT
    sample_parms: SampleParams | None = None
    tools: list | None = None
    tool_choice: str | None = None


@dataclass
class Trajectory:
    uid: str
    env: str
    rollout_state: RolloutState | list[RolloutState] 
    reward: float | list[float] | list[dict]


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


# TODO: 重命名
# 必须要有一个随着 RolloutState 一起流转的类，否则无法满足扩展性
class ProcessorUtilState:
    def __init__(self, hf_checkpoint, sample_params=SampleParams()) -> None:
        # persistent state for the generation process
        self.hf_checkpoint = hf_checkpoint
        self.tokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(hf_checkpoint, trust_remote_code=True)
        self.sample_params = sample_params
