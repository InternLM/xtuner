from dataclasses import dataclass
from enum import Enum
from xtuner.v1.data_proto.rl_data import SampleParams
from transformers import AutoTokenizer, AutoProcessor, PreTrainedTokenizerBase
from transformers.image_processing_utils import ProcessorMixin
from xtuner.v1.ray.rollout.controller import SampleParams
from dataclasses import field


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
    # dataset 输出必须
    message: list
    tokens: list[int]
    
    uid: int
    session_id: int | None = None
    prompt_ids: list[int]
    response: str
    response_ids: list[int] 
    logprobs: list[float] 
    routed_experts: list[int] | None = None
    reward: float | list[float] | list[dict] | None = None
    loss_mask: list[int] | None = None
    state: Status = Status.INIT
    sample_parms: SampleParams | None = None
    tools: list | None = None
    tool_choice: str | None = None


# TODO: 这个对象存在的意义是啥？暂时不用，否则会导致内部循环对象不一致, partial rollout 也不好弄
@dataclass
class Trajectory:
    env: str = 'default'
    rollout_state: RolloutState | list[RolloutState] 


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
