from dataclasses import dataclass
from typing import List

from cyclopts import Parameter
from pydantic import BaseModel
from ray import ObjectRef
from typing_extensions import Annotated


@dataclass
class ReplayMeta:
    action_id: int
    observation_id: int
    state: str
    version: List[int]

    def update_version(self, version: int):
        """更新版本号 :param version: 新的版本号."""
        if version not in self.version:
            self.version.append(version)
            self.version.sort()


@dataclass
class SampleMeta:
    action_id: int
    observation_id: int
    action_ref: ObjectRef
    observation_ref: ObjectRef
    environment: str

    def update_observation_ref(self, observation_ref: ObjectRef):
        """更新 observation_ref :param observation_ref: 新的 observation_ref."""
        self.observation_ref = observation_ref


class SampleParams(BaseModel):
    n: Annotated[int, Parameter(help="Number of samples to generate.")] = 1
    top_k: Annotated[
        int, Parameter(help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    ] = 50
    top_p: Annotated[float, Parameter(help="The cumulative probability for nucleus sampling.")] = 0.95
    temperature: Annotated[float, Parameter(help="The value used to module the next token probabilities.")] = 0.6
    repetition_penalty: Annotated[float, Parameter(help="The parameter for repetition penalty.")] = 1.0
    presence_penalty: Annotated[float, Parameter(help="The parameter for presence penalty.")] = 0.0
    frequency_penalty: Annotated[float, Parameter(help="The parameter for frequency penalty.")] = 0.0
    min_tokens: Annotated[int, Parameter(help="Minimum number of tokens to generate.")] = 2
    max_tokens: Annotated[int, Parameter(help="Maximum number of tokens to generate.")] = 128
    stops: Annotated[List[str], Parameter(help="List of stop sequences.")] = []
    stop_token_ids: Annotated[List[int], Parameter(help="List of stop token IDs.")] = []
    logprobs: Annotated[int, Parameter(help="Number of log probabilities to return.")] = 0
    skip_special_tokens: Annotated[bool, Parameter(help="Whether to skip special tokens.")] = True
