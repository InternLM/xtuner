from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Protocol, cast, runtime_checkable
from cyclopts import Parameter

from xtuner.v1.data_proto import RolloutState


@runtime_checkable
class ComputeMetricProtocol(Protocol):
    def __call__(self, samples: list[RolloutState]) -> dict[str, float]: ...


def default_compute_metric_func(samples: list[RolloutState]) -> dict[str, float]:
    return {"accuracy": sum(s.reward["score"] > 0 for s in samples) / len(samples)}


class Evaluator:
    def __init__(self,
        enable_evaluate: bool,
        enable_initial_evaluate: bool,
        evaluate_step: int,
        compute_metric_func: ComputeMetricProtocol | None = None,
        eval_batch_size: int = 0,
    ):
        self.enable_evaluate = enable_evaluate
        self.enable_initial_evaluate = enable_initial_evaluate
        self.evaluate_step = evaluate_step
        self.compute_metric_func = compute_metric_func or default_compute_metric_func
        self.eval_batch_size = eval_batch_size

    def run(self, samples: list[RolloutState] | list[list[RolloutState]]) -> dict[str, float]:
        # 将 list[list[RolloutState]] 转换为 list[RolloutState]
        if samples and isinstance(samples[0], list):
            flat_samples = [sample for batch in cast(list[list[RolloutState]], samples) for sample in batch]
        else:
            flat_samples = cast(list[RolloutState], samples)
        return self.compute_metric_func(flat_samples)


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    enable_evaluate: Annotated[
        bool,
        Parameter(help="Flag to enable or disable evaluation during training."),
    ] = False
    enable_initial_evaluate: Annotated[
        bool,
        Parameter(help="Flag to enable or disable initial evaluation before training starts."),
    ] = False
    evaluate_step: Annotated[int, Parameter(help="Step interval for evaluation.")] = 1

    eval_sample_ratio: Annotated[
        float,
        Parameter(help="Ratio of samples to evaluate from the generated samples."),
    ] = 0
    eval_sample_num: Annotated[
        int,
        Parameter(help="Number of samples to evaluate from the generated samples."),
    ] = 0

    compute_metric_func: Annotated[
        ComputeMetricProtocol | None,
        Field(exclude=True),
        Parameter(help="An optional metric computation function."),
    ] = None

    def build(self, total_eval_samples: int=0) -> "Evaluator":
        if self.eval_sample_num > 0:
            eval_batch_size = self.eval_sample_num
        else:
            assert total_eval_samples > 0, "Total eval samples must be greater than 0 if eval sample num is not provided"
            if self.eval_sample_ratio > 0:
                eval_batch_size = int(total_eval_samples * self.eval_sample_ratio)
            else:
                eval_batch_size = total_eval_samples

        return Evaluator(
            enable_evaluate=self.enable_evaluate,
            enable_initial_evaluate=self.enable_initial_evaluate,
            evaluate_step=self.evaluate_step,
            compute_metric_func=self.compute_metric_func,
            eval_batch_size=eval_batch_size,
        )