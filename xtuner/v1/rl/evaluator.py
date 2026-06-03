import json
from collections.abc import Mapping
from typing import Annotated, Protocol, cast, runtime_checkable

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import RolloutState


@runtime_checkable
class ComputeMetricProtocol(Protocol):
    def __call__(self, samples: list[RolloutState]) -> dict[str, float]: ...


def default_compute_metric_func(samples: list[RolloutState]) -> dict[str, float]:
    if not samples:
        return {"accuracy": 0.0}

    positives = []
    for s in samples:
        positives.append(1 if _reward_score(s) > 0 else 0)

    metrics = {"accuracy": sum(positives) / len(positives)}
    metrics.update(_compute_grouped_pass_metrics(samples, positives))
    return metrics


def _reward_score(sample: RolloutState) -> float:
    reward = sample.reward
    assert isinstance(reward, Mapping)
    return float(reward["score"])


def _eval_group_key(sample: RolloutState) -> str:
    rollout_item = sample.extra_fields.get("rollout_item")
    item_id = getattr(rollout_item, "id", None)
    if item_id is not None:
        return str(item_id)

    reward_model = sample.reward_model if isinstance(sample.reward_model, Mapping) else {}
    for key in ("id", "task_id", "problem_idx", "case_id"):
        if reward_model.get(key) is not None:
            return str(reward_model[key])

    return json.dumps(sample.message, ensure_ascii=False, sort_keys=True, default=str)


def _compute_grouped_pass_metrics(samples: list[RolloutState], positives: list[int]) -> dict[str, float]:
    source_groups: dict[str, dict[str, list[int]]] = {}
    for sample, positive in zip(samples, positives):
        source = _data_source_key(sample)
        source_groups.setdefault(source, {}).setdefault(_eval_group_key(sample), []).append(positive)

    if all(len(groups) == sum(len(group) for group in groups.values()) for groups in source_groups.values()):
        return {}

    metrics = {}
    use_source_prefix = len(source_groups) > 1
    for source, groups in source_groups.items():
        prefix = f"{source}/" if use_source_prefix else ""
        group_count = len(groups)
        attempt_count = sum(len(group) for group in groups.values())
        inferred_k = max(1, round(attempt_count / group_count))
        metrics[f"{prefix}eval_group_count"] = float(group_count)
        metrics[f"{prefix}eval_attempt_count"] = float(attempt_count)
        metrics[f"{prefix}eval_inferred_k"] = float(inferred_k)
        for k in (1, 2, 4, 8, 16, 32):
            if k > inferred_k:
                continue
            eligible = [group for group in groups.values() if len(group) >= k]
            if not eligible:
                continue
            metrics[f"{prefix}pass@{k}"] = sum(1 if any(group[:k]) else 0 for group in eligible) / len(eligible)
            if k == inferred_k:
                metrics[f"{prefix}avg_pass@{k}"] = sum(sum(group[:k]) / k for group in eligible) / len(eligible)
    return metrics


def _data_source_key(sample: RolloutState) -> str:
    rollout_item = sample.extra_fields.get("rollout_item")
    item_data_source = getattr(rollout_item, "data_source", None)
    if item_data_source is not None:
        return str(item_data_source)

    data_source = sample.data_source
    if isinstance(data_source, str):
        return data_source
    if isinstance(data_source, Mapping) and data_source:
        source, _ = max(data_source.items(), key=lambda item: float(item[1]))
        return str(source)
    return "unknown"


class Evaluator:
    def __init__(
        self,
        compute_metric_func: ComputeMetricProtocol | None = None,
        eval_batch_size: int = 0,
    ):
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
    """Configuration for rollout evaluation.

    ``EvaluatorConfig`` controls how many generated samples are selected for
    evaluation and which metric function is used to summarize them. It is used
    by RL trainers when evaluation is enabled.

    Args:
        eval_sample_ratio (float): Ratio of generated samples to evaluate when
            ``eval_sample_num`` is not set. Defaults to 0.
        eval_sample_num (int): Fixed number of samples to evaluate. A positive
            value takes precedence over ``eval_sample_ratio``. Defaults to 0.
        compute_metric_func (ComputeMetricProtocol | None): Optional function
            that receives evaluated rollout states and returns metrics. Defaults
            to None.

    **Examples:**

    Example evaluator using a fixed sample count::

        config = EvaluatorConfig(
            eval_sample_num=128,
            compute_metric_func=compute_metrics,
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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

    def build(self, total_eval_samples: int = 0) -> "Evaluator":
        if self.eval_sample_num > 0:
            eval_batch_size = self.eval_sample_num
        else:
            assert total_eval_samples > 0, (
                "Total eval samples must be greater than 0 if eval sample num is not provided"
            )
            if self.eval_sample_ratio > 0:
                eval_batch_size = int(total_eval_samples * self.eval_sample_ratio)
            else:
                eval_batch_size = total_eval_samples

        return Evaluator(
            compute_metric_func=self.compute_metric_func,
            eval_batch_size=eval_batch_size,
        )
