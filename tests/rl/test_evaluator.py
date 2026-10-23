"""Evaluation keeps unscored attempts visible without breaking scored-only
callbacks."""

import ast
import math
from pathlib import Path
from unittest.mock import Mock

import pytest

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.evaluator import Evaluator, EvaluatorConfig, default_compute_metric_func


def make_sample(score, *, status=Status.COMPLETED, source="openai/gsm8k", problem=1):
    return RolloutState(
        rollout_id=problem,
        group_id=problem,
        message=[{"role": "user", "content": f"problem {problem}"}],
        status=status,
        reward=None if score is None else {"score": score},
        reward_model={"id": problem},
        extra_fields={"origin_data_source": source},
    )


def test_custom_callback_receives_only_scored_attempts():
    success = make_sample(1.0)
    incorrect = make_sample(0.0)
    failed = make_sample(None, status=Status.FAILED)
    callback = Mock(side_effect=lambda samples: {"score": sum(s.reward["score"] for s in samples) / len(samples)})
    metrics = Evaluator(compute_metric_func=callback).run([[success, failed], [incorrect]])
    callback.assert_called_once_with([success, incorrect])
    assert metrics["score"] == 0.5
    assert metrics["eval_failed_ratio"] == pytest.approx(1 / 3)
    assert failed.status == Status.FAILED and failed.reward is None


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf, True, "1"])
def test_all_unscored_skips_legacy_callback(score):
    callback = Mock(side_effect=AssertionError("A legacy callback must not receive an empty or unscored batch"))
    assert Evaluator(compute_metric_func=callback).run([make_sample(score)]) == {"eval_failed_ratio": 1.0}
    callback.assert_not_called()


def test_failed_status_does_not_become_scored_from_a_stale_reward():
    callback = Mock()
    sample = make_sample(1.0, status=Status.FAILED)
    assert Evaluator(compute_metric_func=callback).run([sample]) == {"eval_failed_ratio": 1.0}
    callback.assert_not_called()
    assert default_compute_metric_func([sample])["accuracy"] == 0.0


def test_custom_callback_can_opt_in_to_all_attempts():
    samples = [make_sample(None, status=Status.FAILED)]
    callback = Mock(return_value={"attempt_count": 1.0, "eval_failed_ratio": -1.0})
    config = EvaluatorConfig(eval_sample_num=10, compute_metric_func=callback, include_failed_samples=True)
    evaluator = config.build()
    assert evaluator.eval_batch_size == 10
    assert config.model_dump()["include_failed_samples"] is True
    assert evaluator.run(samples) == {"attempt_count": 1.0, "eval_failed_ratio": 1.0}
    callback.assert_called_once_with(samples)


@pytest.mark.parametrize("callback", [None, default_compute_metric_func])
def test_builtin_metrics_keep_failed_attempts_in_pass_at_k(callback):
    samples = [make_sample(None, status=Status.FAILED), make_sample(1.0)]
    metrics = EvaluatorConfig(eval_sample_num=2, compute_metric_func=callback).build().run(samples)
    assert metrics["accuracy"] == 1.0
    assert metrics["eval_failed_ratio"] == 0.5
    assert metrics["pass@1"] == 0.0
    assert metrics["pass@2"] == 1.0
    assert metrics["avg_pass@2"] == 0.5
    assert metrics["eval_attempt_count"] == 2.0


def test_empty_custom_evaluation_does_not_invent_scores():
    callback = Mock()
    assert Evaluator(compute_metric_func=callback).run([]) == {"eval_failed_ratio": 0.0}
    callback.assert_not_called()


MOPD_CONFIGS = [
    "rl_dapo_math_mopd.py",
    "rl_dapo_math_mopd_async.py",
    "rl_qwen3_vl_2b_topk_mopd_train_teacher.py",
    "rl_qwen3_vl_2b_sampled_token_mopd_rollout_teacher.py",
]


@pytest.mark.parametrize("filename", MOPD_CONFIGS)
def test_mopd_callback_tolerates_a_source_with_no_scored_attempts(filename):
    path = Path(__file__).resolve().parents[2] / "recipe/on_policy_distillation/config" / filename
    # Load the actual callback and its source mapping without constructing the
    # recipe's models, dataset loaders or GPU resources at module import time.
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "mopd_compute_metric")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "EVAL_DATA_SOURCE_TAGS" for target in node.targets)
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    evaluator = Evaluator(compute_metric_func=namespace["mopd_compute_metric"])
    failed = make_sample(None, status=Status.FAILED, source="hiyouga/geometry3k")
    metrics = evaluator.run([make_sample(1.0), failed])
    assert metrics["score"] == 1.0
    assert metrics["eval_failed_ratio"] == 0.5
    assert any(key.startswith("gsm8k/") for key in metrics)
    assert not any(key.startswith("geo3k/") for key in metrics)
    assert evaluator.run([failed]) == {"eval_failed_ratio": 1.0}
    with pytest.raises(ValueError, match="Unexpected evaluation data source"):
        evaluator.run([make_sample(1.0, source="wrong-source")])
