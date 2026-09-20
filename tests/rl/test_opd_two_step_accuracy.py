"""Two-step OPD smoke-run accuracy checks against golden step-1/2 metrics.

The smoke launchers (recipe/on_policy_distillation/scripts/run_smoke_*.sh)
keep the e2e models, batch size 128, seed 1234, and 1024/2048 sequence
lengths, but stop after ``TOTAL_TRAIN_STEPS=2`` with eval disabled. Data
order is seed-deterministic, so the smoke run's step-1/2 metrics must
reproduce the first two steps of the reference 50-step clusterx runs within
per-tag tolerances:

- topk: work_dirs/opd_e2e/qwen3-vl-2b-train-teacher-topk-clusterx-20260917-063709
- mopd: work_dirs/opd_e2e/qwen3-vl-2b-mopd-clusterx-20260916-120028

Workflow:

    bash recipe/on_policy_distillation/scripts/run_smoke_train_teacher_topk.sh
    OPD_SMOKE_RUN_DIR=<smoke WORK_DIR> OPD_SMOKE_EXPERIMENT=topk \
        pytest tests/rl/test_opd_two_step_accuracy.py -k smoke_run
"""

import json
import os
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

TOPK_GOLDEN = {
    "metrics": {
        "distillation/reduced_topk_opd_kl": {1: 0.30486780405044556, 2: 0.3286096453666687},
        "distillation/reduced_topk_opd_loss": {1: 0.30486780405044556, 2: 0.3286095857620239},
        "distillation/reduced_topk_opd_overlap_fraction": {1: 0.6676791906356812, 2: 0.6546056866645813},
        "distillation/reduced_topk_opd_student_selected_mass": {1: 0.9960947632789612, 2: 0.9966292381286621},
        "distillation/reduced_topk_opd_teacher_selected_mass": {1: 0.9986172318458557, 2: 0.999037504196167},
        "response/rewards/mean": {1: 0.3828125298023224, 2: 0.36328125},
    },
    "invariants": {"num_train_steps": 2, "response/batch_size": 128},
    "tolerance": {
        "default": {"rtol": 0.2, "atol": 0.0},
        "response/rewards/mean": {"rtol": 0.0, "atol": 0.15},
    },
}

MOPD_GOLDEN = {
    "metrics": {
        "distillation/reduced_distillation_kl": {1: 0.34421640634536743, 2: 0.3924480080604553},
        "distillation/reduced_distillation_abs_loss": {1: 0.4967445433139801, 2: 0.5339475274085999},
        "response/rewards/mean": {1: 0.4476562440395355, 2: 0.4703125059604645},
    },
    "invariants": {"num_train_steps": 2, "response/batch_size": 128},
    "tolerance": {
        "default": {"rtol": 0.2, "atol": 0.0},
        "response/rewards/mean": {"rtol": 0.0, "atol": 0.15},
    },
}

GOLDEN_BY_EXPERIMENT = {"topk": TOPK_GOLDEN, "mopd": MOPD_GOLDEN}


def find_exp_tracking(run_dir: Path) -> Path:
    """Locate the ``<run_ts>/logs/exp_tracking`` directory inside a run dir.

    Args:
        run_dir (Path): WORK_DIR of a finished RL run, which contains one or
            more ``<timestamp>`` subdirectories written by the trainer.

    Returns:
        Path: The most recent ``exp_tracking`` directory.

    Raises:
        FileNotFoundError: If no ``exp_tracking`` directory exists.
    """
    candidates = sorted(run_dir.glob("*/logs/exp_tracking"))
    if not candidates:
        raise FileNotFoundError(f"No */logs/exp_tracking found under {run_dir}")
    return candidates[-1]


def read_scalars(log_dir: Path) -> dict[str, dict[int, float]]:
    """Read all scalar series from a TensorBoard event directory.

    Args:
        log_dir (Path): Directory containing ``events.out.tfevents`` files.

    Returns:
        dict[str, dict[int, float]]: Mapping of scalar tag to step-indexed
        values.
    """
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: {event.step: float(event.value) for event in accumulator.Scalars(tag)}
        for tag in accumulator.Tags()["scalars"]
    }


def compare_metrics(
    golden: dict,
    scalars: dict[str, dict[int, float]],
    steps: list[int],
) -> list[str]:
    """Check golden metrics against actually-read scalar series.

    Args:
        golden (dict): Golden baseline (``metrics`` + ``tolerance``).
        scalars (dict[str, dict[int, float]]): Output of :func:`read_scalars`.
        steps (list[int]): Training steps that must be present and compared.

    Returns:
        list[str]: Human-readable failure descriptions; empty when all
        metrics match within tolerance.
    """
    failures: list[str] = []
    tolerance = golden["tolerance"]
    default_tol = tolerance.get("default", {"rtol": 0.0, "atol": 0.0})
    for tag, expected in golden["metrics"].items():
        series = scalars.get(tag)
        if series is None:
            failures.append(f"missing scalar tag: {tag}")
            continue
        tol = tolerance.get(tag, default_tol)
        for step in steps:
            if step not in series:
                failures.append(f"missing {tag} at step {step}")
                continue
            expected_value = expected[step]
            actual_value = series[step]
            bound = max(tol["atol"], tol["rtol"] * abs(expected_value))
            if abs(actual_value - expected_value) > bound:
                failures.append(
                    f"{tag} step {step}: actual {actual_value:.6f} outside golden "
                    f"{expected_value:.6f} +/- {bound:.6f}"
                )
    return failures


def check_invariants(golden: dict, scalars: dict[str, dict[int, float]]) -> list[str]:
    """Check structural invariants (step count, batch size) of the smoke run.

    Args:
        golden (dict): Golden baseline (``invariants`` section).
        scalars (dict[str, dict[int, float]]): Output of :func:`read_scalars`.

    Returns:
        list[str]: Human-readable failure descriptions; empty when all
        invariants hold.
    """
    failures: list[str] = []
    invariants = golden.get("invariants", {})
    num_steps = invariants.get("num_train_steps")
    batch_series = scalars.get("response/batch_size", {})
    if num_steps is not None and sorted(batch_series) != list(range(1, num_steps + 1)):
        failures.append(
            f"expected response/batch_size at steps 1..{num_steps}, got steps {sorted(batch_series)}"
        )
    batch_size = invariants.get("response/batch_size")
    if batch_size is not None:
        for step, value in batch_series.items():
            if int(value) != batch_size:
                failures.append(f"response/batch_size at step {step}: {int(value)} != {batch_size}")
    return failures


def check_run(run_dir: Path | str, golden: dict) -> list[str]:
    """Validate a finished smoke run against a golden baseline.

    Args:
        run_dir (Path | str): WORK_DIR of the smoke run.
        golden (dict): Golden baseline (``TOPK_GOLDEN`` or ``MOPD_GOLDEN``).

    Returns:
        list[str]: All failures (metrics + invariants); empty when the run
        matches the golden baseline.
    """
    metric_steps = sorted(
        {int(step) for expected in golden["metrics"].values() for step in expected}
    )
    scalars = read_scalars(find_exp_tracking(Path(run_dir)))
    return compare_metrics(golden, scalars, metric_steps) + check_invariants(golden, scalars)


def _write_scalars(log_dir: Path, series: dict[str, dict[int, float]]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    for tag, steps in series.items():
        for step, value in steps.items():
            writer.add_scalar(tag, value, global_step=step)
    writer.flush()
    writer.close()


def _fixture_golden() -> dict:
    return {
        "metrics": {
            "distillation/reduced_distillation_kl": {1: 0.34, 2: 0.39},
            "response/rewards/mean": {1: 0.45, 2: 0.47},
        },
        "invariants": {"num_train_steps": 2, "response/batch_size": 128},
        "tolerance": {
            "default": {"rtol": 0.2, "atol": 0.0},
            "response/rewards/mean": {"rtol": 0.0, "atol": 0.15},
        },
    }


def _fixture_scalars() -> dict[str, dict[int, float]]:
    return {
        "distillation/reduced_distillation_kl": {1: 0.35, 2: 0.38},
        "response/rewards/mean": {1: 0.50, 2: 0.42},
        "response/batch_size": {1: 128.0, 2: 128.0},
    }


class TestCompareMetrics:
    def test_metrics_within_tolerance_pass(self) -> None:
        assert compare_metrics(_fixture_golden(), _fixture_scalars(), [1, 2]) == []

    def test_metric_outside_relative_tolerance_fails(self) -> None:
        scalars = _fixture_scalars()
        scalars["distillation/reduced_distillation_kl"] = {1: 0.50, 2: 0.38}
        failures = compare_metrics(_fixture_golden(), scalars, [1, 2])
        assert len(failures) == 1
        assert "reduced_distillation_kl step 1" in failures[0]

    def test_metric_outside_absolute_tolerance_fails(self) -> None:
        scalars = _fixture_scalars()
        scalars["response/rewards/mean"] = {1: 0.61, 2: 0.42}
        failures = compare_metrics(_fixture_golden(), scalars, [1, 2])
        assert len(failures) == 1
        assert "response/rewards/mean step 1" in failures[0]

    def test_missing_tag_fails(self) -> None:
        scalars = _fixture_scalars()
        del scalars["response/rewards/mean"]
        failures = compare_metrics(_fixture_golden(), scalars, [1, 2])
        assert failures == ["missing scalar tag: response/rewards/mean"]

    def test_missing_step_fails(self) -> None:
        scalars = _fixture_scalars()
        scalars["distillation/reduced_distillation_kl"] = {1: 0.35}
        failures = compare_metrics(_fixture_golden(), scalars, [1, 2])
        assert failures == ["missing distillation/reduced_distillation_kl at step 2"]


class TestCheckInvariants:
    def test_invariants_pass(self) -> None:
        assert check_invariants(_fixture_golden(), _fixture_scalars()) == []

    def test_extra_train_step_fails(self) -> None:
        scalars = _fixture_scalars()
        scalars["response/batch_size"] = {1: 128.0, 2: 128.0, 3: 128.0}
        failures = check_invariants(_fixture_golden(), scalars)
        assert len(failures) == 1
        assert "steps 1..2" in failures[0]

    def test_wrong_batch_size_fails(self) -> None:
        scalars = _fixture_scalars()
        scalars["response/batch_size"] = {1: 64.0, 2: 128.0}
        failures = check_invariants(_fixture_golden(), scalars)
        assert len(failures) == 1
        assert "response/batch_size at step 1: 64 != 128" in failures[0]


class TestReadScalars:
    def test_reads_written_scalars(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "exp_tracking"
        _write_scalars(log_dir, {"a/tag": {1: 1.5, 2: 2.5}})
        assert read_scalars(log_dir) == {"a/tag": {1: 1.5, 2: 2.5}}

    def test_find_exp_tracking_picks_latest(self, tmp_path: Path) -> None:
        _write_scalars(tmp_path / "run_a" / "logs" / "exp_tracking", {"t": {1: 1.0}})
        _write_scalars(tmp_path / "run_b" / "logs" / "exp_tracking", {"t": {1: 1.0}})
        assert find_exp_tracking(tmp_path) == tmp_path / "run_b" / "logs" / "exp_tracking"

    def test_find_exp_tracking_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_exp_tracking(tmp_path)


class TestCheckRun:
    def test_check_run_end_to_end(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "smoke-run"
        _write_scalars(run_dir / "20260101000000" / "logs" / "exp_tracking", _fixture_scalars())
        assert check_run(run_dir, _fixture_golden()) == []

    def test_check_run_reports_each_failure_once(self, tmp_path: Path) -> None:
        scalars = _fixture_scalars()
        scalars["response/rewards/mean"] = {1: 0.99, 2: 0.42}
        run_dir = tmp_path / "smoke-run"
        _write_scalars(run_dir / "20260101000000" / "logs" / "exp_tracking", scalars)
        failures = check_run(run_dir, _fixture_golden())
        assert len(failures) == 1
        assert "response/rewards/mean step 1" in failures[0]


class TestSmokeRunMatchesGolden:
    @pytest.mark.parametrize("experiment", ["topk", "mopd"])
    def test_smoke_run_matches_golden(self, experiment: str) -> None:
        run_dir = os.environ.get("OPD_SMOKE_RUN_DIR", "")
        if not run_dir:
            pytest.skip("Set OPD_SMOKE_RUN_DIR to a finished smoke WORK_DIR to enable this check")
        failures = check_run(run_dir, GOLDEN_BY_EXPERIMENT[experiment])
        assert failures == []
