import json
import logging
import os

import numpy as np
from utils.metric_report import publish_comparison_report


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MEMORY_GRADIENT_WARMUP_STEPS = 5
MEMORY_GRADIENT_MIN_SEGMENT_LEN = 8
MEMORY_GRADIENT_POSITIVE_RATIO = 0.65
MEMORY_GRADIENT_MIN_SLOPE_GB = 1e-4
MEMORY_GRADIENT_MIN_REL_DRIFT = 0.00015
# Ignore sub-20MB dips when splitting; avoids treating allocator noise as resume boundaries.
MEMORY_GRADIENT_RESUME_DROP_GB = 0.02
# Require >=500MB swing within a segment before treating noise as a leak.
MEMORY_GRADIENT_MIN_SEGMENT_RANGE_GB = 0.5
# Skip gradient heuristic when per-step baseline drift is already tight vs baseline.
MEMORY_GRADIENT_BASELINE_DRIFT_SKIP_RATIO = 0.05
MEMORY_GRADIENT_MIN_EXTRA_RANGE_GB = 0.1

# RL tracker lines: mini-batch logs vs per-RL-step summary (see rl_trainer._log_step).
RL_STEP_SUMMARY_MARKER = "response/rewards/mean"
RL_PERCENTILE_METRICS: dict[str, int] = {
    "response/response_len/mean": 80,
    "response/rewards/mean": 80,
}

# SFT metrics may opt into percentile aggregation via config:
#   metric: {threshold: 0.2, aggregate: 80}
#   metric: {threshold: 5, method: absolute}
# Without ``aggregate``, behavior stays step-by-step drift vs baseline.
SFT_METRIC_DEFAULT_METHOD: dict[str, str] = {
    "runtime_info/text_tokens": "absolute",
}


def _normalize_sft_metric_cfg(metric: str, value) -> tuple[float, int | None, str]:
    """Accept ``metric: threshold`` or ``metric: {threshold, aggregate, method}``.

    Percentile aggregation is opt-in only (explicit ``aggregate`` in config).
    Comparison method defaults to ``relative``; token counts default to ``absolute``.
    """
    if isinstance(value, dict):
        if "threshold" not in value:
            raise ValueError(f"SFT check_metrics[{metric}] dict must include 'threshold'")
        threshold = float(value["threshold"])
        aggregate = value.get("aggregate")
        method = str(value.get("method", "relative"))
        return threshold, int(aggregate) if aggregate is not None else None, method

    threshold = float(value)
    method = SFT_METRIC_DEFAULT_METHOD.get(metric, "relative")
    return threshold, None, method


def extract_value(file, metrics):
    metric_all = {metric: [] for metric in metrics}
    total_step = 0
    with open(file) as f:
        for line in f:
            line = json.loads(line)
            for metric in metrics:
                if metric in line:
                    metric_all[metric].append(line[metric])
            total_step += 1

    return total_step, metric_all


def extract_rl_value(file, metrics):
    """Extract metrics from RL step-summary lines only (ignore mini-batch
    rows)."""
    metric_all = {metric: [] for metric in metrics}
    total_step = 0
    with open(file) as f:
        for line in f:
            record = json.loads(line)
            if RL_STEP_SUMMARY_MARKER not in record:
                continue
            total_step += 1
            for metric in metrics:
                if metric in record:
                    metric_all[metric].append(record[metric])
    return total_step, metric_all


def _align_first_phase_steps(phase, base_steps, cur_steps, cur_metrics, kind="SFT"):
    """Offline first+resume may append into one tracker; CI uses a snapshot.

    For ``phase=first``, compare only the baseline-length prefix when the
    current tracker is longer. Shorter current trackers still fail.
    """
    if phase == "first" and cur_steps > base_steps:
        logger.warning(
            f"phase=first: current {kind} tracker has {cur_steps} steps vs baseline "
            f"{base_steps}; using first-run prefix (likely merged first+resume offline)."
        )
        trimmed = {metric: values[:base_steps] for metric, values in cur_metrics.items()}
        return base_steps, trimmed
    return cur_steps, cur_metrics


def _resolve_first_phase_baseline_steps(resume_base_path: str) -> int | None:
    """Return step count of companion ``tracker.jsonl`` for a resume baseline."""
    if not resume_base_path.endswith("tracker-resume.jsonl"):
        return None
    first_path = resume_base_path[: -len("tracker-resume.jsonl")] + "tracker.jsonl"
    if not os.path.isfile(first_path):
        return None
    with open(first_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _align_resume_phase_steps(
    phase,
    base_steps,
    cur_steps,
    cur_metrics,
    kind="SFT",
    *,
    first_base_steps: int | None = None,
    resume_base_path: str | None = None,
):
    """Resume CI runs append into the same exp tracker as the first phase.

    Compare ``cur_metrics[first_base_steps:]`` against ``tracker-resume.jsonl``,
    not the trailing ``base_steps`` rows (which can include first-phase tail).
    """
    if phase != "resume" or cur_steps <= base_steps:
        return cur_steps, cur_metrics

    if first_base_steps is None and resume_base_path:
        first_base_steps = _resolve_first_phase_baseline_steps(resume_base_path)

    if first_base_steps is not None and cur_steps > first_base_steps:
        resume_steps = cur_steps - first_base_steps
        logger.warning(
            f"phase=resume: current {kind} tracker has {cur_steps} steps "
            f"(first baseline {first_base_steps} + resume {resume_steps}) vs resume baseline "
            f"{base_steps}; using post-checkpoint suffix from index {first_base_steps}."
        )
        trimmed = {metric: values[first_base_steps:] for metric, values in cur_metrics.items()}
        return resume_steps, trimmed

    if cur_steps > base_steps:
        logger.warning(
            f"phase=resume: current {kind} tracker has {cur_steps} steps vs baseline "
            f"{base_steps}; first-phase baseline length unknown, using trailing {base_steps} rows."
        )
        trimmed = {metric: values[-base_steps:] for metric, values in cur_metrics.items()}
        return base_steps, trimmed
    return cur_steps, cur_metrics


def _step_errors(base_vals: list[float], cur_vals: list[float], method: str) -> list[float]:
    """Compute per-step quantities compared against ``threshold``.

    - ``absolute`` / ``relative``: drift vs baseline
    - ``slowdown``: only penalize regressions (``max(0, cur - base)``); faster is OK
    - ``value``: current metric itself (baseline ignored); used for bounds like
      ``mismatch_k3_kl < 0.001`` on every step
    """
    errors: list[float] = []
    if method == "value":
        return list(cur_vals)

    for base_val, cur_val in zip(base_vals, cur_vals):
        if method == "absolute":
            errors.append(abs(cur_val - base_val))
        elif method == "slowdown":
            errors.append(max(0.0, cur_val - base_val))
        elif method == "relative":
            if abs(base_val) < 1e-10:
                errors.append(float("inf") if abs(cur_val) > 1e-10 else 0.0)
            else:
                errors.append(abs(cur_val - base_val) / abs(base_val))
        else:
            raise ValueError(f"Unknown method: {method}")
    return errors


def _percentile_error_passes(
    base_vals: list[float],
    cur_vals: list[float],
    *,
    method: str,
    threshold: float,
    operator: str,
    percentile: int,
) -> tuple[bool, float, str]:
    errors = _step_errors(base_vals, cur_vals, method)
    agg_error = float(np.percentile(errors, percentile))
    if operator == "<":
        passed = agg_error < threshold
    elif operator == "<=":
        passed = agg_error <= threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")
    detail = f"p{percentile}={agg_error:.6f} (max={max(errors):.6f})"
    return passed, agg_error, detail


def _format_rl_metric_failure(
    metric: str,
    *,
    method: str,
    operator: str,
    threshold: float,
    detail: str,
) -> str:
    return (
        f"{metric} aggregated error does not satisfy threshold {threshold} "
        f"(method: {method}, operator: {operator}, {detail})"
    )


def _split_memory_segments(values: np.ndarray) -> list[np.ndarray]:
    if len(values) < MEMORY_GRADIENT_MIN_SEGMENT_LEN:
        return [values]

    segments: list[np.ndarray] = []
    start = 0
    for idx in range(1, len(values)):
        dropped = values[idx - 1] - values[idx]
        if dropped >= MEMORY_GRADIENT_RESUME_DROP_GB:
            if idx - start >= MEMORY_GRADIENT_MIN_SEGMENT_LEN:
                segments.append(values[start:idx])
            start = idx
    if len(values) - start >= MEMORY_GRADIENT_MIN_SEGMENT_LEN:
        segments.append(values[start:])
    return segments or [values]


def _should_run_memory_gradient_check(
    base_vals: list[float],
    cur_vals: list[float],
    *,
    drift_threshold: float,
    max_rel_drift: float,
) -> bool:
    """Run leak heuristic only when baseline drift is loose or current swing grew."""
    tight_vs_baseline = max_rel_drift < drift_threshold * MEMORY_GRADIENT_BASELINE_DRIFT_SKIP_RATIO
    base_range = float(max(base_vals) - min(base_vals))
    cur_range = float(max(cur_vals) - min(cur_vals))
    range_grew = (cur_range - base_range) >= MEMORY_GRADIENT_MIN_EXTRA_RANGE_GB
    return not tight_vs_baseline or range_grew


def detect_memory_upward_gradient(values: list[float]) -> tuple[bool, str]:
    """Detect sustained upward memory drift (possible leak) in the current
    run."""
    if len(values) <= MEMORY_GRADIENT_WARMUP_STEPS + MEMORY_GRADIENT_MIN_SEGMENT_LEN:
        return False, ""

    series = np.asarray(values[MEMORY_GRADIENT_WARMUP_STEPS:], dtype=float)

    for seg_idx, segment in enumerate(_split_memory_segments(series)):
        if len(segment) < MEMORY_GRADIENT_MIN_SEGMENT_LEN:
            continue

        deltas = np.diff(segment)
        positive_ratio = float(np.mean(deltas > 1e-4))
        x = np.arange(len(segment))
        slope, _ = np.polyfit(x, segment, 1)
        mean_val = float(np.mean(segment))
        if mean_val < 1e-10:
            continue

        segment_range = float(np.max(segment) - np.min(segment))
        if segment_range < MEMORY_GRADIENT_MIN_SEGMENT_RANGE_GB:
            continue

        relative_drift = float(slope * (len(segment) - 1) / mean_val)
        slope_rising = slope > MEMORY_GRADIENT_MIN_SLOPE_GB
        mostly_increasing = positive_ratio >= MEMORY_GRADIENT_POSITIVE_RATIO
        drift_too_large = relative_drift > MEMORY_GRADIENT_MIN_REL_DRIFT

        if slope_rising and mostly_increasing and drift_too_large:
            return True, (
                f"segment {seg_idx}: slope={slope:.6f} GB/step, "
                f"relative_drift={relative_drift:.4f}, positive_ratio={positive_ratio:.2f}"
            )
    return False, ""


def _format_sft_drift_failure(
    metric: str, idx: int, old: float, cur: float, error: float, method: str, threshold: float
) -> str:
    kind = "absolute error" if method == "absolute" else "relative error"
    return (
        f"{metric} {kind} bigger than {threshold} in {idx} steps, "
        f"baseline: {old:.6f}, now: {cur:.6f}, {kind}: {error:.6f}"
    )


def _check_sft_step_drift(
    base_vals: list[float],
    cur_vals: list[float],
    *,
    threshold: float,
    method: str,
) -> tuple[bool, float, int, float | None]:
    """Return (passed, max_error, max_error_idx, failing_error)."""
    max_error = 0.0
    max_error_idx = 0
    for idx, (old, cur) in enumerate(zip(base_vals, cur_vals)):
        error = _step_errors([old], [cur], method)[0]
        if error > max_error:
            max_error = error
            max_error_idx = idx
        if error > threshold:
            return False, max_error, idx, error
    return True, max_error, max_error_idx, None


def check_result(case_name, base_path, cur_path, check_metric, phase=None):
    fail_metric = {}
    metric_cfgs = {metric: _normalize_sft_metric_cfg(metric, value) for metric, value in check_metric.items()}
    metric_list = list(metric_cfgs.keys())
    threshold_dict = {metric: cfg[0] for metric, cfg in metric_cfgs.items()}
    base_steps, base_metrics = extract_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_value(cur_path, metric_list)
    cur_steps, cur_metrics = _align_first_phase_steps(phase, base_steps, cur_steps, cur_metrics, kind="SFT")
    cur_steps, cur_metrics = _align_resume_phase_steps(
        phase,
        base_steps,
        cur_steps,
        cur_metrics,
        kind="SFT",
        resume_base_path=base_path if phase == "resume" else None,
    )
    assert cur_steps == base_steps, (
        f"current steps is not equal to base steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

    publish_comparison_report(case_name, threshold_dict, base_metrics, cur_metrics, base_path, cur_path, phase=phase)

    for metric, (threshold, percentile, method) in metric_cfgs.items():
        max_error = 0.0
        max_error_idx = 0
        check_flag = True
        if metric == "runtime_info/tgs":
            if cur_steps > 10:
                base_vals = np.array(base_metrics[metric][10:-1], dtype=float)
                cur_vals = np.array(cur_metrics[metric][10:-1], dtype=float)
                degradation = np.zeros_like(base_vals, dtype=float)
                valid_base = np.abs(base_vals) >= 1e-10
                degradation[valid_base] = np.maximum(
                    (base_vals[valid_base] - cur_vals[valid_base]) / np.abs(base_vals[valid_base]),
                    0.0,
                )
                max_error = float(np.percentile(degradation, 80))
                if max_error > threshold:
                    fail_metric[metric] = (
                        f"{metric} degradation bigger than {threshold} after step 10, "
                        f"baseline: {base_metrics[metric][10:-1]}, now: {cur_metrics[metric][10:-1]}, "
                        f"degradation: {degradation.tolist()}"
                    )
                    check_flag = False
                else:
                    check_flag = True
            else:
                logger.warning("It's meaningless to compare tgs because of the small steps.")
                check_flag = False
        elif metric == "memory/max_memory_GB":
            check_flag, max_error, max_error_idx, failing_error = _check_sft_step_drift(
                base_metrics[metric],
                cur_metrics[metric],
                threshold=threshold,
                method="relative",
            )
            if not check_flag:
                old = base_metrics[metric][max_error_idx]
                cur = cur_metrics[metric][max_error_idx]
                fail_metric[metric] = _format_sft_drift_failure(
                    metric, max_error_idx, old, cur, failing_error, "relative", threshold
                )

            if check_flag:
                run_gradient = _should_run_memory_gradient_check(
                    base_metrics[metric],
                    cur_metrics[metric],
                    drift_threshold=threshold,
                    max_rel_drift=max_error,
                )
                if run_gradient:
                    has_gradient, gradient_info = detect_memory_upward_gradient(cur_metrics[metric])
                    if has_gradient:
                        fail_metric[metric] = (
                            f"{metric} shows sustained upward gradient in current run, {gradient_info}"
                        )
                        check_flag = False
                else:
                    logger.info(
                        f"✓ {metric} gradient check skipped (max rel drift {max_error:.2%} vs threshold {threshold})"
                    )
        elif percentile is not None:
            agg_method = method if method in ("absolute", "relative") else "relative"
            check_flag, agg_error, detail = _percentile_error_passes(
                base_metrics[metric],
                cur_metrics[metric],
                method=agg_method,
                threshold=threshold,
                operator="<",
                percentile=percentile,
            )
            if not check_flag:
                fail_metric[metric] = f"{metric} {agg_method} error bigger than {threshold} ({detail})"
            else:
                logger.info(f"✓ {metric} check pass，{detail}, threshold={threshold}")
            continue
        else:
            check_flag, max_error, max_error_idx, failing_error = _check_sft_step_drift(
                base_metrics[metric],
                cur_metrics[metric],
                threshold=threshold,
                method=method,
            )
            if not check_flag:
                old = base_metrics[metric][max_error_idx]
                cur = cur_metrics[metric][max_error_idx]
                fail_metric[metric] = _format_sft_drift_failure(
                    metric, max_error_idx, old, cur, failing_error, method, threshold
                )
        if check_flag:
            if method == "absolute":
                logger.info(
                    f"✓ {metric} check pass，the most absolute error is {max_error:.6f} in {max_error_idx} step."
                )
            else:
                logger.info(
                    f"✓ {metric} check pass，the most relative error is {max_error:.2%} in {max_error_idx} step."
                )
    result = not fail_metric
    if result:
        return result, "All metrics check passed."
    return result, f"Some metric check failed: {fail_metric}"


def check_rl_result(case_name, base_path, cur_path, assert_info, phase=None):
    fail_metric = {}
    check_metrics_list = assert_info["check_metrics"]

    metric_list = [item["metric"] for item in check_metrics_list]

    base_steps, base_metrics = extract_rl_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_rl_value(cur_path, metric_list)
    cur_steps, cur_metrics = _align_first_phase_steps(phase, base_steps, cur_steps, cur_metrics, kind="RL")
    first_base_steps = None
    if phase == "resume":
        first_path = base_path.replace("tracker-resume.jsonl", "tracker.jsonl")
        if os.path.isfile(first_path):
            first_base_steps, _ = extract_rl_value(first_path, metric_list[:1])
    cur_steps, cur_metrics = _align_resume_phase_steps(
        phase,
        base_steps,
        cur_steps,
        cur_metrics,
        kind="RL",
        first_base_steps=first_base_steps,
        resume_base_path=base_path if phase == "resume" else None,
    )

    assert cur_steps == base_steps, (
        f"current RL steps is not equal to base RL steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

    check_metric_dict = {item["metric"]: item["threshold"] for item in check_metrics_list}
    publish_comparison_report(
        case_name, check_metric_dict, base_metrics, cur_metrics, base_path, cur_path, phase=phase
    )

    for config in check_metrics_list:
        metric = config["metric"]
        threshold = config["threshold"]
        method = config["method"]
        operator = config["operator"]
        percentile = config.get("aggregate")
        if percentile is None and metric in RL_PERCENTILE_METRICS:
            percentile = RL_PERCENTILE_METRICS[metric]

        base_vals = base_metrics[metric]
        cur_vals = cur_metrics[metric]
        if not base_vals and not cur_vals:
            logger.warning(f"Skip {metric}: absent in both baseline and current RL step summaries.")
            continue
        if len(base_vals) != len(cur_vals):
            fail_metric[metric] = (
                f"{metric} step count mismatch after RL step-summary extraction: "
                f"baseline={len(base_vals)}, current={len(cur_vals)}"
            )
            continue

        max_error = 0.0
        max_error_idx = 0
        check_flag = True

        if percentile is not None:
            check_flag, agg_error, detail = _percentile_error_passes(
                base_vals,
                cur_vals,
                method=method,
                threshold=threshold,
                operator=operator,
                percentile=int(percentile),
            )
            if not check_flag:
                fail_metric[metric] = _format_rl_metric_failure(
                    metric,
                    method=method,
                    operator=operator,
                    threshold=threshold,
                    detail=detail,
                )
            else:
                logger.info(
                    f"✓ {metric} check passed ({detail}, method: {method}, operator: {operator}, "
                    f"threshold: {threshold})"
                )
            continue

        for idx, (base_val, cur_val) in enumerate(zip(base_vals, cur_vals)):
            errors = _step_errors([base_val], [cur_val], method)
            error = round(errors[0], 5)

            if error > max_error:
                max_error = error
                max_error_idx = idx

            if operator == "<":
                passed = error < threshold
            elif operator == "<=":
                passed = error <= threshold
            else:
                raise ValueError(f"Unknown operator: {operator}")

            if not passed:
                if method == "value":
                    fail_metric[metric] = (
                        f"{metric} value {cur_val:.6f} does not satisfy {operator} {threshold} at step {idx}"
                    )
                else:
                    fail_metric[metric] = (
                        f"{metric} error {error:.6f} not less than threshold {threshold} "
                        f"(method: {method}, operator: {operator}) at step {idx}, "
                        f"baseline: {base_val:.6f}, current: {cur_val:.6f}"
                    )
                check_flag = False
                break

        if check_flag:
            if method == "value":
                logger.info(
                    f"✓ {metric} check passed, max value is {max_error:.6f} at step {max_error_idx} "
                    f"(operator: {operator}, threshold: {threshold})"
                )
            else:
                logger.info(
                    f"✓ {metric} check passed, max error is {max_error:.6f} at step {max_error_idx} "
                    f"(method: {method}, operator: {operator})"
                )

    result = not bool(fail_metric)
    if result:
        return result, "All metrics check passed."
    return result, f"Some metric check failed: {fail_metric}"


if __name__ == "__main__":
    print(
        check_result(
            "qwen3-sft",
            "./base/tracker.jsonl",
            "./current/tracker.jsonl",
            {
                "grad_norm": 0.000001,
                "loss/reduced_llm_loss": 0.000001,
                "lr": 0,
                "memory/max_memory_GB": 0.2,
                "runtime_info/tgs": 0.05,
                "runtime_info/text_tokens": 0,
            },
        )
    )
